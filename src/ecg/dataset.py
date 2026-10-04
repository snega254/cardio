"""
dataset.py

PyTorch Dataset wrapping PTB-XL, using the dataset's OFFICIAL
`strat_fold` column for splitting (not a custom random split), as
required by the project spec:

  folds 1-8   -> train
  fold  9     -> validation
  fold  10    -> test

This matches the standard PTB-XL benchmark protocol, which is what
lets our results be compared against published baselines.

UPDATED:
  - Signals are now trained on 2.5 s windows (250 samples @ 100 Hz)
    instead of full 10 s (1000 samples), because the ECG-Digitiser
    only produces 2.5 s per lead for 4x4 layout ECG images.
  - Training uses a RANDOM 2.5 s crop per epoch (data augmentation),
    so the model becomes robust to "which 2.5 s window it sees".
  - Validation and test use the FIRST 2.5 s crop deterministically
    (so metrics are reproducible).
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ecg.data_loader import (
    SUPERCLASSES,
    add_superclass_labels,
    load_metadata,
    load_scp_statements,
    load_signals_for_split,
)
from src.ecg.preprocessing import preprocess_batch

LABEL_TO_IDX = {label: i for i, label in enumerate(SUPERCLASSES)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLDS = [9]
TEST_FOLDS = [10]

# Model input window (matches ECG-Digitiser output for 4x4 images)
WINDOW_SAMPLES = 250       # 2.5 s @ 100 Hz
FULL_SAMPLES = 1000        # 10 s @ 100 Hz (raw PTB-XL record length)


class PTBXLDataset(Dataset):
    """
    Loads and preprocesses PTB-XL signals for one split ('train',
    'val', or 'test'), holding preprocessed tensors in memory.

    Each __getitem__ returns a (8, 250) tensor — 8 leads, 2.5 s window
    at 100 Hz — to match the ECG-Digitiser's output from 4x4-layout
    ECG images.

    - Train split: random 2.5 s crop each call (data augmentation)
    - Val / test split: fixed first 2.5 s crop (reproducible)
    """

    def __init__(
        self,
        ptbxl_root: str,
        split: str = "train",
        sampling_rate: int = 100,
    ):
        assert split in ("train", "val", "test")
        self.ptbxl_root = ptbxl_root
        self.split = split
        self.sampling_rate = sampling_rate

        meta = load_metadata(ptbxl_root)
        agg = load_scp_statements(ptbxl_root)
        labeled = add_superclass_labels(meta, agg)

        fold_map = {"train": TRAIN_FOLDS, "val": VAL_FOLDS, "test": TEST_FOLDS}
        folds = fold_map[split]
        self.df = labeled[labeled["strat_fold"].isin(folds)]

        raw_signals = load_signals_for_split(
            self.df, ptbxl_root, sampling_rate=sampling_rate
        )
        self.signals = preprocess_batch(raw_signals, fs=sampling_rate)  # (N, 1000, 8)
        self.labels = np.array(
            [LABEL_TO_IDX[label] for label in self.df["label"]]
        )

        # Sanity check: raw signals should be ~10 s at this sampling rate
        assert self.signals.shape[1] >= WINDOW_SAMPLES, (
            f"Raw signals have {self.signals.shape[1]} samples, "
            f"expected at least {WINDOW_SAMPLES}."
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]                     # (n_samples, 8), n_samples >= 250

        # Choose a 2.5 s window
        max_start = signal.shape[0] - WINDOW_SAMPLES
        if self.split == "train":
            start = np.random.randint(0, max_start + 1)   # random crop (augmentation)
        else:
            start = 0                                     # deterministic first crop

        window = signal[start:start + WINDOW_SAMPLES]     # (250, 8)

        # PyTorch 1D-conv expects (channels, length)
        window = torch.from_numpy(window.T).float()       # (8, 250)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return window, label

    @property
    def num_classes(self) -> int:
        return len(SUPERCLASSES)

    @property
    def input_channels(self) -> int:
        return self.signals.shape[2]