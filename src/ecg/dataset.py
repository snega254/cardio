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


class PTBXLDataset(Dataset):
    """
    Loads and preprocesses PTB-XL signals for one split ('train',
    'val', or 'test'), holding preprocessed tensors in memory.

    For very large sampling rates (500Hz) consider lazy-loading in
    __getitem__ instead of eager-loading in __init__ if memory becomes
    an issue.
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
        self.signals = preprocess_batch(raw_signals, fs=sampling_rate)
        self.labels = np.array(
            [LABEL_TO_IDX[label] for label in self.df["label"]]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]  # (n_samples, 8)
        # PyTorch 1D-conv expects (channels, length)
        signal = torch.from_numpy(signal.T).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

    @property
    def num_classes(self) -> int:
        return len(SUPERCLASSES)

    @property
    def input_channels(self) -> int:
        return self.signals.shape[2]
