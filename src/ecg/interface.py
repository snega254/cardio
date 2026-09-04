"""
interface.py

THIS is the file the rest of the team depends on. Keep the function
signature stable — src/pipeline.py (integration phase) will import
and call predict() directly.

    predict(signal) -> class, confidence, gradcam_map

Do not rename this function or change its return shape without
telling the team, since it's the agreed contract from the blueprint.
"""

import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch

from src.ecg.dataset import IDX_TO_LABEL
from src.ecg.gradcam import compute_gradcam
from src.ecg.models.cnn1d import CNN1D
from src.ecg.preprocessing import preprocess_signal

DEFAULT_CHECKPOINT = os.environ.get(
    "ECG_CHECKPOINT", "checkpoints/cnn1d_best.pt"
)
NUM_CLASSES = 5
IN_CHANNELS = 8


@lru_cache(maxsize=1)
def _load_model(checkpoint_path: str = DEFAULT_CHECKPOINT) -> CNN1D:
    """
    Loads the trained custom CNN once and caches it, so repeated calls
    to predict() during the same process don't reload weights from
    disk every time.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(
    signal: np.ndarray,
    already_preprocessed: bool = False,
    return_gradcam: bool = True,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
) -> Tuple[str, float, Optional[np.ndarray]]:
    """
    The single public entry point Members 2/3's pipeline code calls.

    Args:
        signal: raw ECG signal, shape (n_samples, 12) — the standard
            12-lead PTB-XL layout — UNLESS already_preprocessed=True,
            in which case it should already be (n_samples, 8) and
            filtered/normalized.
        already_preprocessed: set True to skip preprocessing.py (e.g.
            if the caller already did it).
        return_gradcam: if False, skips Grad-CAM computation for speed
            (e.g. batch scoring where explanations aren't needed).
        checkpoint_path: path to the trained CNN1D weights.

    Returns:
        predicted_class: str, one of NORM / MI / STTC / CD / HYP
        confidence: float in [0, 1] — softmax probability of the
            predicted class
        gradcam_map: np.ndarray of length n_samples (importance over
            time, values in [0, 1]), or None if return_gradcam=False
    """
    model = _load_model(checkpoint_path)
    device = next(model.parameters()).device

    if not already_preprocessed:
        processed = preprocess_signal(signal)
    else:
        processed = signal

    # (n_samples, 8) -> (1, 8, n_samples) for the model
    tensor = torch.from_numpy(processed.T).float().unsqueeze(0).to(device)

    if return_gradcam:
        result = compute_gradcam(model, tensor)
        predicted_idx = result["class"]
        confidence = result["confidence"]
        gradcam_map = result["cam"]
    else:
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_idx = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, predicted_idx].item())
        gradcam_map = None

    predicted_class = IDX_TO_LABEL[predicted_idx]
    return predicted_class, confidence, gradcam_map
