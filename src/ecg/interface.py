"""
interface.py

THIS is the file the rest of the team depends on. Keep the function
signature stable — src/pipeline.py (integration phase) will import
and call predict() directly.

    predict(signal) -> class, confidence, gradcam_map

UPDATED: predict() now uses ONLY Inception1D (the confirmed best model).
No fallback chain — if Inception1D fails, the error is raised directly
so it's easy to diagnose.

CONFIGURATION:
    ECG_CHECKPOINT_DIR  — directory containing inception1d_best.pt
                          (default: "checkpoints")
    ECG_MODEL_PATH      — (optional) explicit path to the .pt file;
                          if set, it overrides the default location.
"""

import os
from functools import lru_cache
from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ecg.dataset import IDX_TO_LABEL
from src.ecg.gradcam_dispatch import compute_gradcam_any
from src.ecg.models.inception1d import Inception1D
from src.ecg.preprocessing import preprocess_signal

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Checkpoint directory (env-configurable)
CHECKPOINT_DIR = os.environ.get("ECG_CHECKPOINT_DIR", "checkpoints")

# Explicit single-model override (optional)
EXPLICIT_MODEL_PATH = os.environ.get("ECG_MODEL_PATH", None)

NUM_CLASSES = 5
IN_CHANNELS = 8

# Only one model now.
MODEL_NAME = "inception1d"

MODEL_REGISTRY = {
    "inception1d": Inception1D,
}


# --------------------------------------------------
# MODEL PATH RESOLUTION
# --------------------------------------------------

def _resolve_checkpoint_path(model_name: str) -> str:
    """
    Resolve the checkpoint file path for a given model name.

    Priority:
        1. ECG_MODEL_PATH env var (if set and file exists)
        2. CHECKPOINT_DIR/{model_name}_best.pt
        3. CHECKPOINT_DIR/{model_name}.pt
    """
    # Priority 1: explicit model path from env
    if EXPLICIT_MODEL_PATH and Path(EXPLICIT_MODEL_PATH).exists():
        return EXPLICIT_MODEL_PATH

    # Priority 2: standard naming
    candidate = Path(CHECKPOINT_DIR) / f"{model_name}_best.pt"
    if candidate.exists():
        return str(candidate)

    # Priority 3: alternate naming
    candidate = Path(CHECKPOINT_DIR) / f"{model_name}.pt"
    if candidate.exists():
        return str(candidate)

    # Nothing found — return the standard path so the error message is useful
    return str(Path(CHECKPOINT_DIR) / f"{model_name}_best.pt")


@lru_cache(maxsize=None)
def _load_model(model_name: str = MODEL_NAME):
    """
    Loads and caches Inception1D. Cached so it's only loaded from disk once.
    """
    checkpoint_path = _resolve_checkpoint_path(model_name)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"No checkpoint found for '{model_name}'. "
            f"Looked in '{CHECKPOINT_DIR}'. "
            f"Set ECG_CHECKPOINT_DIR or ECG_MODEL_PATH in .env."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle checkpoints that wrap state_dict inside a dict
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


# --------------------------------------------------
# PUBLIC API
# --------------------------------------------------

def predict(
    signal: np.ndarray,
    already_preprocessed: bool = False,
    return_gradcam: bool = True,
    return_model_used: bool = False,
    return_probabilities: bool = False,
) -> Union[Tuple[str, float, Optional[np.ndarray]], Tuple[str, float, Optional[np.ndarray], str]]:
    """
    The single public entry point.

    Args:
        signal: raw ECG signal, shape (n_samples, 12) — the standard
            12-lead PTB-XL layout — UNLESS already_preprocessed=True,
            in which case it should already be (n_samples, 8) and
            filtered/normalized.
        already_preprocessed: set True to skip preprocessing.py.
        return_gradcam: if False, skips Grad-CAM computation for speed.
        return_model_used: if True, returns a 4th value naming the model
            that produced this prediction (always "inception1d" now).

    Returns:
        (predicted_class, confidence, gradcam_map) by default, or
        (predicted_class, confidence, gradcam_map, model_used) if
        return_model_used=True.

        predicted_class: str, one of NORM / MI / STTC / CD / HYP
        confidence: float in [0, 1]
        gradcam_map: np.ndarray of length n_samples, or None
        model_used: str, always "inception1d"

    Raises RuntimeError if Inception1D fails to load or run.
    """
    if not already_preprocessed:
        processed = preprocess_signal(signal)
    else:
        processed = signal

    try:
        model = _load_model(MODEL_NAME)
        device = next(model.parameters()).device

        # (n_samples, 8) -> (1, 8, n_samples)
        tensor = torch.from_numpy(processed.T).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            predicted_idx_from_probs = int(probs.argmax(dim=1).item())
            confidence_from_probs = float(probs[0, predicted_idx_from_probs].item())

        if return_gradcam:
            result = compute_gradcam_any(MODEL_NAME, model, tensor)
            predicted_idx = result["class"]
            confidence = result["confidence"]
            gradcam_map = result["cam"]
        else:
            predicted_idx = predicted_idx_from_probs
            confidence = confidence_from_probs
            gradcam_map = None

        predicted_class = IDX_TO_LABEL[predicted_idx]

        if return_probabilities:
            probability_array = probs[0].detach().cpu().numpy()
            if return_model_used:
                return predicted_class, confidence, gradcam_map, MODEL_NAME, probability_array
            return predicted_class, confidence, gradcam_map, probability_array
        if return_model_used:
            return predicted_class, confidence, gradcam_map, MODEL_NAME
        return predicted_class, confidence, gradcam_map

    except Exception as e:
        raise RuntimeError(f"Inception1D prediction failed: {e}")


# --------------------------------------------------
# UTILITY: Check if the model is available
# --------------------------------------------------

def list_available_models() -> dict:
    """List the model and whether its checkpoint exists."""
    path = _resolve_checkpoint_path(MODEL_NAME)
    p = Path(path)
    return {
        MODEL_NAME: {
            "path": path,
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else 0,
        }
    }


# --------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":
    print("Available ECG models:")
    print("=" * 60)
    for name, info in list_available_models().items():
        status = "OK" if info["exists"] else "MISSING"
        print(f"  [{status}] {name}: {info['path']} ({info['size_mb']} MB)")

    print()
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Explicit path: {EXPLICIT_MODEL_PATH or '(none)'}")