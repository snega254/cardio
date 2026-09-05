"""
interface.py

THIS is the file the rest of the team depends on. Keep the function
signature stable — src/pipeline.py (integration phase) will import
and call predict() directly.

    predict(signal) -> class, confidence, gradcam_map

Do not rename this function or change its return shape without
telling the team, since it's the agreed contract from the blueprint.

UPDATED: predict() now tries the best-performing model first
(Inception1D), falling back to ResNet1D, then CNN1D, if a model
fails to load or run. Grad-CAM is computed using the implementation
that matches whichever model actually produced the prediction — never
a mismatched one. The original 3-value return contract is unchanged;
which model was actually used is available via the optional
return_model_used flag for callers that want it, without breaking
existing callers that only expect (class, confidence, gradcam_map).
"""

import os
from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np
import torch

from src.ecg.dataset import IDX_TO_LABEL
from src.ecg.gradcam_dispatch import compute_gradcam_any
from src.ecg.models.cnn1d import CNN1D
from src.ecg.models.resnet1d import ResNet1D
from src.ecg.models.inception1d import Inception1D
from src.ecg.preprocessing import preprocess_signal

CHECKPOINT_DIR = os.environ.get("ECG_CHECKPOINT_DIR", "checkpoints")
NUM_CLASSES = 5
IN_CHANNELS = 8

# Best -> second -> third, per project owner's decision.
FALLBACK_ORDER = ["inception1d", "resnet1d", "cnn1d"]

MODEL_REGISTRY = {
    "cnn1d": CNN1D,
    "resnet1d": ResNet1D,
    "inception1d": Inception1D,
}


@lru_cache(maxsize=None)
def _load_model(model_name: str):
    """
    Loads and caches one model by name. Cached per model_name, so all
    three can be cached simultaneously across the process lifetime —
    each is only ever loaded from disk once.
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(
    signal: np.ndarray,
    already_preprocessed: bool = False,
    return_gradcam: bool = True,
    return_model_used: bool = False,
) -> Union[Tuple[str, float, Optional[np.ndarray]], Tuple[str, float, Optional[np.ndarray], str]]:
    """
    The single public entry point Members 2/3's pipeline code calls.

    Args:
        signal: raw ECG signal, shape (n_samples, 12) — the standard
            12-lead PTB-XL layout — UNLESS already_preprocessed=True,
            in which case it should already be (n_samples, 8) and
            filtered/normalized.
        already_preprocessed: set True to skip preprocessing.py.
        return_gradcam: if False, skips Grad-CAM computation for speed.
        return_model_used: if True, returns a 4th value naming which
            model actually produced this prediction ("inception1d",
            "resnet1d", or "cnn1d") — OFF by default so existing
            callers expecting the original 3-value contract are
            unaffected.

    Returns:
        (predicted_class, confidence, gradcam_map) by default, or
        (predicted_class, confidence, gradcam_map, model_used) if
        return_model_used=True.

        predicted_class: str, one of NORM / MI / STTC / CD / HYP
        confidence: float in [0, 1]
        gradcam_map: np.ndarray of length n_samples, or None
        model_used: str, which model in the fallback chain actually ran

    Tries models in FALLBACK_ORDER (best -> worst). Raises
    RuntimeError only if every model in the chain fails.
    """
    if not already_preprocessed:
        processed = preprocess_signal(signal)
    else:
        processed = signal

    last_error = None

    for model_name in FALLBACK_ORDER:
        try:
            model = _load_model(model_name)
            device = next(model.parameters()).device

            # (n_samples, 8) -> (1, 8, n_samples)
            tensor = torch.from_numpy(processed.T).float().unsqueeze(0).to(device)

            if return_gradcam:
                result = compute_gradcam_any(model_name, model, tensor)
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

            if return_model_used:
                return predicted_class, confidence, gradcam_map, model_name
            return predicted_class, confidence, gradcam_map

        except Exception as e:
            last_error = e
            print(f"  [interface] {model_name} failed ({e}), trying next fallback...")
            continue

    raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")