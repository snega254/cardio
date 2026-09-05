"""
gradcam_dispatch.py

Single entry point for Grad-CAM regardless of which model produced
the prediction. interface.py calls this one function instead of
knowing which specific Grad-CAM class/module to use.
"""

from src.ecg.gradcam import compute_gradcam
from src.ecg.gradcam_resnet import compute_gradcam_resnet
from src.ecg.gradcam_inception import compute_gradcam_inception


def compute_gradcam_any(model_name: str, model, signal, target_class: int = None) -> dict:
    """
    Args:
        model_name: one of "cnn1d", "resnet1d", "inception1d" — must
            match whichever model actually produced the prediction.
        model: the loaded model instance itself.
        signal: (1, in_channels, n_samples) tensor.
        target_class: optional class index to explain; defaults to
            the model's own predicted class.

    Returns:
        dict with keys: 'cam', 'class', 'confidence'
    """
    if model_name == "cnn1d":
        return compute_gradcam(model, signal, target_class)
    elif model_name == "resnet1d":
        return compute_gradcam_resnet(model, signal, target_class)
    elif model_name == "inception1d":
        return compute_gradcam_inception(model, signal, target_class)
    else:
        raise ValueError(f"Unknown model_name for Grad-CAM: {model_name}")