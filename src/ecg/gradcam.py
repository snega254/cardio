"""
gradcam.py

Grad-CAM explainability — ONLY applies to the custom CNN1D (it does
not apply to the RAG retrieval or LLM reasoning components, and it's
not meaningful for the plain ResNet1D/Inception1D baselines in this
project's scope).

Produces a 1D "importance map" over the time axis showing which
time-segments most influenced the model's prediction — this is the
signal-side counterpart to Member 3's evidence-citation explainability
on the text side.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.ecg.models.cnn1d import CNN1D


class GradCAM1D:
    """
    Hooks the last conv block of CNN1D (model.features[-1]) to capture
    forward activations and backward gradients, then combines them
    into a class-discriminative importance map over time.
    """

    def __init__(self, model: CNN1D):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = model.features[-1].conv  # last conv layer
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self, signal: torch.Tensor, target_class: int = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Args:
            signal: (1, in_channels, n_samples) — single example, batch dim = 1
            target_class: class index to explain; if None, uses the
                model's own predicted class.

        Returns:
            cam: 1D numpy array of length n_samples, values in [0, 1],
                 upsampled to match the input signal length.
            predicted_class: int
            confidence: float, softmax probability of predicted_class
        """
        self.model.eval()
        signal = signal.requires_grad_(True)

        logits = self.model(signal)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        # global-average-pool the gradients over time -> per-channel weight
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (weights * self.activations).sum(dim=1)  # (1, T')
        cam = F.relu(cam)

        # upsample from feature-map length back to original signal length
        cam = cam.unsqueeze(1)  # (1, 1, T')
        cam = F.interpolate(
            cam, size=signal.shape[-1], mode="linear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()

        # normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence


def compute_gradcam(
    model: CNN1D, signal: torch.Tensor, target_class: int = None
) -> dict:
    """
    Convenience wrapper used by interface.py.

    Args:
        signal: (1, in_channels, n_samples) tensor.

    Returns:
        dict with keys: 'cam' (np.ndarray), 'class' (int), 'confidence' (float)
    """
    cam_generator = GradCAM1D(model)
    cam, pred_class, confidence = cam_generator.generate(signal, target_class)
    return {"cam": cam, "class": pred_class, "confidence": confidence}
