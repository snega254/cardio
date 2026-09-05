"""
gradcam_inception.py

Grad-CAM for Inception1D. This model has no single "last conv layer"
— its last block (InceptionModule1D) runs 4 parallel branches (kernel
sizes 9/19/39 + a pooled branch) and concatenates them. We hook that
whole module's output (post-concat, post-BN, post-activation), which
is the last point before global average pooling.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.ecg.models.inception1d import Inception1D


class GradCAMInception1D:
    def __init__(self, model: Inception1D):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = model.blocks[-1]  # last InceptionModule1D
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self, signal: torch.Tensor, target_class: int = None
    ) -> Tuple[np.ndarray, int, float]:
        self.model.eval()
        signal = signal.requires_grad_(True)

        logits = self.model(signal)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, target_class].item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = cam.unsqueeze(1)
        cam = F.interpolate(
            cam, size=signal.shape[-1], mode="linear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, confidence


def compute_gradcam_inception(
    model: Inception1D, signal: torch.Tensor, target_class: int = None
) -> dict:
    cam_generator = GradCAMInception1D(model)
    cam, pred_class, confidence = cam_generator.generate(signal, target_class)
    return {"cam": cam, "class": pred_class, "confidence": confidence}