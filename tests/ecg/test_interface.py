"""
test_interface.py

Tests the Grad-CAM output shape and the predict() contract that the
rest of the team relies on, using an untrained model (random weights)
purely to check shapes/types are correct — not accuracy.
"""

import numpy as np
import torch

from src.ecg.gradcam import compute_gradcam
from src.ecg.models.cnn1d import CNN1D

IN_CHANNELS = 8
N_SAMPLES = 1000
NUM_CLASSES = 5


def test_gradcam_output_shape_and_range():
    model = CNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    x = torch.randn(1, IN_CHANNELS, N_SAMPLES)

    result = compute_gradcam(model, x)

    assert "cam" in result and "class" in result and "confidence" in result
    cam = result["cam"]
    assert cam.shape == (N_SAMPLES,)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert 0 <= result["class"] < NUM_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_contract_shape(tmp_path, monkeypatch):
    """
    Verifies predict() returns (class:str, confidence:float, cam:ndarray)
    using a freshly initialized (untrained) checkpoint saved to a temp
    path — this only checks the CONTRACT, not real accuracy.
    """
    from src.ecg import interface

    model = CNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    ckpt_path = tmp_path / "cnn1d_best.pt"
    torch.save(model.state_dict(), ckpt_path)

    interface._load_model.cache_clear()  # ensure a fresh load for this test

    fake_signal = np.random.randn(N_SAMPLES, 12).astype(np.float32)
    pred_class, confidence, cam = interface.predict(
        fake_signal, checkpoint_path=str(ckpt_path)
    )

    assert pred_class in ["NORM", "MI", "STTC", "CD", "HYP"]
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert cam.shape == (N_SAMPLES,)
