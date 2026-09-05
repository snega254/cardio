"""
test_models.py

Shape/sanity tests for all three architectures — confirms each model
accepts (batch, 8, 1000) input and returns (batch, 5) logits, without
needing any real data or trained weights.
"""

import torch

from src.ecg.models.cnn1d import CNN1D
from src.ecg.models.inception1d import Inception1D
from src.ecg.models.resnet1d import ResNet1D

BATCH_SIZE = 4
IN_CHANNELS = 8
N_SAMPLES = 1000
NUM_CLASSES = 5


def _check_forward(model_cls):
    model = model_cls(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, N_SAMPLES)
    out = model(x)
    assert out.shape == (BATCH_SIZE, NUM_CLASSES)


def test_cnn1d_forward():
    _check_forward(CNN1D)


def test_resnet1d_forward():
    _check_forward(ResNet1D)


def test_inception1d_forward():
    _check_forward(Inception1D)


def test_cnn1d_forward_features_shape():
    model = CNN1D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    x = torch.randn(1, IN_CHANNELS, N_SAMPLES)
    feats = model.forward_features(x)
    # (batch, channels, reduced_time) — just confirm it runs and has 3 dims
    assert feats.dim() == 3
    assert feats.shape[0] == 1
