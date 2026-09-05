"""
cnn1d.py

The custom 1D CNN — the PRIMARY model for this project (the one
Grad-CAM is applied to).

Design notes:
  - Input: (batch, 8, 1000) for 100Hz/10s records (8 leads after
    reduction, adjust n_samples if using 500Hz).
  - `self.features` is the conv stack; kept as a named attribute so
    gradcam.py can hook the last conv layer's activations/gradients.
  - Global average pooling + linear head for classification.
  - Dropout raised to 0.5 (from 0.3) to address overfitting observed
    during training (val_loss rising while train_loss kept falling
    in later epochs).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, pool=2):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class CNN1D(nn.Module):
    """
    Custom 1D CNN for ECG superclass classification.

    Args:
        in_channels: number of leads (8 after preprocessing).
        num_classes: number of output classes (5 superclasses).
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=15, pool=2),
            ConvBlock(32, 64, kernel_size=11, pool=2),
            ConvBlock(64, 128, kernel_size=9, pool=2),
            ConvBlock(128, 256, kernel_size=7, pool=2),
            ConvBlock(256, 256, kernel_size=5, pool=1),  # last conv block
            # ^ gradcam.py targets this final block's conv output
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, n_samples)
        Returns:
            logits: (batch, num_classes)
        """
        feats = self.features(x)
        pooled = self.global_pool(feats)
        return self.classifier(pooled)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Exposes raw conv feature maps — used by gradcam.py."""
        return self.features(x)