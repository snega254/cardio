"""
inception1d.py

Inception1D baseline — multi-scale kernel architecture (parallel
convolutions of different kernel sizes per block), the second
published baseline the custom CNN is compared against.
"""

import torch
import torch.nn as nn


class InceptionModule1D(nn.Module):
    """
    One Inception block: parallel branches with kernel sizes
    9 / 19 / 39 plus a pooled branch, concatenated on the channel dim.
    Kernel sizes are odd multiples chosen to capture short (QRS-scale)
    and long (ST/T-scale) morphology simultaneously.
    """

    def __init__(self, in_ch, out_ch_per_branch=32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, out_ch_per_branch, 1)

        self.branch9 = nn.Conv1d(
            out_ch_per_branch, out_ch_per_branch, 9, padding=4
        )
        self.branch19 = nn.Conv1d(
            out_ch_per_branch, out_ch_per_branch, 19, padding=9
        )
        self.branch39 = nn.Conv1d(
            out_ch_per_branch, out_ch_per_branch, 39, padding=19
        )

        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch_per_branch, 1),
        )

        self.bn = nn.BatchNorm1d(out_ch_per_branch * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        b1 = self.branch9(bottleneck)
        b2 = self.branch19(bottleneck)
        b3 = self.branch39(bottleneck)
        b4 = self.pool_branch(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.act(self.bn(out))


class Inception1D(nn.Module):
    """
    Stack of Inception1D modules with residual shortcuts every 3
    blocks (as in the original InceptionTime design), global average
    pool, linear classifier.
    """

    def __init__(self, in_channels: int = 8, num_classes: int = 5, depth: int = 6):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        ch = in_channels
        block_out_ch = 32 * 4  # 4 branches * 32 channels each
        for i in range(depth):
            self.blocks.append(InceptionModule1D(ch, out_ch_per_branch=32))
            ch = block_out_ch
            if (i + 1) % 3 == 0:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels if i < 3 else block_out_ch, block_out_ch, 1),
                        nn.BatchNorm1d(block_out_ch),
                    )
                )
        self.depth = depth
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(block_out_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        out = x
        shortcut_idx = 0
        for i, block in enumerate(self.blocks):
            out = block(out)
            if (i + 1) % 3 == 0:
                shortcut = self.shortcuts[shortcut_idx](residual_input)
                out = torch.relu(out + shortcut)
                residual_input = out
                shortcut_idx += 1
        pooled = self.global_pool(out).flatten(1)
        return self.classifier(pooled)
