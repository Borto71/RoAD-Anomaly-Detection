"""
Simple 1D CNN for time-series classification.

This module defines a small, readable baseline you can use on the
RoadDataset (or any 1D signal dataset). The network is intentionally
minimal and well‑commented to make it easy to understand and extend.

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A small convenience block: Conv1d -> BatchNorm -> ReLU -> Dropout    

    Args:
        in_ch (int): number of input channels.
        out_ch (int): number of output channels.
        kernel_size (int): convolution kernel size.
        stride (int): convolution stride.
        padding (int): convolution padding.
        dropout (float): dropout rate (0.0 disables dropout).
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if padding is None:
            # "Same-ish" padding to roughly preserve length when stride = 1
            padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride = stride, padding = padding, bias = False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p = dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class CNN1D(nn.Module):
    """A compact 1D CNN baseline for classification.

    This architecture uses three convolutional stages, each optionally
    downsampling the temporal resolution, followed by global average
    pooling and a linear classifier.

    Input shape:
        (batch_size, in_channels, seq_len)

    Output shape:
        (batch_size, num_classes)

    Args:
        in_channels (int): number of input channels (e.g., sensor axes). Default: 1.
        num_classes (int): number of target classes.
        base_channels (int): channels of the first conv block. The next
            blocks will double this width progressively. Default: 32.
        dropout (float): dropout rate applied inside conv blocks. Default: 0.1.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Stage 1 (no downsampling)
        self.block1 = ConvBlock(in_channels, c1, kernel_size = 7, stride = 1, dropout = dropout)

        # Stage 2 (downsample by 2)
        self.block2 = ConvBlock(c1, c2, kernel_size = 5, stride = 2, dropout = dropout)

        # Stage 3 (downsample by 2)
        self.block3 = ConvBlock(c2, c3, kernel_size = 3, stride = 2, dropout = dropout)

        # Global average pooling collapses time dimension to 1
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Final classifier on pooled features
        self.fc = nn.Linear(c3, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal for convs and zeros/biases where sensible.

        This is a common initialization for ReLU networks and tends to work well
        out of the box.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): shape (B, C, T) where
                B = batch size, C = in_channels, T = sequence length.

        Returns:
            Tensor: logits of shape (B, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)  # (B, C, 1)
        x = x.squeeze(-1)  # (B, C)
        x = self.fc(x)     # (B, num_classes)
        return x


def build_model(
    in_channels: int = 1,
    num_classes: int = 2,
    base_channels: int = 32,
    dropout: float = 0.1,
) -> CNN1D:
    """Factory function used by training scripts.

    Returns a CNN1D with the given hyperparameters. Keeping a thin wrapper
    makes it easier to load the model from a config or CLI without exposing
    internal details.
    """
    return CNN1D(
        in_channels = in_channels,
        num_classes = num_classes,
        base_channels = base_channels,
        dropout = dropout,
    )
