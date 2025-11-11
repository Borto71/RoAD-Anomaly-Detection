from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A basic 1D convolutional block:
    Conv1d -> BatchNorm -> GELU -> Dropout

    Args:
        in_ch:    input channels
        out_ch:   output channels
        k:        kernel size
        s:        stride
        p:        padding
        dropout:  dropout probability
    """
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if p is None:
            p = k // 2  # "same" padding when stride = 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size = k, stride = s, padding = p, bias = False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.do = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two ConvBlocks and a 1x1 shortcut if dimensions change.
    This stabilizes training on long sequences and improves gradient flow.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, dropout: float = 0.1):
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch, k = k, s = s, dropout = dropout)
        self.block2 = ConvBlock(out_ch, out_ch, k = k, s = 1, dropout = dropout)
        self.proj = None
        if in_ch != out_ch or s != 1:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size = 1, stride = s, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        return F.gelu(out + identity)


class RoADCNN1D(nn.Module):
    """
    CNN-1D model for anomaly detection.

    Inputs:
        x: Tensor of shape (B, C, T)
           B = batch size, C = feature channels (e.g., 86), T = timesteps.

    Outputs:
        A tuple (logit, frame_logits) where:
        - logit: shape (B, 1) sequence-level anomaly score (raw logit).
        - frame_logits: shape (B, 1, T') optional per-timestep logits after the
          last convolutional downsampling stage (useful for weak supervision).
          Returned only if `return_frame_logits=True`.

    Design:
        - Stem + 3 residual stages with progressive channel expansion and
          occasional temporal downsampling (stride=2).
        - Dual global pooling (avg + max) then a small MLP head.
        - Optional 1x1 conv head for frame-wise scores.
    """
    def __init__(
        self,
        in_channels: int = 86,      # use 86 features; exclude channel 87 (label)
        base_channels: int = 64,    # width of the first stage
        dropout: float = 0.1,
        return_frame_logits: bool = False,
    ):
        super().__init__()
        self.return_frame_logits = return_frame_logits

        # ----- Feature extractor (hierarchical CNN) -----
        self.stem = ConvBlock(in_channels, base_channels, k = 7, s = 1, dropout=dropout)

        self.stage1 = ResidualBlock(base_channels, base_channels, k = 7, s = 1, dropout = dropout)
        self.stage2 = ResidualBlock(base_channels, base_channels * 2, k = 5, s = 2, dropout = dropout)   # downsample
        self.stage3 = ResidualBlock(base_channels * 2, base_channels * 4, k = 3, s = 2, dropout = dropout) # downsample
        self.stage4 = ResidualBlock(base_channels * 4, base_channels * 4, k = 3, s = 1, dropout = dropout)

        feat_dim = base_channels * 4

        # ----- Sequence-level head (global pooling + MLP) -----
        # We concatenate avg and max pooled features to capture both trend and bursts.
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1)  # single logit for binary anomaly classification
        )

        # ----- Optional frame-level head -----
        if self.return_frame_logits:
            self.frame_head = nn.Conv1d(feat_dim, 1, kernel_size = 1)

        self._init_weights()

    def _init_weights(self):
        # Kaiming initialization for conv layers, xavier for linear layers.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity = "relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, C, T)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)     # features: (B, F, T')

        # Global pooling (keep both statistics)
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(-1)   # (B, F)
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(-1)   # (B, F)
        pooled = torch.cat([avg_pool, max_pool], dim=1)      # (B, 2F)

        logit = self.head(pooled)                             # (B, 1)

        if self.return_frame_logits:
            frame_logits = self.frame_head(x)                 # (B, 1, T')
            return logit, frame_logits
        else:
            return logit, None


def build_model(
    in_channels: int = 86,
    base_channels: int = 64,
    dropout: float = 0.1,
    return_frame_logits: bool = False,
) -> RoADCNN1D:
    """
    Factory to create the CNN model. Keep the signature stable so it can be
    imported from a training script (main).

    Example:
        from model_cnn1d import build_model
        model = build_model(in_channels=86, base_channels=64, dropout=0.1)
    """
    return RoADCNN1D(
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
        return_frame_logits=return_frame_logits,
    )
