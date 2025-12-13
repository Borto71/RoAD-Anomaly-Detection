from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic 1D convolutional block:
    Conv1d -> BatchNorm1d -> GELU (or other non-linearity) -> Dropout.

    This block keeps the temporal resolution (same T) when:
        padding = dilation * (kernel_size - 1) // 2
        stride = 1
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Choose padding so that, with stride = 1, the temporal length is preserved.
        padding = (k - 1) // 2 * dilation

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, C_in, T)

        Returns:
            Tensor of shape (B, C_out, T_out)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two ConvBlocks and an optional 1x1 shortcut.

    The main idea is:
        y = F(main(x) + shortcut(x))

    If in_ch != out_ch or stride > 1, the shortcut uses a 1x1 conv
    to match shapes.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = ConvBlock(
            in_ch,
            out_ch,
            k=k,
            s=s,
            dilation=dilation,
            dropout=dropout,
        )
        # Second conv keeps stride=1 so that only the first convolution
        # controls the potential downsampling.
        self.conv2 = ConvBlock(
            out_ch,
            out_ch,
            k=k,
            s=1,
            dilation=dilation,
            dropout=dropout,
        )

        # Optional shortcut to match (C, T) shape.
        if s != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        # Extra non-linearity after the residual addition.
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, C_in, T)

        Returns:
            Tensor of shape (B, C_out, T_out)
        """
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        out = self.out_act(out)
        return out


class RoADCNN1D(nn.Module):
    """
    CNN-1D model for anomaly detection on multi-variate time series.

    Base design:
    - The network preserves the temporal resolution (same T) throughout.
    - A 1x1 convolution produces per-timestep logits (frame_logits).
    - The sequence-level logit is derived from frame_logits using a pooling
      operation across time (e.g. top-k or max pooling).
    """
    def __init__(
        self,
        in_channels: int = 86,      # number of input features; exclude label channel
        base_channels: int = 64,    # width of the first stage
        dropout: float = 0.1,
        return_frame_logits: bool = False,
        seq_pooling: str = "topk",  # "topk", "max", or "avg"
        topk_ratio: float = 0.1,    # fraction of timesteps used in top-k pooling
    ) -> None:
        super().__init__()
        self.return_frame_logits = return_frame_logits
        self.seq_pooling = seq_pooling
        self.topk_ratio = topk_ratio

        # Feature extractor: 4 stages of residual blocks with increasing capacity
        self.stem = ConvBlock(
            in_channels,
            base_channels,
            k=7,
            s=1,
            dilation=1,
            dropout=dropout,
        )

        # Stage 1: no dilation, still local.
        self.stage1 = ResidualBlock(
            base_channels,
            base_channels,
            k=7,
            s=1,
            dilation=1,
            dropout=dropout,
        )

        # Stage 2: increase channels, moderate dilation.
        self.stage2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            k=5,
            s=1,
            dilation=2,
            dropout=dropout,
        )

        # Stage 3: more channels, larger dilation for longer temporal context.
        self.stage3 = ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            k=5,
            s=1,
            dilation=4,
            dropout=dropout,
        )

        # Stage 4: keep width, even larger dilation to capture slow trends.
        self.stage4 = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            k=3,
            s=1,
            dilation=8,
            dropout=dropout,
        )

        feat_dim = base_channels * 4

        # Frame-level head:
        # 1x1 convolution that maps the feature dimension to a single logit per timestep.
        self.frame_head = nn.Conv1d(
            feat_dim,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    # Feature extraction utilities
    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passa l'input attraverso stem + stage1..4 e ritorna il feature map (B, F, T).
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)   # (B, F, T)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estrae un embedding di finestra di shape (B, F) per metodi unsupervised
        tipo Deep SVDD.

        - x: (B, C_in, T)
        - ritorno: (B, F) dove F = base_channels * 4
        """
        feat = self._extract_feature_map(x)   # (B, F, T)
        z = feat.mean(dim=-1)                # global average pooling su T -> (B, F)
        return z

    # Forward 
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass (supervised-style).
        """
        # Feature extractor
        x = self._extract_feature_map(x)      # (B, F, T)

        # Frame-level logits: per-timestep anomaly score.
        frame_logits = self.frame_head(x)     # (B, 1, T)

        # Pool over the temporal dimension to obtain a single score per window.
        if self.seq_pooling == "avg":
            # Simple average over time.
            seq_logit = frame_logits.mean(dim=-1)  # (B, 1)
        elif self.seq_pooling == "max":
            # Max pooling focuses on the most anomalous timestep.
            seq_logit, _ = frame_logits.max(dim=-1)  # (B, 1)
        else:
            # Top-k pooling (default):
            # take the mean over the k most anomalous timesteps.
            B, C, T = frame_logits.shape
            k = max(1, int(self.topk_ratio * T))
            topk_vals, _ = torch.topk(frame_logits, k=k, dim=-1)
            seq_logit = topk_vals.mean(dim=-1)  # (B, 1)

        if self.return_frame_logits:
            return seq_logit, frame_logits
        else:
            # If you do not need per-frame scores (e.g., at inference), you can
            # set return_frame_logits=False when building the model to save memory.
            return seq_logit, None


def build_model(
    in_channels: int = 86,
    base_channels: int = 64,
    dropout: float = 0.1,
    return_frame_logits: bool = False,
) -> RoADCNN1D:
    """
    Factory to create the CNN model. The signature is kept stable so it can be
    imported from a training script.
    """
    return RoADCNN1D(
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
        return_frame_logits=return_frame_logits,
    )
