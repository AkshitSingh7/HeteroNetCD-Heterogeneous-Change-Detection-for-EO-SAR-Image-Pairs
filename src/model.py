"""HeteroNetCD: heterogeneous change detection for EO-SAR pairs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class CrossModalAttention(nn.Module):
    """EO -> SAR cross-modal attention. Applied only at deep feature levels
    (HxW <= 4096) to keep memory bounded."""

    def __init__(self, channels):
        super().__init__()
        reduced = max(8, channels // 8)
        self.q = nn.Conv2d(channels, reduced, kernel_size=1)
        self.k = nn.Conv2d(channels, reduced, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, eo, sar):
        B, C, H, W = eo.shape
        if H * W > 4096:
            return sar
        q = self.q(eo).view(B, -1, H * W).permute(0, 2, 1)
        k = self.k(sar).view(B, -1, H * W)
        attn = torch.bmm(q, k) / (q.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        v = self.v(sar).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + sar


class FeatureNorm(nn.Module):
    """GroupNorm applied separately to EO and SAR feature maps."""

    def __init__(self, channels):
        super().__init__()
        ng = min(8, channels) if channels >= 1 else 1
        self.eo = nn.GroupNorm(ng, channels)
        self.sar = nn.GroupNorm(ng, channels)

    def forward(self, a, b):
        return self.eo(a), self.sar(b)


def channel_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel z-score normalization over spatial dims."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + eps
    return (x - mean) / std


class HeteroNetCD(nn.Module):
    """Heterogeneous Change Detection model for EO-SAR pairs.

    Two ResNet50 encoders (SSL EO + SAR-MoCo SAR), per-level FeatureNorm,
    optional cross-modal attention at deep levels, per-channel z-score
    normalization, absolute feature difference with valid-mask gating in the
    forward pass, and a standard UNet decoder.

    For inference, the SAR encoder weights are loaded from the checkpoint,
    so no external pretraining weights are required.

    Forward:
        pre:        (B, 3, H, W) — normalized EO RGB (ImageNet stats)
        post:       (B, 3, H, W) — normalized 3-channel SAR
                                    (raw, Lee-filtered, local-texture)
        valid_mask: (B, 1, H, W) — float in [0, 1]

    Returns:
        logits:     (B, 1, H, W) — pre-sigmoid change scores
    """

    def __init__(
        self,
        sar_weights_path: str = None,
        eo_weights: str = None,
        decoder_channels=(256, 128, 64, 32, 16),
        seg_head_weight_std: float = 1.0,
        seg_head_bias: float = -2.0,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        # EO encoder
        self.enc_eo = smp.encoders.get_encoder(
            "resnet50", in_channels=3, weights=eo_weights,
        )
        # SAR encoder
        self.enc_sar = smp.encoders.get_encoder(
            "resnet50", in_channels=3, weights=None,
        )
        # Optionally load SARhub MoCo weights for SAR (only for training)
        if sar_weights_path is not None:
            sar_state = torch.load(sar_weights_path, map_location="cpu",
                                   weights_only=False)
            self.enc_sar.load_state_dict(sar_state)

        enc_ch = self.enc_eo.out_channels
        self.norms = nn.ModuleList([
            FeatureNorm(ch) if ch > 0 else nn.Identity() for ch in enc_ch
        ])
        self.attn = nn.ModuleList([
            CrossModalAttention(ch) if (ch >= 256 and use_attention) else None
            for ch in enc_ch
        ])

        # Concatenation per level: [EO, SAR, diff] -> 3x at every level except level 0
        fused_ch = tuple(c * 3 if i > 0 else c for i, c in enumerate(enc_ch))

        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=fused_ch,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
        )

        self.seg_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        nn.init.normal_(self.seg_head.weight, std=seg_head_weight_std)
        nn.init.constant_(self.seg_head.bias, seg_head_bias)

    def forward(self, pre, post, valid_mask):
        eo_feats = self.enc_eo(pre)
        sar_feats = self.enc_sar(post)

        fused = []
        for i, (a, b) in enumerate(zip(eo_feats, sar_feats)):
            if i == 0:
                fused.append(a)  
                continue

            # FeatureNorm
            a, b = self.norms[i](a, b)

            # Cross-modal attention at deep levels
            if self.attn[i] is not None:
                b = self.attn[i](a, b)

            # Per-channel z-score before difference
            a_hat = channel_zscore(a)
            b_hat = channel_zscore(b)
            d = torch.abs(a_hat - b_hat)

            # Valid-mask gating
            vm = F.interpolate(valid_mask, size=d.shape[-2:], mode="nearest")
            d = d * vm

            fused.append(torch.cat([a, b, d], dim=1))

        try:
            out = self.decoder(*fused)
        except TypeError:
            out = self.decoder(fused)
        return self.seg_head(out)
