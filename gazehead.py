"""
gazehead.py - frozen-backbone driver-gaze decoder.

One decoder design shared across backbones (DINOv3 now, V-JEPA 2.1 later).
Only the per-backbone input adapter changes; report its parameter count
alongside the decoder's.

Design choices (each traceable to the DINOv3 paper, arXiv 2508.10104):
  * Four equally spaced intermediate blocks, concatenated  (App. D.12 ViT-L/VGGT,
    D.10/D.11 dense heads).
  * Intermediate features taken WITHOUT the final LayerNorm and normalized with a
    learned BatchNorm instead. App. A.2: the final LN is trained for the last layer
    and "may be suboptimal" for earlier ones; BN suppresses the feature-dimension
    outliers.
  * Aspect-preserving resize to multiples of 16 (RoPE handles any aspect ratio);
    the README's square Resize((s, s)) would squash 16:9 dashcam frames.
  * Backbone under fp16 autocast (V100 / sm_70 has no bf16); decoder in fp32.

Benchmark rule: input is RGB frames only. No telemetry, GPS, maps, task labels,
and no hand-crafted center prior channel (convolution zero-padding still lets
the decoder learn position, and it is scored against a center-bias baseline).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

IMNET_MEAN = (0.485, 0.456, 0.406)  # LVD-1689M weights (README "Image transforms")
IMNET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def size_for_aspect(width: int, height: int, short_side: int = 288, mult: int = 16) -> tuple[int, int]:
    """(H, W) that keeps the frame aspect ratio, both sides multiples of `mult`.

    BDD-A / DR(eye)VE (16:9) at short_side=288 -> (288, 512) -> 18x32 tokens.
    DADA-2000 (1584x660) at 256 -> (256, 608).
    """
    if width >= height:
        h = short_side
        w = int(round(width * short_side / height / mult)) * mult
    else:
        w = short_side
        h = int(round(height * short_side / width / mult)) * mult
    assert h % mult == 0 and w % mult == 0
    return h, w


def make_transform(size_hw: tuple[int, int]):
    """Frame transform. antialias is set explicitly, so the torchvision 0.17
    default change does not apply. Feed PIL images or uint8 tensors."""
    from torchvision.transforms import v2

    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size_hw, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMNET_MEAN, IMNET_STD),
    ])


# --------------------------------------------------------------------------- #
# Frozen encoder wrapper
# --------------------------------------------------------------------------- #
def equally_spaced_blocks(depth: int, n: int = 4) -> tuple[int, ...]:
    """0-indexed block ids. depth 12 -> (2,5,8,11); depth 24 -> (5,11,17,23)."""
    step = depth // n
    return tuple(step * (k + 1) - 1 for k in range(n))


class FrozenDINOv3(nn.Module):
    """Wraps a torch.hub DINOv3 ViT; returns a list of B x C x h x w maps.

    Only patch tokens come back from get_intermediate_layers(reshape=True):
    the CLS and the 4 register/storage tokens are dropped by the backbone,
    so nothing non-spatial gets reshaped into the grid.
    """

    def __init__(self, backbone: nn.Module, blocks: Sequence[int] | None = None,
                 fp16: bool = True):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.depth = len(backbone.blocks)
        self.blocks = tuple(blocks) if blocks is not None else equally_spaced_blocks(self.depth)
        self.embed_dim = int(backbone.embed_dim)
        self.patch_size = int(getattr(backbone, "patch_size", 16))
        self.fp16 = fp16

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # never let backbone norm/dropout switch to train
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        H, W = x.shape[-2:]
        p = self.patch_size
        if H % p or W % p:
            raise ValueError(f"input {H}x{W} not a multiple of patch size {p}")
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16 and x.is_cuda):
            feats = self.backbone.get_intermediate_layers(
                x, n=list(self.blocks), reshape=True, norm=False)
        return [f.float() for f in feats]


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class GazeDecoder(nn.Module):
    """Per-layer BN -> concat -> 1x1 adapter (C*L -> width) -> shared CNN body.

    `adapter` (+ `bn`) is backbone-specific; `body` is identical across backbones.
    Output: log-density over out_size, i.e. logsumexp over pixels == 0.
    """

    def __init__(self, in_dim: int, n_layers: int = 4, width: int = 128):
        super().__init__()
        self.bn = nn.ModuleList(nn.BatchNorm2d(in_dim) for _ in range(n_layers))
        self.adapter = nn.Conv2d(in_dim * n_layers, width, kernel_size=1)
        self.body = nn.Sequential(
            nn.Conv2d(width, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, feats: list[torch.Tensor], out_size: tuple[int, int]) -> torch.Tensor:
        if len(feats) != len(self.bn):
            raise ValueError(f"expected {len(self.bn)} feature maps, got {len(feats)}")
        x = torch.cat([bn(f) for bn, f in zip(self.bn, feats)], dim=1)
        z = self.body(self.adapter(x))                       # B x 1 x h x w logits
        z = F.interpolate(z, size=out_size, mode="bilinear", align_corners=False)
        B = z.shape[0]
        return z.reshape(B, -1).log_softmax(dim=-1).reshape(B, *out_size)

    def param_counts(self) -> dict[str, int]:
        n = lambda m: sum(p.numel() for p in m.parameters())
        return {"bn": n(self.bn), "adapter": n(self.adapter), "body": n(self.body), "total": n(self)}


class GazeModel(nn.Module):
    def __init__(self, encoder: FrozenDINOv3, decoder: GazeDecoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        return self.decoder(self.encoder(x), out_size)

    def trainable_parameters(self):
        return self.decoder.parameters()


# --------------------------------------------------------------------------- #
# Loss (training only; evaluation uses the shared metrics script)
# --------------------------------------------------------------------------- #
def kld_loss(log_pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(H || J). gt: B x H x W non-negative continuous attention maps at out_size.
    Frames with an empty GT map are dropped, not silently scored."""
    g = gt.reshape(gt.shape[0], -1).float()
    s = g.sum(-1)
    keep = s > eps
    if not keep.any():
        return log_pred.sum() * 0.0
    g = g[keep] / s[keep, None]
    lp = log_pred.reshape(log_pred.shape[0], -1)[keep]
    return F.kl_div(lp, g, reduction="batchmean")  # uses xlogy: zeros in g are fine


# --------------------------------------------------------------------------- #
# Gaze-free map (unsupervised experiment, Method A in the notes)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def novelty_map(feat: torch.Tensor, k: int = 3) -> torch.Tensor:
    """1 - cos(F(x,y), mean of its k x k neighbours, centre excluded).
    feat: B x C x h x w (e.g. BN-free last block, or final-LN output).
    Returns B x h x w normalized to sum 1. No gaze data touches this."""
    f = F.normalize(feat.float(), dim=1)
    pad = k // 2
    ones = torch.ones_like(f[:, :1])
    s = F.avg_pool2d(f, k, 1, pad, count_include_pad=True) * k * k
    c = F.avg_pool2d(ones, k, 1, pad, count_include_pad=True) * k * k
    nb = F.normalize((s - f) / (c - 1).clamp_min(1), dim=1)
    m = (1 - (f * nb).sum(1)).clamp_min(0)
    return m / m.flatten(1).sum(-1).clamp_min(1e-12)[:, None, None]


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build_dinov3_gaze(repo_dir: str, hub_name: str, weights: str,
                      blocks: Sequence[int] | None = None, width: int = 128,
                      fp16: bool = True) -> GazeModel:
    backbone = torch.hub.load(repo_dir, hub_name, source="local", weights=weights)
    enc = FrozenDINOv3(backbone, blocks=blocks, fp16=fp16)
    dec = GazeDecoder(enc.embed_dim, n_layers=len(enc.blocks), width=width)
    return GazeModel(enc, dec)
