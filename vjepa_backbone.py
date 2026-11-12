"""Frozen V-JEPA 2.1 encoder -> 4 intermediate feature maps of the clip's LAST tubelet.

Same interface as the DINOv3 wrapper: forward(clip) returns a list of 4 tensors (B, D, H/16, W/16),
so the GazeDecoder is used unchanged.

Construction replicates the official hub loader (src/hub/backbones.py::_make_vjepa2_1_model):
  app.vjepa_2_1.models.vision_transformer.<arch>(img_size=(384,384), num_frames=64, tubelet_size=2,
      use_rope=True, interpolate_rope=True, img_temporal_dim_size=1, use_sdpa=True, ...)
  weights = checkpoint["ema_encoder"] with "module." and "backbone." stripped.
The hub uses strict=False; we load the same way but RECORD missing/unexpected keys (see verify_vjepa.py).

Notes for Methods:
  * Readout blocks default to the model's own hierarchical layers ([5, 11, 17, 23] for ViT-L),
    identical to the DINOv3-L blocks.
  * feat_norm="none" (default): raw residual stream, as DINOv3 (get_intermediate_layers norm=False);
    the decoder's per-layer BatchNorm normalises. feat_norm="model": V-JEPA's trained norms_block LN.
  * Non-square input: RoPE positions use the true H x W patch grid (handle_nonsquare_inputs=True) and
    interpolate_rope rescales each axis to the pretraining grid range (normalised coordinates).
  * Input clip: (B, 3, T, H, W), ImageNet-normalised, T even and >= 2 (T == 1 would take the image path).
    Tokens are frame-major, so the last H*W tokens are the last tubelet (frames T-2, T-1).
"""
import contextlib
import sys

import torch
import torch.nn as nn

VJEPA_SRC = "/opt/vjepa2/src"

HUB_ENCODER_KWARGS = dict(
    patch_size=16,
    img_size=(384, 384),
    num_frames=64,
    tubelet_size=2,
    use_sdpa=True,
    use_SiLU=False,       # (sic) capitalised as in the hub; swallowed by **kwargs -> GELU MLP, as in the hub
    wide_SiLU=True,
    uniform_power=False,
    use_rope=True,
    img_temporal_dim_size=1,
    interpolate_rope=True,
)


def clean_backbone_keys(sd):
    """Identical to the hub's _clean_backbone_key."""
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


class FrozenVJEPA21(nn.Module):
    def __init__(self, ckpt_path, arch="vit_large", layers=None, feat_norm="none",
                 checkpoint_key="ema_encoder", vjepa_src=VJEPA_SRC, fp16=True):
        super().__init__()
        self.fp16 = fp16
        if feat_norm not in ("none", "model"):
            raise ValueError(f"feat_norm must be 'none' or 'model', got {feat_norm!r}")
        if vjepa_src not in sys.path:
            sys.path.insert(0, vjepa_src)
        from app.vjepa_2_1.models import vision_transformer as vit

        self.enc = vit.__dict__[arch](**HUB_ENCODER_KWARGS)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = clean_backbone_keys(ck[checkpoint_key])
        res = self.enc.load_state_dict(sd, strict=False)
        self.missing_keys = list(res.missing_keys)
        self.unexpected_keys = list(res.unexpected_keys)
        self.n_ckpt_tensors = len(sd)
        del ck, sd

        self.layers = list(layers) if layers is not None else list(self.enc.hierarchical_layers)
        for i in self.layers:
            if feat_norm == "model" and i not in self.enc.hierarchical_layers:
                raise ValueError(f"feat_norm='model' needs layers in {self.enc.hierarchical_layers}")
        self.feat_norm = feat_norm
        self.patch_size = self.enc.patch_size
        self.tubelet_size = self.enc.tubelet_size
        self.embed_dim = self.enc.embed_dim
        self.arch = arch
        self.enc.requires_grad_(False)
        self.enc.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()          # backbone pinned to eval regardless of the wrapper's mode
        return self

    @torch.no_grad()
    def forward(self, clip):
        B, C, T, H, W = clip.shape
        if T < 2 or T % self.tubelet_size:
            raise ValueError(f"clip length T={T} must be a multiple of {self.tubelet_size} and >= 2")
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"H, W = {H}, {W} must be multiples of {self.patch_size}")
        hp, wp = H // self.patch_size, W // self.patch_size
        n_last = hp * wp

        captured = {}
        hooks = [self.enc.blocks[i].register_forward_hook(
                     lambda mod, inp, out, i=i: captured.__setitem__(i, out[0]))
                 for i in self.layers]
        # same convention as FrozenDINOv3: backbone under fp16 autocast, float32 out
        ctx = (torch.autocast("cuda", dtype=torch.float16) if self.fp16 and clip.is_cuda
               else contextlib.nullcontext())
        try:
            with ctx:
                self.enc(clip)
        finally:
            for h in hooks:
                h.remove()

        outs = []
        for i in self.layers:
            x = captured[i]
            if x.shape[1] != (T // self.tubelet_size) * n_last:
                raise RuntimeError(f"unexpected token count {x.shape[1]} at block {i}")
            if self.feat_norm == "model":
                x = self.enc.norms_block[self.enc.hierarchical_layers.index(i)](x)
            x = x[:, -n_last:, :]                                   # last tubelet
            outs.append(x.reshape(B, hp, wp, -1).permute(0, 3, 1, 2).float().contiguous())
        return outs


def clip_frame_indices(gt_frame, n_frames, step, available):
    """Causal clip ending at gt_frame: [gt-(n-1)*step, ..., gt-step, gt].

    Indices before the video's first available frame are clamped to it (pad by repetition).
    `available` is the sorted list of frame indices present for this video.
    """
    first = available[0]
    avail = set(available)
    idx = []
    for k in range(n_frames - 1, -1, -1):
        f = gt_frame - k * step
        if f < first:
            f = first
        if f not in avail:
            raise KeyError(f"frame {f} missing (gt {gt_frame}, step {step})")
        idx.append(f)
    return idx
