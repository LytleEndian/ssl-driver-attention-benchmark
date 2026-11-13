#!/usr/bin/env python3
"""
salmm_predict.py — run SalM² (Zhao et al., AAAI 2025) on arbitrary dashcam frames or video.

Drop this file in the root of a SaliencyMamba checkout. It wraps the authors' SalMM model
without modifying it and adds:
  1. Inference on any image, folder of images, or video file (not just their JSON splits).
  2. Loading stripped "incomplete" checkpoints directly (--clip-dir), or complete ones with
     no CLIP download; --save-complete writes a full checkpoint.
  3. A pure-PyTorch bidirectional Mamba path (--device cpu / xpu) for hosts without the
     compiled kernels. The repo's built-in slow path silently drops the backward scan.
     --selftest on a CUDA host checks it against the fused kernels.

Preprocessing matches the authors' loaders: RGB, cv2 INTER_CUBIC resize to 256x256, /255,
no mean/std normalisation. Output is the raw sigmoid map (0..1), never per-frame min-max
normalised, so maps stay comparable across frames and datasets.
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
MODEL_HW = (256, 256)          # SalMM's CLIP projector hard-codes 256x256 input
CLIP_PREFIX = "CrossModelAtt.model."


def _import_mamba(reference: bool):
    try:
        import mamba_ssm  # noqa: F401  (compiled install)
        return True
    except Exception as e:
        if not reference:
            sys.exit(f"[salmm] compiled mamba_ssm not importable ({e}).\n"
                     f"         Build it or run with --device cpu/xpu.")
    for name in ("causal_conv1d_cuda", "selective_scan_cuda"):
        sys.modules.setdefault(name, types.ModuleType(name))
    cc = types.ModuleType("causal_conv1d")
    cc.causal_conv1d_fn = cc.causal_conv1d_update = None
    sys.modules["causal_conv1d"] = cc
    pkg = types.ModuleType("mamba_ssm")                     # skip __init__ (pulls transformers)
    pkg.__path__ = [str(REPO / "utils/models/mamba/mamba_ssm")]
    sys.modules["mamba_ssm"] = pkg
    from mamba_ssm.modules.mamba_simple import Mamba
    pkg.Mamba = Mamba
    return False


def _mamba_forward_ref(self, hidden_states, inference_params=None, T=1):
    """Pure-PyTorch equivalent of Mamba.forward's fused bimamba path."""
    from einops import rearrange
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    L = hidden_states.shape[1]
    xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                   "d (b l) -> b d l", l=L)
    if self.in_proj.bias is not None:
        xz = xz + self.in_proj.bias.to(xz.dtype)[:, None]

    def direction(xz, conv, x_proj, dt_proj, A_log, D):
        x, z = xz.chunk(2, dim=1)
        x = F.silu(conv(x)[..., :L])
        x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj.weight)
        dt = rearrange(dt_proj.weight @ x_dbl[:, :self.dt_rank].t(), "d (b l) -> b d l", l=L)
        B = rearrange(x_dbl[:, self.dt_rank:self.dt_rank + self.d_state], "(b l) n -> b n l", l=L)
        C = rearrange(x_dbl[:, -self.d_state:], "(b l) n -> b n l", l=L)
        A = -torch.exp(A_log.float())
        return selective_scan_ref(x, dt, A, B.contiguous(), C.contiguous(), D.float(), z=z,
                                  delta_bias=dt_proj.bias.float(), delta_softplus=True)

    out = direction(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
    if self.bimamba:
        out_b = direction(xz.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b,
                          self.A_b_log, self.D_b)
        out = out + out_b.flip([-1])
    return F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)


def set_reference_mamba(model: torch.nn.Module, on: bool):
    from mamba_ssm.modules.mamba_simple import Mamba
    for m in model.modules():
        if isinstance(m, Mamba):
            if on:
                m.forward = types.MethodType(_mamba_forward_ref, m)
            elif "forward" in m.__dict__:
                del m.forward


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_salmm(ckpt_path: Path, device: torch.device, clip_dir: Path | None):
    reference = device.type != "cuda"
    _import_mamba(reference)
    sys.path.insert(0, str(REPO))
    import utils.models.clip.clip as clipmod
    from utils.models.clip.model import build_model as build_clip, convert_weights

    ckpt = _torch_load(ckpt_path)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    clip_sd = {k[len(CLIP_PREFIX):]: v for k, v in sd.items() if k.startswith(CLIP_PREFIX)}
    orig_load = clipmod.load

    def load_clip(name, device=None, jit=False, download_root=None):
        if clip_sd:
            m = build_clip({k: v.float() for k, v in clip_sd.items()})
        else:
            if clip_dir is None:
                sys.exit("[salmm] checkpoint has no CLIP weights; pass --clip-dir with ViT-B-32.pt")
            m, _ = orig_load(name, device="cpu", jit=False, download_root=str(clip_dir))
        # Match upstream: mixed fp16 CLIP on GPU (as trained), fp32 on CPU.
        if reference:
            m.float()
        else:
            convert_weights(m)
        return m.eval(), None

    clipmod.load = load_clip
    from utils.models.SalMM import SalMM
    model = SalMM()
    clipmod.load = orig_load

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith(CLIP_PREFIX)]
    if missing or unexpected:
        sys.exit(f"[salmm] checkpoint mismatch.\n  missing: {missing[:10]}\n  unexpected: {unexpected[:10]}")
    model = model.to(device).eval()
    if reference:
        set_reference_mamba(model, True)
    return model, ckpt


def iter_frames(inp: Path, stride: int):
    if inp.is_dir():
        files = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXTS)
        for p in files[::stride]:
            yield p.stem, cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif inp.suffix.lower() in IMG_EXTS:
        yield inp.stem, cv2.cvtColor(cv2.imread(str(inp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif inp.suffix.lower() in VID_EXTS:
        cap = cv2.VideoCapture(str(inp))
        if not cap.isOpened():
            sys.exit(f"[salmm] cannot open video {inp}")
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                yield f"{i:06d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
    else:
        sys.exit(f"[salmm] unsupported input {inp}")


def to_tensor(rgb):
    x = cv2.resize(rgb, MODEL_HW, interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def batched(it, n):
    buf = []
    for item in it:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def run(args):
    device = torch.device(args.device)
    model, ckpt = build_salmm(Path(args.ckpt), device, Path(args.clip_dir) if args.clip_dir else None)
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(CLIP_PREFIX))
    print(f"[salmm] loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')}), "
          f"{n_params/1e6:.4f}M trainable params, device={device}, "
          f"mamba={'pure-pytorch' if device.type != 'cuda' else 'cuda kernels'}")

    if args.save_complete:
        full = {k: v for k, v in ckpt.items() if k != "optim_dict"}
        full["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(full, args.save_complete)
        print(f"[salmm] wrote complete checkpoint -> {args.save_complete}")

    if args.selftest:
        return selftest(model, device)
    if not args.input:
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npy_maps, names, t0, n = [], [], time.time(), 0
    writer = None
    with torch.inference_mode():
        for batch in batched(iter_frames(Path(args.input), args.stride), args.batch):
            x = torch.stack([to_tensor(rgb) for _, rgb in batch]).to(device)
            sal = model(x)[0][:, 0].float().cpu().numpy()
            for (name, rgb), s in zip(batch, sal):
                h, w = rgb.shape[:2]
                s_out = s if args.keep_256 else cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
                if "png" in args.save:
                    cv2.imwrite(str(out / f"{name}.png"), np.clip(s_out * 255 + 0.5, 0, 255).astype(np.uint8))
                if "npy" in args.save:
                    npy_maps.append(s.astype(np.float16)); names.append(name)
                if args.overlay:
                    heat = cv2.applyColorMap(np.clip(s_out * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
                    frame = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)
                    if writer is None:
                        writer = cv2.VideoWriter(str(out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 args.overlay_fps, (w, h))
                    writer.write(frame)
            n += len(batch)
            if n % (args.batch * 20) == 0:
                print(f"[salmm] {n} frames, {n/(time.time()-t0):.1f} fps", flush=True)
    if writer is not None:
        writer.release()
    if npy_maps:
        np.savez_compressed(out / "saliency_256.npz", maps=np.stack(npy_maps), names=np.array(names))
    dt = time.time() - t0
    print(f"[salmm] done: {n} frames in {dt:.1f}s ({n/max(dt,1e-9):.1f} fps) -> {out}")


def selftest(model, device):
    if device.type != "cuda":
        sys.exit("[salmm] --selftest compares against the CUDA kernels; run it with --device cuda:N")
    torch.manual_seed(0)
    imgs = sorted((REPO / "fig").glob("case*.jpg"))
    x = (torch.stack([to_tensor(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) for p in imgs])
         if imgs else torch.rand(4, 3, *MODEL_HW)).to(device)
    with torch.inference_mode():
        set_reference_mamba(model, False)
        torch.cuda.synchronize(); t = time.time(); a = model(x)[0]; torch.cuda.synchronize(); ta = time.time() - t
        set_reference_mamba(model, True)
        torch.cuda.synchronize(); t = time.time(); b = model(x)[0]; torch.cuda.synchronize(); tb = time.time() - t
        set_reference_mamba(model, False)
    d = (a - b).abs()
    cc = np.corrcoef(a.flatten().cpu().numpy(), b.flatten().cpu().numpy())[0, 1]
    print(f"[selftest] {x.shape[0]} images | max|diff|={d.max().item():.2e} mean|diff|={d.mean().item():.2e} "
          f"CC={cc:.6f} | kernels {ta*1e3:.0f} ms, reference {tb*1e3:.0f} ms")
    ok = d.max().item() < 1e-3
    print("[selftest] PASS — pure-PyTorch path matches the CUDA kernels" if ok else
          "[selftest] FAIL — reference path diverges; do not mix CPU/XPU and CUDA results")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="SalM² checkpoint (.tar), incomplete or complete")
    ap.add_argument("--input", help="image, directory of images, or video file")
    ap.add_argument("--out", default="salmm_out")
    ap.add_argument("--device", default="cuda:0", help="cuda:N (fused kernels), cpu or xpu (pure PyTorch)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--save", nargs="+", default=["png"], choices=["png", "npy"])
    ap.add_argument("--keep-256", action="store_true", help="write PNGs at 256x256 instead of source size")
    ap.add_argument("--overlay", action="store_true", help="also write overlay.mp4")
    ap.add_argument("--overlay-fps", type=float, default=25.0)
    ap.add_argument("--clip-dir", help="dir holding ViT-B-32.pt (only for incomplete checkpoints)")
    ap.add_argument("--save-complete", help="write a checkpoint with CLIP restored, then continue")
    ap.add_argument("--selftest", action="store_true", help="compare CUDA kernels vs pure-PyTorch path")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
