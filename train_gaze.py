#!/usr/bin/env python
"""
train_gaze.py - train the frozen-backbone gaze decoder on manifest splits, pick the
best epoch on validation, then write test predictions.

Smoke run (minutes):
  python train_gaze.py --hub dinov3_vitb16 --weights W --out /srv/results/dinov3/smoke \
      --max-steps 300 --val-stride 50 --no-predict
Full run (BDD-A, defaults):
  python train_gaze.py --hub dinov3_vitb16 --weights W --out /srv/results/dinov3/bdda_vitb16
Predict only:
  python train_gaze.py --hub ... --weights W --out RUN --predict-only --ckpt RUN/best.pt

Another dataset (CoCAtt): point --root / --cache at it and name the manifests explicitly:
  python train_gaze.py --hub dinov3_vitl16 --weights W --root /srv/datasets/CoCAtt \
      --input 256x512 --target 64x128 --pred-size 64x128 --cache /srv/cache/cocatt_256x512 \
      --train-manifest cocatt_train --val-manifest cocatt_val --test-manifests cocatt_test \
      --stride 2 --val-stride 2 --epochs 30 --out /srv/results/dinov3/cocatt_vitl16_light_s2

V-JEPA 2.1 (frozen video backbone, same decoder and recipe; train/val from cached features,
test predictions end-to-end on clips from the frame cache):
  python train_gaze.py --backbone vjepa21 --weights /srv/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
      --clip-frames 8 --clip-step 7 --cache /srv/cache/bdda_288x512 \
      --train-features /srv/cache/vjepa21L_bdda_c8s7/train_s10 --val-features /srv/cache/vjepa21L_bdda_c8s7/val_s5 \
      --stride 10 --epochs 30 --out /srv/results/vjepa21/bdda_vitl16_light_s10_c8s7

Validation KLD / CC here are for model selection only. Reported numbers come from the
shared metrics script run on RUN/pred_<test-manifest>/*.npz.

Outputs in --out:
  run.json            args, versions, hashes of weights / manifests / code
  log.csv             per-epoch train loss, val KLD, val CC, lr, frames/s
  last.pt, best.pt    decoder state_dict only (backbone is the frozen checkpoint)
  pred_<name>/<vid>.npz   frames (int), pred (float16 N x h x w, each sums to 1)

History:
  archive/train_gaze_dinov3.py      version used for the BDD-A DINOv3 runs
  archive/train_gaze_vjepa_bdda.py  version used for the BDD-A V-JEPA runs
  this version adds --train-manifest/--val-manifest, hashes the manifests it actually
  uses (no dependence on manifest_summary.json), and accepts string video ids.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CachedFrames, ManifestFrames
from gazehead import IMNET_MEAN, IMNET_STD, GazeDecoder, GazeModel, build_dinov3_gaze, kld_loss

_MEAN = _STD = None


def prep(x, dev):
    """Move a batch to the GPU. uint8 (from the cache) is scaled and normalized here,
    matching v2.ToDtype(float32, scale=True) + v2.Normalize(IMNET_MEAN, IMNET_STD)."""
    global _MEAN, _STD
    x = x.to(dev, non_blocking=True)
    if x.dtype == torch.uint8:
        if _MEAN is None:
            _MEAN = torch.tensor(IMNET_MEAN, device=dev).view(1, 3, 1, 1)
            _STD = torch.tensor(IMNET_STD, device=dev).view(1, 3, 1, 1)
        x = (x.float() / 255.0 - _MEAN) / _STD
    return x


def forward_batch(model, b, size, dev):
    """frames (B,3,H,W) -> DINOv3 path, unchanged; clips (B,T,3,H,W) -> video backbone;
    cached features (B,L,D,h,w) -> decoder only (backbone already applied offline)."""
    if "f" in b:
        f = b["f"].to(dev, non_blocking=True).float()
        return model.decoder(list(f.unbind(1)), size)
    x = b["x"]
    if x.ndim == 5:
        from clipdata import prep_clip
        return model(prep_clip(x, dev), size)
    return model(prep(x, dev), size)


def sha256(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hw(s):
    h, w = s.lower().split("x")
    return int(h), int(w)


def kld_per_frame(lp, gt, eps=1e-12):
    g = gt.flatten(1)
    s = g.sum(1)
    keep = s > eps
    g = g[keep] / s[keep, None]
    return F.kl_div(lp.flatten(1)[keep], g, reduction="none").sum(1)


def cc_per_frame(lp, gt):
    p = lp.exp().flatten(1)
    g = gt.flatten(1)
    p = p - p.mean(1, keepdim=True)
    g = g - g.mean(1, keepdim=True)
    return (p * g).sum(1) / (p.norm(dim=1) * g.norm(dim=1)).clamp_min(1e-12)


@torch.no_grad()
def evaluate(model, loader, size, dev, max_batches=None):
    model.eval()
    k, c = [], []
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        lp = forward_batch(model, b, size, dev)
        gt = b["gt"].to(dev, non_blocking=True)
        k.append(kld_per_frame(lp, gt).cpu())
        c.append(cc_per_frame(lp, gt).cpu())
    model.train()
    return torch.cat(k).mean().item(), torch.cat(c).nanmean().item()


@torch.no_grad()
def predict(model, loader, size, dev, out_dir: pathlib.Path):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = collections.defaultdict(lambda: ([], []))
    n = 0
    for b in loader:
        p = forward_batch(model, b, size, dev).exp().half().cpu().numpy()
        # int vids collate to a tensor (BDD-A); string vids (CoCAtt 'A_0') collate to a list
        vids = b["vid"] if isinstance(b["vid"], list) else b["vid"].tolist()
        for v, f, m in zip(vids, b["frame"].tolist(), p):
            buf[v][0].append(f)
            buf[v][1].append(m)
        n += len(p)
    for v, (fs, ms) in buf.items():
        o = np.argsort(fs)
        np.savez_compressed(out_dir / f"{v}.npz", frames=np.asarray(fs)[o], pred=np.stack(ms)[o])
    print(f"wrote {n} predictions for {len(buf)} videos -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="dinov3_vitb16")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--repo", default="/opt/dinov3")
    ap.add_argument("--root", default="/srv/datasets/BDDA")
    ap.add_argument("--splits", default="meta/splits")
    ap.add_argument("--train-manifest", default="train", help="manifest name under --splits")
    ap.add_argument("--val-manifest", default="val", help="manifest name under --splits")
    ap.add_argument("--test-manifests", default="test_common", help="comma list of manifest names to predict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", type=hw, default="288x512")
    ap.add_argument("--target", type=hw, default="72x128", help="GT size for loss / val")
    ap.add_argument("--pred-size", type=hw, default="72x128", help="saved prediction size")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1, help="train frame stride (10 ~ 2.9 Hz on BDD-A)")
    ap.add_argument("--val-stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke test)")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-predict", action="store_true")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--cache", help="dir from tools/build_frame_cache.py (must match --input/--target)")
    ap.add_argument("--backbone", default="dinov3", choices=["dinov3", "vjepa21"])
    ap.add_argument("--vjepa-arch", default="vit_large")
    ap.add_argument("--feat-norm", default="none", choices=["none", "model"], help="vjepa21 readout norm")
    ap.add_argument("--clip-frames", type=int, default=0, help="vjepa21: frames per clip")
    ap.add_argument("--clip-step", type=int, default=0, help="vjepa21: frame step within a clip")
    ap.add_argument("--train-features", help="vjepa21: dir from tools/extract_vjepa_features.py")
    ap.add_argument("--val-features", help="vjepa21: dir from tools/extract_vjepa_features.py")
    a = ap.parse_args()
    video = a.backbone == "vjepa21"
    if video and not (a.clip_frames and a.clip_step and a.cache):
        raise SystemExit("--backbone vjepa21 needs --clip-frames, --clip-step and --cache")

    splits = pathlib.Path(a.splits)
    test_names = a.test_manifests.split(",")
    for n in ([] if a.predict_only else [a.train_manifest, a.val_manifest]) + ([] if a.no_predict else test_names):
        if not (splits / f"{n}.csv").is_file():
            raise SystemExit(f"manifest not found: {splits / f'{n}.csv'}")

    def check_features(d, manifest, stride):
        m = json.load(open(pathlib.Path(d) / "meta.json"))
        want = {"manifest": manifest, "stride": stride, "clip_frames": a.clip_frames, "clip_step": a.clip_step,
                "manifest_sha256": sha256(splits / f"{manifest}.csv"),
                "ckpt_sha256": sha256(a.weights)}
        got = {k: m.get(k) for k in want}
        got["layers_norm"] = (m["readout"]["arch"], m["readout"]["feat_norm"])
        want["layers_norm"] = (a.vjepa_arch, a.feat_norm)
        bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
        if bad or not m.get("complete"):
            raise SystemExit(f"{d}: features do not match this run: {bad} complete={m.get('complete')}")
        return m

    if a.cache:
        cm = json.load(open(pathlib.Path(a.cache) / "meta.json"))
        if tuple(cm["frames_shape"][2:]) != tuple(a.input) or tuple(cm["gt_shape"][1:]) != tuple(a.target):
            raise SystemExit(f"cache shapes {cm['frames_shape']} / {cm['gt_shape']} do not match "
                             f"--input {a.input} / --target {a.target}")

    def make_ds(name, stride=1, with_gt=True, size=None, features=None):
        path = splits / f"{name}.csv"
        if video:
            from clipdata import CachedClips, CachedFeatures
            if size is not None and tuple(size) != tuple(a.target):
                raise SystemExit("vjepa21 predictions use the cache GT size: --pred-size must equal --target")
            if features:
                check_features(features, name, stride)
                return CachedFeatures(features, with_gt=with_gt)
            return CachedClips(path, a.cache, a.clip_frames, a.clip_step, stride=stride, with_gt=with_gt)
        if a.cache and (size is None or tuple(size) == tuple(a.target)):
            return CachedFrames(path, a.cache, stride=stride, with_gt=with_gt)
        return ManifestFrames(path, a.root, a.input, size or a.target, stride=stride, with_gt=with_gt)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(f"cuda:{a.gpu}")

    if video:
        from vjepa_backbone import FrozenVJEPA21
        enc = FrozenVJEPA21(a.weights, arch=a.vjepa_arch, feat_norm=a.feat_norm, fp16=True)
        if enc.missing_keys or enc.unexpected_keys:
            print("WARNING backbone key mismatch:", enc.missing_keys, enc.unexpected_keys)
        model = GazeModel(enc, GazeDecoder(enc.embed_dim, n_layers=len(enc.layers), width=a.width)).to(dev)
        print("decoder params:", model.decoder.param_counts(), "| blocks:", enc.layers,
              f"| clip {a.clip_frames} frames step {a.clip_step}")
    else:
        model = build_dinov3_gaze(a.repo, a.hub, a.weights, width=a.width).to(dev)
        print("decoder params:", model.decoder.param_counts(), "| blocks:", model.encoder.blocks)

    lk = dict(batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)

    if not a.predict_only:
        meta = {
            "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items()},
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(dev),
            "weights_sha256": sha256(a.weights),
            "dinov3_commit": subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip(),
            "code_sha256": {p: sha256(p) for p in ("gazehead.py", "data.py", "train_gaze.py")},
            **({"vjepa2_commit": subprocess.run(["git", "-C", "/opt/vjepa2/src", "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout.strip(),
                "vjepa_code_sha256": {p: sha256(p) for p in ("vjepa_backbone.py", "clipdata.py")},
                "backbone_keys": {"missing": enc.missing_keys, "unexpected": enc.unexpected_keys},
                "features": {"train": a.train_features, "val": a.val_features}} if video else {}),
            # hashes of the manifests this run actually uses (independent of any summary file)
            "manifests": {n: sha256(splits / f"{n}.csv")
                          for n in [a.train_manifest, a.val_manifest, *test_names]},
            "cache": {"dir": a.cache, **({k: cm[k] for k in cm if k != "frames"} if a.cache else {})},
            "decoder_params": model.decoder.param_counts(),
        }
        json.dump(meta, open(out / "run.json", "w"), indent=2, default=str)

        tr = make_ds(a.train_manifest, stride=a.stride, features=a.train_features)
        va = make_ds(a.val_manifest, stride=a.val_stride, features=a.val_features)
        print(f"train {a.train_manifest}: {len(tr)} frames | val {a.val_manifest}: {len(va)} frames")
        tl = DataLoader(tr, shuffle=True, drop_last=True, **lk)
        vl = DataLoader(va, shuffle=False, **lk)

        opt = torch.optim.AdamW(model.trainable_parameters(), lr=a.lr, weight_decay=a.wd)
        total = a.max_steps or a.epochs * len(tl)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

        logf = open(out / "log.csv", "w", newline="")
        logw = csv.writer(logf)
        logw.writerow(["epoch", "step", "train_kld", "val_kld", "val_cc", "lr", "frames_per_s", "data_wait_frac"])
        best, step, done = float("inf"), 0, False
        model.train()
        for ep in range(1, a.epochs + 1):
            run_loss, n_b, t0, wait = 0.0, 0, time.time(), 0.0
            tw = time.time()
            for b in tl:
                wait += time.time() - tw
                lp = forward_batch(model, b, a.target, dev)
                loss = kld_loss(lp, b["gt"].to(dev, non_blocking=True))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run_loss += loss.item()
                n_b += 1
                step += 1
                if step % a.log_every == 0:
                    el = time.time() - t0
                    print(f"ep {ep} step {step}/{total} kld {run_loss/n_b:.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{n_b*a.bs/el:.0f} fr/s data-wait {wait/el:.0%}", flush=True)
                if a.max_steps and step >= a.max_steps:
                    done = True
                    break
                tw = time.time()
            el = time.time() - t0
            vk, vc = evaluate(model, vl, a.target, dev, max_batches=50 if a.max_steps else None)
            logw.writerow([ep, step, f"{run_loss/max(n_b,1):.5f}", f"{vk:.5f}", f"{vc:.5f}",
                           f"{sched.get_last_lr()[0]:.3e}", f"{n_b*a.bs/el:.1f}", f"{wait/el:.3f}"])
            logf.flush()
            ck = {"decoder": model.decoder.state_dict(), "epoch": ep, "step": step,
                  "val_kld": vk, "val_cc": vc, "args": meta["args"]}
            torch.save(ck, out / "last.pt")
            tag = ""
            if vk < best:
                best = vk
                torch.save(ck, out / "best.pt")
                tag = "  * best"
            print(f"== epoch {ep}: train kld {run_loss/max(n_b,1):.4f} | val kld {vk:.4f} cc {vc:.4f}{tag}", flush=True)
            if done:
                break
        logf.close()

    if a.no_predict:
        return
    ck = torch.load(a.ckpt or out / "best.pt", map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(ck["decoder"])
    print(f"predicting with epoch {ck['epoch']} (val kld {ck['val_kld']:.4f})")
    for name in test_names:
        ds = make_ds(name, with_gt=False, size=a.pred_size)
        predict(model, DataLoader(ds, shuffle=False, **lk), a.pred_size, dev, out / f"pred_{name}")


if __name__ == "__main__":
    main()
