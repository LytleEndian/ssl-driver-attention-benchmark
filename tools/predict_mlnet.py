#!/usr/bin/env python
"""Zero-shot ML-Net predictions in the gazehead prediction contract.

Intended path: /opt/gazehead/tools/predict_mlnet.py

Writes <out>/pred_test_common/<vid>.npz (frames + float32 `sal` (N,45,80) for 16:9 input — the native
60x80 map with the padding rows removed) and <out>/run.json. Dataset-agnostic: reads `cam` paths from
the manifest relative to --root.

Check mode (no predictions): --check-frames 12 --check-dir /srv/results/mlnet/check
  prints weight names / param count and writes frame | overlay | raw-map PNGs to look at.

Decisions (also written to run.json):
  input 480x640, aspect-preserving resize + centred zero pad (utilities.padding), BGR, Caffe means,
  kernel flip and pooling variant as chosen by check_mlnet_port.py, prior upsampled bilinear
  (align_corners=False), output = native 1/8-res map after the final relu, padding removed by
  half-pixel linear resampling (mlnet_torch.crop_native), no blur, no normalisation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/opt/gazehead")
import mlnet_torch as M  # noqa: E402
import zeroshot_io as Z  # noqa: E402


class Frames(Dataset):
    def __init__(self, rows, root):
        self.rows, self.root = rows, Path(root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(self.root / r["cam"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / r["cam"])
        x, box = M.preprocess_bgr(img)
        return torch.from_numpy(x), i, torch.tensor(box, dtype=torch.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="run dir, e.g. /srv/results/mlnet/bdda_v1")
    ap.add_argument("--weights", default="/srv/models/mlnet/mlnet_salicon_weights.pkl")
    ap.add_argument("--kernel-flip", choices=["yes", "no"], required=True, help="from check_mlnet_port.py")
    ap.add_argument("--pool", choices=["theano", "standard"], required=True, help="from check_mlnet_port.py")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=0, help="smoke test: only the first N videos")
    ap.add_argument("--check-frames", type=int, default=0)
    ap.add_argument("--check-dir", default=None)
    args = ap.parse_args()

    print("mlnet_torch from", M.__file__, "| zeroshot_io from", Z.__file__)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = True

    rows = Z.read_manifest(args.manifest)
    vids = sorted({r["vid"] for r in rows})
    if args.max_videos:
        keep = set(vids[:args.max_videos])
        rows = [r for r in rows if r["vid"] in keep]
        vids = sorted(keep)
    print(f"{len(rows)} frames / {len(vids)} videos from {args.manifest}")

    model = M.MLNet(pool=args.pool).to(dev).eval()
    names = M.load_keras_weights(model, args.weights, kernel_flip=(args.kernel_flip == "yes"))
    npar = M.n_params(model)
    print(f"loaded {len(names)} arrays from {args.weights}; params {npar:,} (expected {M.N_PARAMS_EXPECTED:,})")
    assert npar == M.N_PARAMS_EXPECTED

    if args.check_frames:
        cdir = Path(args.check_dir or Path(args.out).parent / "check")
        for r in Z.pick_check_rows(rows, args.check_frames):
            img = cv2.imread(str(Path(args.root) / r["cam"]), cv2.IMREAD_COLOR)
            x, box = M.preprocess_bgr(img)
            with torch.no_grad():
                pred = model(torch.from_numpy(x)[None].to(dev))[0, 0].cpu().numpy()
            sal = M.crop_native(pred, box)
            Z.overlay_png(img, sal, cdir / f"{r['vid']}_{r['frame']:05d}.png")
            print(f"{r['vid']} {r['frame']:5d} native {pred.shape} -> saved {sal.shape} "
                  f"max {sal.max():.3f} mean {sal.mean():.3f}")
        print(f"overlays in {cdir}")
        return

    out = Path(args.out)
    pred_dir = out / "pred_test_common"
    if pred_dir.exists() and any(pred_dir.iterdir()):
        raise SystemExit(f"{pred_dir} is not empty — use a new run directory")
    writer = Z.VideoWriter(pred_dir, key="sal")
    dl = DataLoader(Frames(rows, args.root), batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    t0 = time.time()
    done = 0
    native_shape = saved_shape = None
    with torch.no_grad():
        for x, idx, boxes in dl:
            pred = model(x.to(dev, non_blocking=True))[:, 0].cpu().numpy()
            for p, i, b in zip(pred, idx.tolist(), boxes.tolist()):
                sal = M.crop_native(p, tuple(b))
                native_shape, saved_shape = p.shape, sal.shape
                writer.add(rows[i]["vid"], rows[i]["frame"], sal)
            writer.flush()
            done += len(idx)
            if done % (args.bs * 50) < args.bs:
                el = time.time() - t0
                print(f"{done}/{len(rows)} frames, {writer.written} videos written, "
                      f"{done / el:.1f} fps, eta {(len(rows) - done) / (done / el) / 60:.1f} min", flush=True)
    writer.flush(keep_last=False)
    el = time.time() - t0
    print(f"done: {writer.frames_written} frames / {writer.written} videos in {el / 60:.1f} min")

    Z.write_run_json(out, {
        "model": "ML-Net (Cornia et al., ICPR 2016), zero-shot, authors' SALICON weights",
        "port": {"file": M.__file__, "gazehead_commit": Z.git_commit("/opt/gazehead"),
                 "reference_repo": "/opt/mlnet", "reference_commit": Z.git_commit("/opt/mlnet")},
        "weights": {"path": args.weights, "sha256": Z.sha256_file(args.weights), "arrays": names, "n_params": npar},
        "manifest": {"path": args.manifest, "sha256": Z.sha256_file(args.manifest),
                     "frames": len(rows), "videos": len(vids)},
        "decisions": {
            "input": "480x640, aspect-preserving resize + centred zero pad (utilities.padding), BGR, "
                     "Caffe mean subtraction 103.939/116.779/123.68, no scaling",
            "kernel_flip": args.kernel_flip, "pool_variant": args.pool,
            "prior_upsampling": "bilinear x10, align_corners=False",
            "output": "native 1/8-res map after final relu; padding removed by half-pixel linear resampling "
                      "(crop_native); no blur, no normalisation; evaluate.py resizes + blurs",
            "native_shape": list(native_shape), "saved_shape": list(saved_shape),
            "dropout": "identity (eval)", "precision": "fp32",
        },
        "timing": {"seconds": el, "fps": len(rows) / el, "batch_size": args.bs, "gpu_index": args.gpu},
        "outputs": {"pred_dir": str(pred_dir), "npz_key": "sal", "videos_written": writer.written,
                    "frames_written": writer.frames_written},
    })
    print(f"run.json written to {out / 'run.json'}")


if __name__ == "__main__":
    main()
