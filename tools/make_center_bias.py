#!/usr/bin/env python
"""Center-bias baseline: mean of per-frame sum-normalised GT maps over a TRAINING manifest.

The map is area-downsampled to the prediction grid (default 72x128, an exact 8x8 block mean
of the 576x1024 BDD-A GT) and saved as a .npy, so evaluate.py pushes it through exactly the
same resize -> (blur) -> metrics path as every model.

  python tools/make_center_bias.py --manifest meta/splits/train.csv --root /srv/datasets/BDDA \
      --out /srv/results/baselines/bdda_center_bias_72x128.npy
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _chunk_sum(job):
    root, paths = job
    acc, n, skipped = None, 0, 0
    for p in paths:
        g = np.asarray(Image.open(Path(root) / p).convert("L"), dtype=np.float64)
        s = g.sum()
        if s <= 0:
            skipped += 1
            continue
        acc = g / s if acc is None else acc + g / s
        n += 1
    return acc, n, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help=".npy path")
    ap.add_argument("--grid", default="72x128", help="HxW of the saved map (must divide the GT size)")
    ap.add_argument("--stride", type=int, default=1, help="use every k-th row of the manifest")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest).sort_values(["vid", "frame"])
    paths = man["gt"].tolist()[:: args.stride]
    chunks = [(args.root, paths[i: i + 500]) for i in range(0, len(paths), 500)]
    total, n, skipped = None, 0, 0
    with Pool(args.workers) as pool:
        for k, (acc, c, sk) in enumerate(pool.imap_unordered(_chunk_sum, chunks)):
            if acc is not None:
                total = acc if total is None else total + acc
            n += c
            skipped += sk
            if k % 50 == 0:
                print(f"  {k + 1}/{len(chunks)} chunks, {n} frames", flush=True)
    mean = total / n
    H, W = mean.shape
    h, w = (int(x) for x in args.grid.split("x"))
    if H % h or W % w:
        raise SystemExit(f"grid {h}x{w} does not divide GT {H}x{W}")
    small = mean.reshape(h, H // h, w, W // w).mean(axis=(1, 3))
    small = (small / small.sum()).astype(np.float32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, small)
    np.save(out.with_name(out.stem + f"_full{H}x{W}.npy"), (mean / mean.sum()).astype(np.float32))
    meta = {"manifest": args.manifest,
            "manifest_sha256": hashlib.sha256(open(args.manifest, "rb").read()).hexdigest(),
            "frames_used": n, "frames_skipped_empty": skipped, "stride": args.stride,
            "gt_shape": [H, W], "grid": [h, w],
            "definition": "mean over frames of GT/sum(GT); block-mean to grid; renormalised to sum 1",
            "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
