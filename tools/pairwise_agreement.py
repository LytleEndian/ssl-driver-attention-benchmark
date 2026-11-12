#!/usr/bin/env python
"""Experiment 3: pairwise agreement between gaze maps (model vs model, model vs human GT).

/opt/gazehead/tools/pairwise_agreement.py

For every pair among the --model sources plus the human GT ("gt"), compute per frame:
  CC_raw   Pearson correlation of the two maps                       (gazemetrics.cc, symmetric)
  SIM_raw  histogram intersection, both maps sum-normalised           (standard symmetric SIM;
                                                                       NOT gazemetrics.sim, which is
                                                                       asymmetric by SCOUT's quirk)
  CC_cf    CC after subtracting the target center-bias map (--center) from both sum-normalised maps
  SIM_cf   SIM between the positive parts of those residuals ("above-prior mass"), each renormalised;
           NaN for a frame where either residual has no positive mass
All maps (predictions, GT, center) are resized to --size (H W) with the scorer's resize path
(bilinear, antialias=True), clamped at 0, and sum-normalised per frame.
Aggregation as evaluate.py: frame -> mean per video -> mean over videos; 95% CI by resampling
videos with shared indices. Refuses a non-empty --out.

Example (CoCAtt target):
  python /opt/gazehead/tools/pairwise_agreement.py --manifest meta/splits/cocatt_test.csv \
    --root /srv/datasets/CoCAtt --size 64 128 --center center_cocatt \
    --model center_cocatt=/srv/results/baselines/cocatt_center_bias_64x128.npy \
    --model vjepa21_l_cocatt=/srv/results/vjepa21/cocatt_vitl16_light_s2_c8s1/pred_cocatt_test \
    ... --out /srv/results/eval/cocatt_pairwise_v1
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/opt/gazehead")
import gazemetrics as gm                                    # noqa: E402
from evaluate import VideoGTPred, _identity, bootstrap_indices, sha256_file  # noqa: E402

EPS = gm.EPS


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def dir_hash(d):
    files = sorted(Path(d).glob("*.npz"))
    h = hashlib.sha256()
    for f in files:
        h.update(sha256_file(f).encode())
    return len(files), h.hexdigest()


def sum_norm(x):
    x = x.clamp_min(0)
    return x / (x.flatten(1).sum(1).view(-1, 1, 1) + EPS)


def sim_sym(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    a = a / (a.sum(1, keepdim=True) + EPS)
    b = b / (b.sum(1, keepdim=True) + EPS)
    return torch.minimum(a, b).sum(1)


def sim_pos_residual(ra, rb):
    a = ra.clamp_min(0).flatten(1)
    b = rb.clamp_min(0).flatten(1)
    sa = a.sum(1, keepdim=True)
    sb = b.sum(1, keepdim=True)
    out = torch.minimum(a / (sa + EPS), b / (sb + EPS)).sum(1)
    bad = (sa.squeeze(1) <= 0) | (sb.squeeze(1) <= 0)
    out[bad] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=SOURCE",
                    help="dir of <vid>.npz, or .npy constant map")
    ap.add_argument("--center", required=True, help="NAME of the --model that is the target center bias")
    ap.add_argument("--size", type=int, nargs=2, required=True, metavar=("H", "W"), help="working grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=512, help="frames per GPU batch")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-videos", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing non-empty --out {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")

    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    H, W = args.size

    models = []
    for spec in args.model:
        name, src = spec.split("=", 1)
        if name == "gt":
            sys.exit("the name 'gt' is reserved for the human ground truth")
        p = Path(src)
        if p.is_dir():
            models.append((name, "dir", str(p)))
        elif p.suffix == ".npy" and p.is_file():
            models.append((name, "const", str(p)))
        else:
            sys.exit(f"{src}: not a directory or .npy file")
    names = [m[0] for m in models]
    if args.center not in names:
        sys.exit(f"--center {args.center} is not among --model names {names}")
    if len(names) != len(set(names)):
        sys.exit("duplicate model names")

    const_maps = {}
    for name, kind, src in models:
        if kind == "const":
            m = torch.from_numpy(np.load(src).astype(np.float32)).unsqueeze(0)
            const_maps[name] = sum_norm(gm.resize_to(m, (H, W)).to(dev))[0]       # (H,W)
    center = const_maps[args.center]

    man = pd.read_csv(args.manifest)
    groups = [(str(v), g["frame"].to_numpy(np.int64), g["gt"].tolist()) for v, g in man.groupby("vid", sort=True)]
    if args.limit_videos:
        groups = groups[: args.limit_videos]
    n_videos = len(groups)
    n_frames = sum(len(g[1]) for g in groups)
    print(f"manifest: {n_videos} videos, {n_frames} frames; working grid {H}x{W}; device {dev}", flush=True)

    all_names = names + ["gt"]
    pairs = list(itertools.combinations(all_names, 2))
    variants = ["raw", "cf"]
    metrics = ["CC", "SIM"]

    ds = VideoGTPred(groups, args.root, models)
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers, collate_fn=_identity)

    rows = []            # per-frame records
    for vi, (vid, frames, gts, preds) in enumerate(dl):
        N = len(frames)
        per_pair = {(a, b, v, m): np.empty(N, np.float32) for (a, b) in pairs for v in variants for m in metrics}
        for s in range(0, N, args.chunk):
            sl = slice(s, min(s + args.chunk, N))
            n = sl.stop - sl.start
            maps = {}
            gt = gm.gt_from_uint8(torch.from_numpy(gts[sl]).to(dev))
            maps["gt"] = sum_norm(gm.resize_to(gt, (H, W)))
            for name, kind, src in models:
                if kind == "const":
                    maps[name] = const_maps[name].unsqueeze(0).expand(n, H, W)
                else:
                    raw = torch.from_numpy(preds[name][sl].astype(np.float32)).to(dev)
                    maps[name] = sum_norm(gm.resize_to(raw, (H, W)))
            resid = {k: v - center.unsqueeze(0) for k, v in maps.items()}
            for a, b in pairs:
                per_pair[(a, b, "raw", "CC")][sl] = gm.cc(maps[a], maps[b]).cpu().numpy()
                per_pair[(a, b, "raw", "SIM")][sl] = sim_sym(maps[a], maps[b]).cpu().numpy()
                per_pair[(a, b, "cf", "CC")][sl] = gm.cc(resid[a], resid[b]).cpu().numpy()
                per_pair[(a, b, "cf", "SIM")][sl] = sim_pos_residual(resid[a], resid[b]).cpu().numpy()
        for (a, b, v, m), arr in per_pair.items():
            rows.append(pd.DataFrame({"vid": vid, "frame": frames, "a": a, "b": b, "variant": v, "metric": m, "value": arr}))
        if (vi + 1) % 10 == 0 or vi + 1 == n_videos:
            print(f"{vi + 1}/{n_videos} videos  {time.time() - t0:.0f}s", flush=True)

    per_frame = pd.concat(rows, ignore_index=True)
    per_frame.to_parquet(out / "agreement_per_frame.parquet", index=False)
    nan_counts = per_frame.groupby(["variant", "metric"])["value"].apply(lambda s: int(s.isna().sum())).to_dict()
    per_video = per_frame.groupby(["a", "b", "variant", "metric", "vid"])["value"].mean().reset_index()
    per_video.to_csv(out / "agreement_per_video.csv", index=False)

    vids = sorted(per_video["vid"].unique())
    idx = bootstrap_indices(len(vids), args.n_boot, args.seed)
    summary = []
    for (a, b, v, m), g in per_video.groupby(["a", "b", "variant", "metric"]):
        x = g.set_index("vid").loc[vids, "value"].to_numpy(np.float64)
        if np.isnan(x).all():
            continue
        boots = np.nanmean(x[idx], axis=1)
        summary.append({"a": a, "b": b, "variant": v, "metric": m, "mean": float(np.nanmean(x)),
                        "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
                        "n_videos": len(vids), "n_frames": n_frames})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "agreement_long.csv", index=False)

    for v in variants:
        for m in metrics:
            M = pd.DataFrame(np.nan, index=all_names, columns=all_names)
            for n_ in all_names:
                M.loc[n_, n_] = 1.0
            for r in summary[(summary.variant == v) & (summary.metric == m)].itertuples():
                M.loc[r.a, r.b] = r.mean
                M.loc[r.b, r.a] = r.mean
            M.to_csv(out / f"matrix_{v}_{m}.csv", float_format="%.4f")

    src_meta = {}
    for name, kind, src in models:
        if kind == "const":
            src_meta[name] = {"kind": "const", "source": src, "sha256": sha256_file(src)}
        else:
            n_npz, h = dir_hash(src)
            src_meta[name] = {"kind": "dir", "source": src, "n_npz": n_npz, "sha256_of_npz_hashes": h}
    meta = {
        "args": vars(args), "started": started, "seconds": round(time.time() - t0, 1), "host": platform.node(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "device": str(dev),
        "manifest_sha256": sha256_file(args.manifest),
        "code_sha256": {"pairwise_agreement.py": sha256_file(__file__), "gazemetrics.py": sha256_file(gm.__file__)},
        "n_videos": n_videos, "n_frames": n_frames, "working_grid": [H, W],
        "preprocessing": "resize_to (bilinear, antialias=True) to working grid; clamp >=0; sum-normalise per frame; GT via gt_from_uint8 then same",
        "metrics": {
            "CC_raw": "gazemetrics.cc(a, b) (Pearson; symmetric)",
            "SIM_raw": "sum(min(a, b)) with both sum-normalised (standard symmetric SIM; not gazemetrics.sim)",
            "CC_cf": "gazemetrics.cc(a - center, b - center), center = sum-normalised --center map",
            "SIM_cf": "sum(min(a+, b+)) with a+ = max(a - center, 0) renormalised; NaN if either has no positive mass",
        },
        "center": args.center,
        "aggregation": "frame -> mean per video -> mean over videos; NaN skipped",
        "bootstrap": {"unit": "video", "n_boot": args.n_boot, "seed": args.seed, "ci": "percentile 95%"},
        "nan_counts": {f"{k[0]}|{k[1]}": v for k, v in nan_counts.items()},
        "sources": src_meta,
    }
    json.dump(meta, open(out / "run.json", "w"), indent=2, default=str)
    print(f"wrote {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
