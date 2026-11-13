#!/usr/bin/env python
"""Score saliency predictions against ground truth with SCOUT's metric definitions.

Every model given with --model is scored in ONE pass over the GT (each GT PNG decoded once),
under two protocols:
  scout   resize to GT size -> GaussianBlur(11, 2) -> metrics   (SCOUT test.py protocol)
  noblur  resize to GT size -> metrics                          (no post-processing)
Aggregation as SCOUT: frame -> mean per video -> mean over videos. 95% CIs by resampling
videos (shared resample indices for all models, so pairwise differences are paired).

Prediction formats (--model name=SOURCE):
  SOURCE = directory with <vid>.npz holding `frames` (int, N) and one (N,h,w) float map array
  SOURCE = file.npy  -> one constant (h,w) map used for every frame (center-bias baseline)

Example:
  python evaluate.py --manifest meta/splits/test_scored.csv --root /srv/datasets/BDDA \
    --model dinov3_b=/srv/results/dinov3/bdda_vitb16_light_s10/pred_test_common \
    --model dinov3_l=/srv/results/dinov3/bdda_vitl16_light_s10/pred_test_common \
    --model center=/srv/results/baselines/bdda_center_bias_72x128.npy \
    --out /srv/results/eval/bdda_test_scored
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gazemetrics as gm  # noqa: E402

PROTOCOLS = ("scout", "noblur")


def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def find_map_key(npz):
    keys = [k for k in npz.files if k != "frames" and npz[k].ndim == 3]
    if len(keys) != 1:
        raise ValueError(f"cannot identify map array in npz (keys={npz.files})")
    return keys[0]


class VideoGTPred(Dataset):
    """One item = one video: GT stack (uint8) + each model's predictions for exactly those frames."""

    def __init__(self, groups, root, models):
        self.groups = groups      # list of (vid, frames ndarray, gt relpaths list)
        self.root = Path(root)
        self.models = models      # list of (name, kind, source)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        vid, frames, gt_paths = self.groups[i]
        gts = np.stack([np.array(Image.open(self.root / p).convert("L")) for p in gt_paths])
        preds = {}
        for name, kind, src in self.models:
            if kind != "dir":
                continue
            f = Path(src) / f"{vid}.npz"
            with np.load(f) as d:
                pf = d["frames"].astype(np.int64)
                maps = d[find_map_key(d)]
            pos = {int(fr): j for j, fr in enumerate(pf)}
            missing = [int(fr) for fr in frames if int(fr) not in pos]
            if missing:
                raise KeyError(f"{name}: video {vid} lacks {len(missing)} frames, e.g. {missing[:5]}")
            preds[name] = maps[[pos[int(fr)] for fr in frames]]
        return vid, frames, gts, preds


def _identity(x):
    # batch_size=None would otherwise run default_convert (numpy -> tensor); keep numpy
    return x


def bootstrap_indices(n_videos, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_videos, size=(n_boot, n_videos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with vid, frame, gt (gt relative to --root)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=SOURCE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=48, help="frames per GPU batch")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-videos", type=int, default=0, help="debug: score only the first N videos")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"--out {out} exists and is not empty; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- manifest ---------------------------------------------------------------------------
    man = pd.read_csv(args.manifest)
    for col in ("vid", "frame", "gt"):
        if col not in man.columns:
            sys.exit(f"manifest lacks column {col!r}: {list(man.columns)}")
    if man.duplicated(["vid", "frame"]).any():
        sys.exit("manifest has duplicate (vid, frame) rows")
    man = man.sort_values(["vid", "frame"]).reset_index(drop=True)
    groups = [(str(v), g["frame"].to_numpy(np.int64), g["gt"].tolist()) for v, g in man.groupby("vid", sort=True)]
    if args.limit_videos:
        groups = groups[: args.limit_videos]
    n_frames = sum(len(g[1]) for g in groups)
    print(f"manifest: {len(groups)} videos, {n_frames} frames")

    # ---- models -----------------------------------------------------------------------------
    models, const_maps, provenance = [], {}, {}
    for spec in args.model:
        name, _, src = spec.partition("=")
        if not name or not src:
            sys.exit(f"bad --model {spec!r}, expected NAME=SOURCE")
        if name in [m[0] for m in models]:
            sys.exit(f"duplicate model name {name}")
        p = Path(src)
        if p.is_dir():
            models.append((name, "dir", str(p)))
            files = sorted(p.glob("*.npz"), key=lambda f: f.name)
            h = hashlib.sha256()
            for f in files:
                h.update(f.name.encode() + sha256_file(f).encode())
            provenance[name] = {"kind": "dir", "source": str(p), "n_npz": len(files), "sha256_of_npz_hashes": h.hexdigest()}
        elif p.suffix == ".npy" and p.is_file():
            m = np.load(p).astype(np.float32)
            if m.ndim != 2:
                sys.exit(f"{p}: constant map must be 2-D, got {m.shape}")
            models.append((name, "const", str(p)))
            const_maps[name] = torch.from_numpy(m).to(dev)
            provenance[name] = {"kind": "const", "source": str(p), "shape": list(m.shape), "sha256": sha256_file(p)}
        else:
            sys.exit(f"{src}: not a directory or .npy file")
    names = [m[0] for m in models]
    print("models:", ", ".join(f"{n} ({k})" for n, k, _ in models))

    # ---- scoring pass -----------------------------------------------------------------------
    ds = VideoGTPred(groups, args.root, models)
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers, collate_fn=_identity,
                    prefetch_factor=2 if args.workers else None, persistent_workers=False)
    rows = {(n, p): [] for n in names for p in PROTOCOLS}
    diag = {n: {"pred_shape": None, "zero_px": 0, "total_px": 0, "min": float("inf")} for n in names}
    gt_shape = None

    with torch.inference_mode():
        for vi, (vid, frames, gts, preds) in enumerate(dl):
            for s in range(0, len(frames), args.chunk):
                sl = slice(s, s + args.chunk)
                fr = frames[sl]
                gt = gm.gt_from_uint8(torch.from_numpy(gts[sl]).to(dev))
                if gt_shape is None:
                    gt_shape = tuple(gt.shape[-2:])
                elif tuple(gt.shape[-2:]) != gt_shape:
                    raise ValueError(f"GT size changed in video {vid}: {tuple(gt.shape[-2:])} vs {gt_shape}")
                for n in names:
                    if n in const_maps:
                        raw = const_maps[n].expand(len(fr), *const_maps[n].shape)
                    else:
                        raw = torch.from_numpy(preds[n][sl]).to(dev).float()
                    d = diag[n]
                    d["pred_shape"] = list(raw.shape[-2:])
                    d["zero_px"] += int((raw == 0).sum())
                    d["total_px"] += raw.numel()
                    d["min"] = min(d["min"], float(raw.min()))
                    up = gm.resize_to(raw, gt_shape)
                    for proto, pm in (("noblur", up), ("scout", gm.blur(up))):
                        sc = gm.all_metrics(pm, gt)
                        rows[(n, proto)].append(pd.DataFrame(
                            {"vid": vid, "frame": fr, **{k: v.cpu().numpy().astype(np.float64) for k, v in sc.items()}}))
            if vi % 25 == 0 or vi == len(groups) - 1:
                print(f"  [{vi + 1}/{len(groups)}] vid {vid}  {time.time() - t0:.0f}s", flush=True)

    # ---- per-frame table --------------------------------------------------------------------
    metrics = list(gm.METRICS)
    per_frame = pd.concat(
        [pd.concat(v, ignore_index=True).assign(model=n, protocol=p) for (n, p), v in rows.items()],
        ignore_index=True)[["model", "protocol", "vid", "frame"] + metrics]
    per_frame.to_parquet(out / "scores_per_frame.parquet", index=False)

    nan_counts = per_frame.groupby(["model", "protocol"])[metrics].apply(lambda d: d.isna().sum()).astype(int)
    if nan_counts.values.any():
        print("WARNING: NaN scores (skipped in means, as pandas/SCOUT do):\n", nan_counts[nan_counts.any(axis=1)])

    # ---- aggregation (SCOUT: mean per video, then mean over videos; NaN skipped) ------------
    per_video = per_frame.groupby(["model", "protocol", "vid"])[metrics].mean().reset_index()
    per_video.to_csv(out / "scores_per_video.csv", index=False)
    vids = sorted(per_video["vid"].unique())
    idx = bootstrap_indices(len(vids), args.n_boot, args.seed)

    mats = {}
    summary = []
    for (n, p), g in per_video.groupby(["model", "protocol"]):
        g = g.set_index("vid").loc[vids, metrics]
        X = g.to_numpy()
        mats[(n, p)] = X
        boot = np.nanmean(X[idx], axis=1)                       # (n_boot, n_metrics)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        for j, m in enumerate(metrics):
            summary.append({"model": n, "protocol": p, "metric": m, "mean": np.nanmean(X[:, j]),
                            "ci_lo": lo[j], "ci_hi": hi[j], "n_videos": len(vids),
                            "n_frames": int(per_frame[(per_frame.model == n) & (per_frame.protocol == p)].shape[0])})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    pairs = []
    for p in PROTOCOLS:
        for a, b in itertools.combinations(names, 2):
            D = mats[(a, p)] - mats[(b, p)]
            boot = np.nanmean(D[idx], axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            for j, m in enumerate(metrics):
                pairs.append({"protocol": p, "a": a, "b": b, "metric": m, "diff_a_minus_b": np.nanmean(D[:, j]),
                              "ci_lo": lo[j], "ci_hi": hi[j], "frac_boot_a_gt_b": float((boot[:, j] > 0).mean()),
                              "better": "lower" if m in gm.LOWER_IS_BETTER else "higher"})
    pd.DataFrame(pairs).to_csv(out / "pairwise.csv", index=False)

    # ---- provenance -------------------------------------------------------------------------
    for n in names:
        d = diag[n]
        d["zero_px_frac"] = d["zero_px"] / max(d["total_px"], 1)
    run = {
        "args": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        "seconds": round(time.time() - t0, 1), "host": platform.node(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": __import__("torchvision").__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
        "device": str(dev), "manifest_sha256": sha256_file(args.manifest),
        "evaluate_py_sha256": sha256_file(__file__), "gazemetrics_py_sha256": sha256_file(gm.__file__),
        "n_videos": len(vids), "n_frames": n_frames, "gt_shape": list(gt_shape),
        "protocols": {"scout": "bilinear resize (antialias) to GT size, GaussianBlur(11, sigma=2)",
                      "noblur": "bilinear resize (antialias) to GT size"},
        "metric_definitions": "SCOUT loss.py (see gazemetrics.py docstring)",
        "aggregation": "frame -> mean per video -> mean over videos; NaN skipped",
        "bootstrap": {"unit": "video", "n_boot": args.n_boot, "seed": args.seed, "ci": "percentile 95%"},
        "models": provenance, "pred_diagnostics": diag,
        "nan_counts": {f"{n}|{p}": r.to_dict() for (n, p), r in nan_counts.iterrows()},
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=float))

    # ---- console table ----------------------------------------------------------------------
    pd.set_option("display.width", 200)
    for p in PROTOCOLS:
        t = summary[summary.protocol == p].copy()
        t["value"] = t.apply(lambda r: f"{r['mean']:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
        print(f"\n=== protocol: {p}  ({len(vids)} videos, {n_frames} frames) ===")
        print(t.pivot(index="model", columns="metric", values="value")[metrics].to_string())
    for n in names:
        d = diag[n]
        flag = "  <-- check fp16 underflow" if d["zero_px_frac"] > 1e-4 else ""
        print(f"{n}: pred {d['pred_shape']}, min {d['min']:.3e}, zero px {d['zero_px_frac']:.2e}{flag}")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
