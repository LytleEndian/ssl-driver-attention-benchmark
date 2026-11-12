#!/usr/bin/env python
"""shuffled_agreement.py - Experiment 3 sensitivity check: matched vs shuffled-frame agreement.

Intended path: /opt/gazehead/tools/shuffled_agreement.py
Env: /opt/dinov3/.venv. Writes only to --out (refuses a non-empty --out).

Committee comment 5: subtracting the mean gaze map (CC_cf / SIM_cf) need not remove all shared
spatial structure. This measures how much agreement survives when the two maps come from
unrelated frames.

For each pair (a, b) among the prediction directories and the human GT ("gt"):
  matched   a and b from the same frame (as pairwise_agreement.py)
  shuffled  a from frame i of video v; b from video u = pi(v), a seeded derangement (u != v),
            at the same relative position: j = round(i * (n_u - 1) / (n_v - 1))
  gap       matched - shuffled per video (shuffled pooled over --n-derange derangements)
Preprocessing and metrics are pairwise_agreement.py's own functions (imported, not copied):
resize to the working grid (bilinear, antialias), clamp >= 0, sum-normalise per frame;
CC_raw, SIM_raw, CC_cf, SIM_cf with the sum-normalised target center map.
Constant (.npy) models are skipped (no frame-specific map to shuffle); the center is used for cf only.
Aggregation: frame -> video mean -> mean over videos; 95% percentile CI on the gap by video-level
bootstrap (evaluate.bootstrap_indices, shared indices across pairs).
--group-by-subject keeps derangements within a vid prefix (CoCAtt J_*, K_*).
--frame-stride subsamples frames identically for matched and shuffled.
--reference <pairwise run>/agreement_per_video.csv checks matched values (exact at stride 1).
"""
import argparse
import itertools
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/opt/gazehead")
sys.path.insert(0, "/opt/gazehead/tools")
import gazemetrics as gm                                                   # noqa: E402
from evaluate import VideoGTPred, _identity, bootstrap_indices, sha256_file  # noqa: E402
from pairwise_agreement import sum_norm, sim_sym, sim_pos_residual, dir_hash  # noqa: E402

VARIANTS = ["raw", "cf"]
METRICS = ["CC", "SIM"]


def derangement(items, rng):
    n = len(items)
    if n < 2:
        sys.exit(f"cannot derange a group of {n} video(s): {items}")
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return {items[i]: items[p[i]] for i in range(n)}


def pair_metrics(A, B, center):
    rA, rB = A - center, B - center
    return {("raw", "CC"): gm.cc(A, B), ("raw", "SIM"): sim_sym(A, B),
            ("cf", "CC"): gm.cc(rA, rB), ("cf", "SIM"): sim_pos_residual(rA, rB)}


def nanmean(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-run", required=True, help="run.json of the pairwise_agreement run to mirror")
    ap.add_argument("--repo", default="/opt/gazehead", help="base for a relative manifest path")
    ap.add_argument("--exclude", action="append", default=[], metavar="NAME")
    ap.add_argument("--n-derange", type=int, default=5)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--group-by-subject", action="store_true")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing non-empty --out {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")

    src = json.load(open(args.from_run))
    ra = src["args"]
    manifest = Path(ra["manifest"])
    if not manifest.is_absolute():
        manifest = Path(args.repo) / manifest
    H, W = ra["size"]
    specs = [s.split("=", 1) for s in ra["model"]]
    srcmap = dict(specs)
    center_path = srcmap[ra["center"]]
    models = [(n, "dir", s) for n, s in specs if Path(s).is_dir() and n not in args.exclude]
    skipped = [n for n, s in specs if not Path(s).is_dir()] + [n for n in args.exclude if n in srcmap]
    names = [m[0] for m in models]
    all_names = names + ["gt"]
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    c = torch.from_numpy(np.load(center_path).astype(np.float32)).unsqueeze(0)
    center = sum_norm(gm.resize_to(c, (H, W)).to(dev))[0]

    man = pd.read_csv(manifest)
    groups = [(str(v), g["frame"].to_numpy(np.int64), g["gt"].tolist()) for v, g in man.groupby("vid", sort=True)]
    if args.limit_videos:
        groups = groups[: args.limit_videos]
    print(f"{len(groups)} videos; grid {H}x{W}; stride {args.frame_stride}; models {names}; "
          f"skipped {skipped}; device {dev}", flush=True)

    # ---- load once: sum-normalised maps on the working grid, kept on CPU ----
    store = {n: {} for n in all_names}
    ds = VideoGTPred(groups, ra["root"], models)
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=args.workers, collate_fn=_identity)
    for vi, (vid, frames, gts, preds) in enumerate(dl):
        vid = str(vid)
        keep = np.arange(0, len(frames), args.frame_stride)
        parts = {n: [] for n in all_names}
        for s in range(0, len(keep), args.chunk):
            k = keep[s: s + args.chunk]
            gt = gm.gt_from_uint8(torch.from_numpy(np.ascontiguousarray(gts[k])).to(dev))
            parts["gt"].append(sum_norm(gm.resize_to(gt, (H, W))).float().cpu())
            for n in names:
                raw = torch.from_numpy(preds[n][k].astype(np.float32)).to(dev)
                parts[n].append(sum_norm(gm.resize_to(raw, (H, W))).float().cpu())
        for n in all_names:
            store[n][vid] = torch.cat(parts[n])
        if (vi + 1) % 25 == 0 or vi + 1 == len(groups):
            print(f"loaded {vi + 1}/{len(groups)}  {time.time() - t0:.0f}s", flush=True)

    # ---- derangements ----
    vids = sorted(store["gt"])
    key = (lambda v: v.split("_")[0]) if args.group_by_subject else (lambda v: "all")
    grp = {}
    for v in vids:
        grp.setdefault(key(v), []).append(v)
    rng = np.random.default_rng(args.seed)
    perms = []
    for _ in range(args.n_derange):
        p = {}
        for g in sorted(grp):
            p.update(derangement(grp[g], rng))
        perms.append(p)
    pd.DataFrame([{"derangement": k, "vid": v, "partner": u} for k, p in enumerate(perms) for v, u in p.items()]
                 ).to_csv(out / "derangements.csv", index=False)

    # ---- matched vs shuffled ----
    pairs = list(itertools.combinations(all_names, 2))
    rows = []
    for vi, v in enumerate(vids):
        n_v = store["gt"][v].shape[0]
        i = np.arange(n_v)
        dv = {n: store[n][v].to(dev) for n in all_names}
        partners = []
        for p in perms:
            u = p[v]
            n_u = store["gt"][u].shape[0]
            j = np.zeros(n_v, np.int64) if n_v == 1 else np.rint(i * (n_u - 1) / (n_v - 1)).astype(np.int64)
            jt = torch.from_numpy(j)
            partners.append({n: store[n][u][jt].to(dev) for n in all_names})
        for a, b in pairs:
            m = {k: t.float().cpu().numpy() for k, t in pair_metrics(dv[a], dv[b], center).items()}
            sh = [{k: t.float().cpu().numpy() for k, t in pair_metrics(dv[a], du[b], center).items()} for du in partners]
            for var, met in itertools.product(VARIANTS, METRICS):
                rows.append({"a": a, "b": b, "variant": var, "metric": met, "vid": v, "n_frames": n_v,
                             "matched": nanmean(m[(var, met)]),
                             "shuffled": nanmean(np.concatenate([s[(var, met)] for s in sh]))})
        if (vi + 1) % 25 == 0 or vi + 1 == len(vids):
            print(f"scored {vi + 1}/{len(vids)}  {time.time() - t0:.0f}s", flush=True)

    per_video = pd.DataFrame(rows)
    per_video["gap"] = per_video["matched"] - per_video["shuffled"]
    per_video.to_csv(out / "shuffle_per_video.csv", index=False)

    ref_check = None
    if args.reference:
        ref = pd.read_csv(args.reference, dtype={"vid": str})
        mrg = per_video.merge(ref, on=["a", "b", "variant", "metric", "vid"], how="inner")
        d = (mrg["matched"] - mrg["value"]).abs()
        ref_check = {"reference": args.reference, "n_compared": int(len(mrg)),
                     "max_abs_diff": float(d.max()) if len(mrg) else None,
                     "max_abs_diff_by": {f"{k[0]}_{k[1]}": float(x) for k, x in
                                         mrg.assign(d=d).groupby(["variant", "metric"])["d"].max().items()},
                     "note": "exact agreement expected only at --frame-stride 1"}

    idx = bootstrap_indices(len(vids), args.n_boot, args.seed)
    summary = []
    for (a, b, var, met), g in per_video.groupby(["a", "b", "variant", "metric"], sort=False):
        g = g.set_index("vid").loc[vids]
        xm, xs = g["matched"].to_numpy(float), g["shuffled"].to_numpy(float)
        gap = xm - xs
        if np.isnan(gap).all():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            boots = np.nanmean(gap[idx], axis=1)
        summary.append({"a": a, "b": b, "variant": var, "metric": met, "matched": nanmean(xm),
                        "shuffled": nanmean(xs), "gap": nanmean(gap),
                        "ci_lo": float(np.nanpercentile(boots, 2.5)), "ci_hi": float(np.nanpercentile(boots, 97.5)),
                        "videos_gap_pos": int(np.sum(gap > 0)), "n_videos": len(vids)})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "shuffle_summary.csv", index=False)

    if args.group_by_subject:
        pv = per_video.assign(subject=per_video["vid"].map(key))
        pv.groupby(["subject", "a", "b", "variant", "metric"])[["matched", "shuffled", "gap"]].mean() \
          .reset_index().to_csv(out / "shuffle_by_subject.csv", index=False)

    # ---- LaTeX: model vs human, CC raw and center-removed ----
    hum = summary[summary["b"] == "gt"]
    order = hum[(hum.variant == "raw") & (hum.metric == "CC")].sort_values("matched", ascending=False)["a"].tolist()
    L = [r"\begin{tabular}{lrrlrrl}", r"\toprule",
         r" & \multicolumn{3}{c}{CC} & \multicolumn{3}{c}{CC, center removed} \\",
         r"Model & matched & shuffled & gap & matched & shuffled & gap \\", r"\midrule"]
    for mname in order:
        cells = []
        for var in VARIANTS:
            r = hum[(hum.a == mname) & (hum.variant == var) & (hum.metric == "CC")].iloc[0]
            cells += [f"{r.matched:.3f}", f"{r.shuffled:.3f}",
                      f"{r.gap:.3f} {{\\scriptsize[{r.ci_lo:.3f}, {r.ci_hi:.3f}]}}"]
        L.append(mname.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "exp3_tab_shuffle_human.tex").write_text("\n".join(L) + "\n")

    pd.set_option("display.width", 220)
    rep = [f"source run: {args.from_run}", f"videos {len(vids)}, stride {args.frame_stride}, "
           f"derangements {args.n_derange}, group_by_subject {args.group_by_subject}", ""]
    if ref_check:
        rep += [f"reference check: {json.dumps(ref_check, indent=1)}", ""]
    rep += ["== model vs human (b = gt) ==",
            hum.sort_values(["variant", "metric", "matched"], ascending=[True, True, False])
               .round(3).to_string(index=False), "",
            "== model vs model ==",
            summary[summary["b"] != "gt"].round(3).to_string(index=False)]
    (out / "report.txt").write_text("\n".join(rep) + "\n")
    print((out / "report.txt").read_text().split("== model vs model ==")[0])

    json.dump({
        "args": vars(args), "from_run_sha256": sha256_file(args.from_run), "mirrored_args": ra,
        "started": started, "seconds": round(time.time() - t0, 1), "host": platform.node(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "device": str(dev), "script_sha256": sha256_file(__file__),
        "pairwise_agreement_py_sha256": sha256_file("/opt/gazehead/tools/pairwise_agreement.py"),
        "evaluate_py_sha256": sha256_file("/opt/gazehead/evaluate.py"),
        "gazemetrics_py_sha256": sha256_file("/opt/gazehead/gazemetrics.py"),
        "manifest_sha256": sha256_file(str(manifest)), "center_sha256": sha256_file(center_path),
        "pred_dirs": {n: list(dir_hash(Path(s))) for n, _, s in models},
        "models_used": names, "models_skipped": skipped, "n_videos": len(vids),
        "reference_check": ref_check,
        "definitions": "shuffled: b from deranged video at same relative frame position; "
                       "gap = matched - shuffled per video; video-level bootstrap on gap",
    }, open(out / "run.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
