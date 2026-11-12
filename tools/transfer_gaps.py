#!/usr/bin/env python3
"""transfer_gaps.py - Experiment 2 transfer gaps with paired video-level bootstrap.

Intended path: /opt/gazehead/tools/transfer_gaps.py
Env: /opt/dinov3/.venv (numpy, pandas). CPU only. Reads scores_per_video.csv from the
cross-dataset runs of record; writes only to --out (which must not exist).

Definitions (target dataset T, source dataset S != T, model family M):
  in_domain   = M trained on T, scored on T test videos
  transferred = M trained on S, scored on the same T test videos
  degradation = transferred - in_domain   for KLD (lower is better)
              = in_domain - transferred   for CC, SIM, NSS (higher is better)
  => degradation > 0 always means "worse after transfer".
Paired model comparison: diff(A,B) = degradation(A) - degradation(B) per video;
diff > 0 means A degrades more than B. 95% percentile CIs from a video-level
bootstrap using evaluate.bootstrap_indices (same resampling as pairwise_agreement.py);
one index set per target dataset is shared by all models, so comparisons are paired.
"""
import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/gazehead")
from evaluate import bootstrap_indices  # noqa: E402

METRICS = ["KLD", "CC", "SIM", "NSS"]
LOWER_BETTER = {"KLD"}
DISPLAY = {"vjepa21_l": "V-JEPA~2.1", "dinov3_l": "DINOv3", "scout_wo_task": "SCOUT",
           "salm2": "SalM$^2$", "center": "Center bias"}
DSNAME = {"bdda": "BDD-A", "cocatt": "CoCAtt"}
# Temporal input per model family. Fill in the TODOs before the table goes in the dissertation.
TEMPORAL = {
    "vjepa21_l": "8-frame clip, step round(fps*0.25) (CLIP_DECISION.txt)",
    "scout_wo_task": "TODO: SCOUT native clip",
    "dinov3_l": "TODO",
    "salm2": "TODO",
    "center": "static map (train mean)",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def boot_idx(n, n_boot, seed):
    idx = np.asarray(bootstrap_indices(n, n_boot, seed))
    if idx.ndim == 2 and idx.shape == (n, n_boot) and n != n_boot:
        idx = idx.T
    if idx.shape != (n_boot, n):
        sys.exit(f"unexpected bootstrap_indices shape {idx.shape}, expected ({n_boot}, {n})")
    return idx


def mean_ci(series, idx):
    v = series.to_numpy(dtype=float)
    boots = v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def cell(mean, lo, hi):
    return f"{mean:.3f} {{\\scriptsize[{lo:.3f}, {hi:.3f}]}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="TARGET=DIR",
                    help="cross-dataset eval run scored on TARGET, e.g. bdda=/srv/results/eval/bdda_crossdataset_v1")
    ap.add_argument("--protocol", default="noblur")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(f"--out {args.out} exists; use a new directory")
    t0 = time.time()
    runs = dict(r.split("=", 1) for r in args.run)
    datasets = sorted(runs)
    os.makedirs(args.out)

    gap_rows, pair_rows, pv_rows, inputs, protocols_seen = [], [], [], {}, {}
    for tgt in datasets:
        f = os.path.join(runs[tgt], "scores_per_video.csv")
        inputs[f] = sha256_file(f)
        df = pd.read_csv(f, dtype={"vid": str})
        protocols_seen[tgt] = sorted(df["protocol"].unique())
        if args.protocol not in protocols_seen[tgt]:
            sys.exit(f"protocol {args.protocol!r} not in {f}: {protocols_seen[tgt]}")
        df = df[df["protocol"] == args.protocol]
        vids = sorted(df["vid"].unique())
        n = len(vids)
        idx = boot_idx(n, args.n_boot, args.seed)
        models = set(df["model"].unique())

        for src in datasets:
            if src == tgt:
                continue
            fams = sorted({m[: -len(tgt) - 1] for m in models if m.endswith("_" + tgt)}
                          & {m[: -len(src) - 1] for m in models if m.endswith("_" + src)})
            deg = {}
            for fam in fams:
                for m in METRICS:
                    a = df[df["model"] == f"{fam}_{tgt}"].set_index("vid")[m].reindex(vids)
                    b = df[df["model"] == f"{fam}_{src}"].set_index("vid")[m].reindex(vids)
                    if a.isna().any() or b.isna().any():
                        sys.exit(f"missing per-video values for {fam} {m} on {tgt}")
                    d = (b - a) if m in LOWER_BETTER else (a - b)
                    deg[(fam, m)] = d
                    mean, lo, hi = mean_ci(d, idx)
                    gap_rows.append(dict(target=tgt, source=src, model=fam, metric=m, n_videos=n,
                                         in_domain=float(a.mean()), transferred=float(b.mean()),
                                         degradation=mean, ci_lo=lo, ci_hi=hi,
                                         rel_pct=100.0 * mean / abs(float(a.mean())),
                                         videos_worse=int((d > 0).sum()),
                                         temporal=TEMPORAL.get(fam, "TODO")))
                    for v in vids:
                        pv_rows.append(dict(target=tgt, source=src, model=fam, metric=m, vid=v,
                                            in_domain=float(a[v]), transferred=float(b[v]),
                                            degradation=float(d[v])))
            for fa, fb in itertools.combinations(fams, 2):
                for m in METRICS:
                    diff = deg[(fa, m)] - deg[(fb, m)]
                    mean, lo, hi = mean_ci(diff, idx)
                    pair_rows.append(dict(target=tgt, source=src, model_a=fa, model_b=fb, metric=m,
                                          n_videos=n, diff=mean, ci_lo=lo, ci_hi=hi,
                                          ci_excludes_0=bool(lo > 0 or hi < 0),
                                          videos_a_degrades_less=int((diff < 0).sum())))

    gaps = pd.DataFrame(gap_rows)
    pairs = pd.DataFrame(pair_rows)
    gaps.to_csv(os.path.join(args.out, "transfer_gaps.csv"), index=False)
    pairs.to_csv(os.path.join(args.out, "transfer_gap_pairs.csv"), index=False)
    pd.DataFrame(pv_rows).to_csv(os.path.join(args.out, "transfer_gaps_per_video.csv"), index=False)

    # LaTeX: degradation per model and direction, mean [95% CI]
    L = [r"\begin{tabular}{llllll}", r"\toprule",
         r"Transfer & Model & $\Delta$KLD & $\Delta$CC & $\Delta$SIM & $\Delta$NSS \\", r"\midrule"]
    for (tgt, src), g in gaps.groupby(["target", "source"], sort=False):
        first = True
        for fam, gm in g.groupby("model", sort=False):
            r = gm.set_index("metric")
            cells = [cell(r.loc[m, "degradation"], r.loc[m, "ci_lo"], r.loc[m, "ci_hi"]) for m in METRICS]
            lab = f"{DSNAME.get(src, src)}$\\to${DSNAME.get(tgt, tgt)}" if first else ""
            L.append(f"{lab} & {DISPLAY.get(fam, fam)} & " + " & ".join(cells) + r" \\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(args.out, "exp2_tab_transfer_gaps.tex"), "w").write("\n".join(L) + "\n")

    L = [r"\begin{tabular}{llllll}", r"\toprule",
         r"Transfer & A vs.\ B & KLD & CC & SIM & NSS \\", r"\midrule"]
    for (tgt, src), g in pairs.groupby(["target", "source"], sort=False):
        first = True
        for (fa, fb), gp in g.groupby(["model_a", "model_b"], sort=False):
            r = gp.set_index("metric")
            cells = [cell(r.loc[m, "diff"], r.loc[m, "ci_lo"], r.loc[m, "ci_hi"]) for m in METRICS]
            lab = f"{DSNAME.get(src, src)}$\\to${DSNAME.get(tgt, tgt)}" if first else ""
            L.append(f"{lab} & {DISPLAY.get(fa, fa)} vs.\\ {DISPLAY.get(fb, fb)} & " + " & ".join(cells) + r" \\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(args.out, "exp2_tab_transfer_gap_pairs.tex"), "w").write("\n".join(L) + "\n")

    with open(os.path.join(args.out, "report.txt"), "w") as rep:
        pd.set_option("display.width", 200)
        for (tgt, src), g in gaps.groupby(["target", "source"], sort=False):
            nv = int(g["n_videos"].iloc[0])
            rep.write(f"== {src} -> {tgt}  ({nv} target videos) ==\n")
            if nv < 10:
                rep.write(f"NOTE: only {nv} videos; treat bootstrap CIs as descriptive, not inferential.\n")
            rep.write(g[["model", "metric", "in_domain", "transferred", "degradation", "ci_lo", "ci_hi",
                         "rel_pct", "videos_worse"]].round(3).to_string(index=False) + "\n\n")
            gp = pairs[(pairs.target == tgt) & (pairs.source == src)]
            rep.write("paired differences in degradation (diff>0: A degrades more):\n")
            rep.write(gp[["model_a", "model_b", "metric", "diff", "ci_lo", "ci_hi", "ci_excludes_0",
                          "videos_a_degrades_less"]].round(3).to_string(index=False) + "\n\n")
        todo = sorted({f for f in gaps["model"].unique() if "TODO" in TEMPORAL.get(f, "TODO")})
        if todo:
            rep.write(f"TODO: fill TEMPORAL entries for {todo} before using the table.\n")
    print(open(os.path.join(args.out, "report.txt")).read())

    json.dump({
        "args": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        "seconds": round(time.time() - t0, 1), "host": platform.node(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "evaluate_py_sha256": sha256_file("/opt/gazehead/evaluate.py"),
        "inputs_sha256": inputs, "protocols_seen": protocols_seen,
        "definitions": "degradation>0 = worse after transfer; diff(A,B)=deg(A)-deg(B); "
                       "video-level paired bootstrap, percentile 95% CI, shared indices per target",
        "temporal": TEMPORAL,
    }, open(os.path.join(args.out, "run.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
