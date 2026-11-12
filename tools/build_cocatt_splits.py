#!/usr/bin/env python
"""
build_cocatt_splits.py - freeze CoCAtt frame manifests for the benchmark.

Source: the authors' processed release, manual-drive sessions only.
  camera  processed_dataset/<SUBJ>/manual_without/<RUN>/images_4hz/<k:05d>.png      (4 fps, 512x256)
  gaze    processed_dataset/<SUBJ>/manual_without/<RUN>/heatmap_4hz_20_eye_tracker/<k:05d>.png
          (GP3 eye tracker, Gaussian-smoothed, 512x256, grayscale stored as RGB,
           per-frame max-normalised to 255 - loader must convert('L') and renormalise)
Labels: final_labels.txt['labels'] (~60 Hz CARLA ticks, relative timestamps).
  Frame k (1-based) is placed at t = labels[0].timestamp + (k-1)/4 and matched to the
  nearest tick. Per-session alignment is checked (frames / (label_span*4) ~ 1).

Split: leave-subjects-out over the 11 subjects (seeded), so no participant's gaze
appears in more than one of train/val/test.

Outputs (in --out, default meta/splits):
  cocatt_train.csv, cocatt_val.csv, cocatt_test.csv
      rows: split, vid, frame, cam, gt, subject, run, intention, segment, distraction, speed_mps
      vid   = <SUBJ>_<RUN>          frame = 0-based index (file name is frame+1)
      cam/gt paths relative to --cocatt
  cocatt_excluded_frames.csv   vid, frame, reason      (empty GT map = tracker dropout)
  cocatt_sessions.csv          per-session counts and alignment ratio
  cocatt_manifest_summary.json counts, split subjects, sha256 of every output file
"""
import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import sys

import numpy as np
from PIL import Image

FPS = 4
MODE_DIR = "manual_without"
IMG_DIR = "images_4hz"
GT_DIR = "heatmap_4hz_20_eye_tracker"
INTERSECTION_INTENTS = {"GO_STRAIGHT", "LEFT_TURN", "RIGHT_TURN"}
ALIGN_TOL = 0.02
FIELDS = ["split", "vid", "frame", "cam", "gt", "subject", "run",
          "intention", "segment", "distraction", "speed_mps"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_labels(session):
    d = json.loads((session / "final_labels.txt").read_text())
    labels = d["labels"]
    ts = np.array([r["timestamp"] for r in labels], dtype=np.float64)
    if not np.all(np.diff(ts) >= 0):
        order = np.argsort(ts, kind="stable")
        ts, labels = ts[order], [labels[i] for i in order]
    return labels, ts


def nearest_label(ts, k):
    t = ts[0] + (k - 1) / FPS
    i = int(np.searchsorted(ts, t))
    if i == 0:
        return 0
    if i >= len(ts):
        return len(ts) - 1
    return i if abs(ts[i] - t) < abs(ts[i - 1] - t) else i - 1


def gt_is_empty(p):
    a = np.asarray(Image.open(p))
    return int(a.max()) == 0


def scan_session(root, subj, run, session):
    frames = sorted((session / IMG_DIR).glob("*.png"))
    labels, ts = load_labels(session)
    span = float(ts[-1] - ts[0])
    ratio = len(frames) / (span * FPS) if span > 0 else float("nan")
    vid = f"{subj}_{run}"
    rows, excluded = [], []
    n_inter = n_distr = 0
    for fr in frames:
        k = int(fr.stem)
        gt = session / GT_DIR / fr.name
        if not gt.exists():
            excluded.append((vid, k - 1, "missing_gt"))
            continue
        if gt_is_empty(gt):
            excluded.append((vid, k - 1, "empty_gt"))
            continue
        lab = labels[nearest_label(ts, k)]
        inter = lab["intention"] in INTERSECTION_INTENTS
        distr = lab["distraction state"] == "Distracted"
        n_inter += inter
        n_distr += distr
        rows.append({
            "vid": vid, "frame": k - 1,
            "cam": str(fr.relative_to(root)), "gt": str(gt.relative_to(root)),
            "subject": subj, "run": run,
            "intention": lab["intention"],
            "segment": "intersection" if inter else "lane_following",
            "distraction": "distracted" if distr else "attentive",
            "speed_mps": round(float(np.linalg.norm(lab["speed"])), 3),
        })
    summary = {
        "vid": vid, "subject": subj, "run": run,
        "n_frames": len(frames), "n_usable": len(rows), "n_excluded": len(excluded),
        "n_labels": len(labels), "label_span_s": round(span, 2),
        "align_ratio": round(ratio, 4), "align_ok": int(abs(ratio - 1) < ALIGN_TOL),
        "frac_intersection": round(n_inter / max(len(rows), 1), 3),
        "frac_distracted": round(n_distr / max(len(rows), 1), 3),
    }
    return rows, excluded, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cocatt", default="/srv/datasets/CoCAtt", help="dataset root; paths in CSVs are relative to this")
    ap.add_argument("--out", default="meta/splits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=2)
    ap.add_argument("--test-subjects", default="", help="comma list to pin test subjects instead of seeded draw")
    ap.add_argument("--val-subjects", default="", help="comma list to pin val subjects instead of seeded draw")
    args = ap.parse_args()

    root = pathlib.Path(args.cocatt)
    src = root / "processed_dataset"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows, all_excl, sessions = [], [], []
    for subj_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        mode_dir = subj_dir / MODE_DIR
        if not mode_dir.is_dir():
            continue
        for run_dir in sorted((p for p in mode_dir.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
            rows, excl, summ = scan_session(root, subj_dir.name, run_dir.name, run_dir)
            all_rows += rows
            all_excl += excl
            sessions.append(summ)
            flag = "" if summ["align_ok"] else "   <-- ALIGNMENT OFF"
            print(f"{summ['vid']}: usable={summ['n_usable']}/{summ['n_frames']} "
                  f"ratio={summ['align_ratio']:.4f} inter={summ['frac_intersection']:.2f} "
                  f"distr={summ['frac_distracted']:.2f}{flag}")

    bad = [s["vid"] for s in sessions if not s["align_ok"]]
    if bad:
        print(f"!! alignment outside tolerance: {bad}", file=sys.stderr)
        sys.exit(1)

    subjects = sorted({s["subject"] for s in sessions})
    rng = random.Random(args.seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    test = sorted(args.test_subjects.split(",")) if args.test_subjects else sorted(shuffled[:args.n_test])
    rest = [s for s in shuffled if s not in test]
    val = sorted(args.val_subjects.split(",")) if args.val_subjects else sorted(rest[:args.n_val])
    train = sorted(s for s in subjects if s not in test and s not in val)
    assign = {**{s: "train" for s in train}, **{s: "val" for s in val}, **{s: "test" for s in test}}

    written = {}
    for split in ("train", "val", "test"):
        p = out / f"cocatt_{split}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            n = 0
            for r in all_rows:
                if assign[r["subject"]] == split:
                    w.writerow({"split": split, **r})
                    n += 1
        written[p.name] = n

    p = out / "cocatt_excluded_frames.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vid", "frame", "reason"])
        w.writerows(all_excl)
    p = out / "cocatt_sessions.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sessions[0].keys()))
        w.writeheader()
        w.writerows(sessions)

    excl_by_reason = collections.Counter(e[2] for e in all_excl)
    summary = {
        "source": str(src), "mode": MODE_DIR, "gt_dir": GT_DIR, "fps": FPS,
        "seed": args.seed, "split_scheme": "leave-subjects-out",
        "subjects": {"train": train, "val": val, "test": test},
        "n_sessions": len(sessions), "n_frames_total": sum(s["n_frames"] for s in sessions),
        "n_rows": written, "n_excluded": dict(excl_by_reason),
        "sha256": {name: sha256(out / name) for name in
                   ["cocatt_train.csv", "cocatt_val.csv", "cocatt_test.csv",
                    "cocatt_excluded_frames.csv", "cocatt_sessions.csv"]},
    }
    (out / "cocatt_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nrows: {written}  excluded: {dict(excl_by_reason)}")
    print(f"subjects: train={train} val={val} test={test}")
    print(f"wrote {out}/cocatt_*.csv and cocatt_manifest_summary.json")


if __name__ == "__main__":
    main()
