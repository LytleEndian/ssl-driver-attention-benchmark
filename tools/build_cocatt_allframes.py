#!/usr/bin/env python
"""
build_cocatt_allframes.py - every camera frame of the CoCAtt sessions used by the benchmark,
so that video models get real clip history across eye-tracker dropout.

The benchmark manifests (cocatt_{train,val,test}.csv, from build_cocatt_splits.py) contain only
frames with a valid gaze map. Dropout frames (empty / missing GT) are scattered through the
sessions (1-3 frames typically, holes up to ~28 frames, ~30% of subject I), so an 8-frame
step-1 clip ending on a valid frame often needs frames that are not in the manifests.

This writes cocatt_allframes.csv: one row per camera frame of every session in the three
manifests, same columns as the frame-cache index (split, vid, frame, cam, gt), with gt = ""
for dropout frames. build_frame_cache.py stores an all-zero GT slot for those rows; they are
never used as a training target or scored, because CachedFrames / CachedClips take their
target rows from the filtered manifests, not from the cache index.

  python tools/build_cocatt_allframes.py            # -> meta/splits/cocatt_allframes.csv (+ _summary.json)

Checks: every manifest row is present with an identical gt path; the dropout rows equal the
set in cocatt_excluded_frames.csv.
"""
import argparse
import csv
import hashlib
import json
import pathlib

MODE_DIR = "manual_without"
IMG_DIR = "images_4hz"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cocatt", default="/srv/datasets/CoCAtt")
    ap.add_argument("--splits", default="meta/splits")
    ap.add_argument("--manifests", default="cocatt_train,cocatt_val,cocatt_test")
    ap.add_argument("--out", default="cocatt_allframes", help="output manifest name under --splits")
    a = ap.parse_args()
    root, splits = pathlib.Path(a.cocatt), pathlib.Path(a.splits)

    # sessions and valid (gt-bearing) rows from the benchmark manifests
    sessions = {}                       # vid -> (split, subject, run)
    valid = {}                          # (vid, frame) -> gt path
    for name in a.manifests.split(","):
        for r in csv.DictReader(open(splits / f"{name}.csv")):
            sessions.setdefault(r["vid"], (r["split"], r["subject"], r["run"]))
            if sessions[r["vid"]][0] != r["split"]:
                raise SystemExit(f"{r['vid']} appears in two splits")
            valid[(r["vid"], int(r["frame"]))] = r["gt"]

    rows, n_drop = [], 0
    per_vid = {}
    for vid, (split, subj, run) in sorted(sessions.items()):
        sess = root / "processed_dataset" / subj / MODE_DIR / run
        frames = sorted((sess / IMG_DIR).glob("*.png"))
        if not frames:
            raise SystemExit(f"no frames under {sess / IMG_DIR}")
        n_v = 0
        for fr in frames:
            k = int(fr.stem) - 1                                   # 0-based, as in the manifests
            gt = valid.get((vid, k), "")
            n_v += bool(gt)
            n_drop += not gt
            rows.append([split, vid, k, str(fr.relative_to(root)), gt])
        per_vid[vid] = {"split": split, "n_frames": len(frames), "n_valid": n_v, "n_dropout": len(frames) - n_v}

    # checks
    present = {(r[1], r[2]): r[4] for r in rows}
    missing = [k for k in valid if k not in present]
    mismatch = [k for k, g in valid.items() if k in present and present[k] != g]
    if missing or mismatch:
        raise SystemExit(f"manifest rows missing from scan: {missing[:5]} ({len(missing)}); gt mismatch: {mismatch[:5]}")
    excl_path = splits / "cocatt_excluded_frames.csv"
    if excl_path.is_file():
        excl = {(r["vid"], int(r["frame"])) for r in csv.DictReader(open(excl_path)) if r["vid"] in sessions}
        drop = {k for k, g in present.items() if not g}
        if excl != drop:
            raise SystemExit(f"dropout set differs from cocatt_excluded_frames.csv: "
                             f"only here {sorted(drop - excl)[:5]} ({len(drop - excl)}), "
                             f"only there {sorted(excl - drop)[:5]} ({len(excl - drop)})")

    out = splits / f"{a.out}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "vid", "frame", "cam", "gt"])
        w.writerows(rows)
    summary = {
        "n_rows": len(rows), "n_valid": len(valid), "n_dropout": n_drop,
        "dropout_frac": round(n_drop / len(rows), 4),
        "source_manifests": {m: sha256(splits / f"{m}.csv") for m in a.manifests.split(",")},
        "sha256": {out.name: sha256(out)},
        "sessions": per_vid,
    }
    json.dump(summary, open(splits / f"{a.out}_summary.json", "w"), indent=2)
    print(f"{len(rows)} frames, {len(valid)} valid, {n_drop} dropout ({summary['dropout_frac']:.1%}) -> {out}")


if __name__ == "__main__":
    main()
