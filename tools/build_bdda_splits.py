#!/usr/bin/env python
"""
build_bdda_splits.py - freeze BDD-A frame manifests for the benchmark.

Frame index = our 29 fps extraction (tools/extract_bdda_frames.sh), 0-based:
  camera  <split>/camera_frames/<vid>/<frame:05d>.jpg
  gaze    <split>/gazemap_frames/<vid>/<frame:05d>.png
Usable frames per video: 0 .. min(n_camera, n_gaze) - 1.

Outputs (in --out):
  train.csv            official training/   minus SCOUT exclusions, unreadable, unextracted
  val.csv              official validation/ minus the same
  test_common.csv      the benchmark test set = exactly the frames SCOUT scores
                       (SCOUT BDDADataset(mode="test") sample_list; gt = start + 15)
  test_salm2_own.csv   SalM2's own test.json frames mapped to our index
                       (ours = theirs - first index of that video) - for its reproduction check only
  excluded_videos.csv  split, vid, reason
  manifest_summary.json  counts + sha256 of every input and output file

Every CSV row: split, vid, frame, cam, gt  (paths relative to --bdda)
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import pathlib

SPLITS = {"training": "train", "validation": "val", "test": "test"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def n_frames(bdda, split, vid):
    c = bdda / split / "camera_frames" / str(vid)
    g = bdda / split / "gazemap_frames" / str(vid)
    if not (c.is_dir() and g.is_dir()):
        return 0
    return min(sum(1 for x in os.scandir(c) if x.name.endswith(".jpg")),
               sum(1 for x in os.scandir(g) if x.name.endswith(".png")))


def row(split, vid, fr):
    return {"split": split, "vid": vid, "frame": fr,
            "cam": f"{split}/camera_frames/{vid}/{fr:05d}.jpg",
            "gt": f"{split}/gazemap_frames/{vid}/{fr:05d}.png"}


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["split", "vid", "frame", "cam", "gt"])
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdda", default="/srv/datasets/BDDA")
    ap.add_argument("--scout-exclude", default="/opt/SCOUT/extra_annotations/BDD-A/exclude_videos.json")
    ap.add_argument("--scout-test", default="meta/splits/scout_test_samples.csv")
    ap.add_argument("--salm2-test", default="/opt/SaliencyMamba/datasets/BDDA/test.json")
    ap.add_argument("--out", default="meta/splits")
    a = ap.parse_args()
    bdda, out = pathlib.Path(a.bdda), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    excl = {k: {int(x) for x in v} for k, v in json.load(open(a.scout_exclude)).items()}
    excluded, nfr = [], {}
    keep = {}
    for split in SPLITS:
        vids = sorted(int(p.stem) for p in (bdda / split / "camera_videos").glob("*.mp4"))
        keep[split] = []
        for v in vids:
            n = n_frames(bdda, split, v)
            nfr[(split, v)] = n
            reasons = []
            if v in excl.get(split, set()):
                reasons.append("scout_exclude")
            if not (bdda / split / "gazemap_videos" / f"{v}_pure_hm.mp4").exists():
                reasons.append("no_gazemap")
            if n == 0:
                reasons.append("not_extracted_or_unreadable")
            if reasons:
                excluded.append({"split": split, "vid": v, "reason": "+".join(reasons)})
            else:
                keep[split].append(v)

    results = {}
    for split in ("training", "validation"):
        rows = [row(split, v, f) for v in keep[split] for f in range(nfr[(split, v)])]
        name = f"{SPLITS[split]}.csv"
        write(out / name, rows)
        results[name] = {"videos": len(keep[split]), "frames": len(rows)}

    # common test = SCOUT's scored frames, verified against disk
    rows, bad = [], 0
    for r in csv.DictReader(open(a.scout_test)):
        v, f = int(r["vid"]), int(r["gt_frame"])
        if v not in keep["test"] or f >= nfr[("test", v)]:
            bad += 1
            continue
        rows.append(row("test", v, f))
    rows.sort(key=lambda r: (r["vid"], r["frame"]))
    write(out / "test_common.csv", rows)
    results["test_common.csv"] = {"videos": len({r["vid"] for r in rows}), "frames": len(rows),
                                  "dropped_not_on_disk": bad}

    # SalM2's own test set, mapped to our index
    sal = collections.defaultdict(list)
    for l in open(a.salm2_test):
        if l.strip():
            v, f = json.loads(l).split("/")
            sal[int(v)].append(int(f[:-4]))
    rows, bad, offsets = [], 0, collections.Counter()
    for v, xs in sorted(sal.items()):
        off = min(xs)
        offsets[off] += 1
        for x in sorted(xs):
            f = x - off
            if f >= nfr.get(("test", v), 0):
                bad += 1
                continue
            rows.append(row("test", v, f))
    write(out / "test_salm2_own.csv", rows)
    results["test_salm2_own.csv"] = {"videos": len(sal), "frames": len(rows), "dropped_not_on_disk": bad,
                                     "first_index_counts": dict(offsets.most_common())}

    with open(out / "excluded_videos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["split", "vid", "reason"])
        w.writeheader()
        w.writerows(excluded)

    # leakage guard: no test video in train/val
    tv = {int(r["vid"]) for r in csv.DictReader(open(out / "test_common.csv"))}
    leak = sorted(tv & (set(keep["training"]) | set(keep["validation"])))

    summary = {
        "frame_index": "0-based, our ffmpeg -r 29 extraction; usable = min(camera, gaze) frames",
        "gt_crop": "gazemap 1024x768 -> crop=1024:576:0:91 (SCOUT)",
        "salm2_mapping": "ours = salm2_index - first salm2 index of that video",
        "results": results,
        "excluded_by_reason": dict(collections.Counter(e["reason"] for e in excluded)),
        "test_video_leak_into_train_val": leak,
        "inputs": {p: sha256(p) for p in (a.scout_exclude, a.scout_test, a.salm2_test)},
        "outputs": {p.name: sha256(p) for p in sorted(out.glob("*.csv")) if p.name != "scout_test_samples.csv"},
    }
    json.dump(summary, open(out / "manifest_summary.json", "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("results", "excluded_by_reason", "test_video_leak_into_train_val")},
                     indent=2))


if __name__ == "__main__":
    main()
