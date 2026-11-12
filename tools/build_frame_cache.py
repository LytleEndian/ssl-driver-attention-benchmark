#!/usr/bin/env python
"""
build_frame_cache.py - one-time cache of pre-resized frames and GT for fast training.

Stores exactly the intermediate the on-the-fly pipeline produces, so training on the
cache is numerically equivalent to training on JPEGs:
  frames: PIL RGB -> v2.ToImage() -> v2.Resize((H, W), antialias=True)  -> uint8 [N,3,H,W]
          (the same ops, in the same order, as gazehead.make_transform before ToDtype/Normalize)
  gt:     PIL L -> resize((w, h), BILINEAR)                             -> uint8 [N,h,w]
          (the same op as data.ManifestFrames)
ToDtype/Normalize then run on the GPU in train_gaze.py.

  python tools/build_frame_cache.py --manifests train,val,test_salm2_own \
      --out /srv/cache/bdda_288x512 --jobs 12

Writes: frames.npy, gt.npy (np.load(..., mmap_mode="r")), index.csv (row -> cam/gt path),
        meta.json (shapes, transform, versions, manifest hashes).
Rows are the union of the listed manifests (test_salm2_own is a superset of test_common).
"""
import argparse
import csv
import hashlib
import json
import pathlib
import time
from multiprocessing import Pool

import numpy as np

G = {}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def init(out, root, hw, thw):
    import torch
    from torchvision.transforms import v2
    torch.set_num_threads(1)
    G["X"] = np.load(out / "frames.npy", mmap_mode="r+")
    G["Y"] = np.load(out / "gt.npy", mmap_mode="r+")
    G["root"], G["thw"] = root, thw
    G["tf"] = v2.Compose([v2.ToImage(), v2.Resize(hw, antialias=True)])


def work(chunk):
    from PIL import Image
    for j, cam, gt in chunk:
        img = Image.open(f"{G['root']}/{cam}").convert("RGB")
        G["X"][j] = G["tf"](img).numpy()
        if not gt:   # history-only frame (no gaze map, e.g. CoCAtt tracker dropout): zero GT slot, never a target
            G["Y"][j] = 0
            continue
        g = Image.open(f"{G['root']}/{gt}").convert("L").resize((G["thw"][1], G["thw"][0]), Image.BILINEAR)
        G["Y"][j] = np.asarray(g, dtype=np.uint8)
    return len(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/srv/datasets/BDDA")
    ap.add_argument("--splits", default="meta/splits")
    ap.add_argument("--manifests", default="train,val,test_salm2_own")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", default="288x512")
    ap.add_argument("--target", default="72x128")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    hw = tuple(int(v) for v in a.input.split("x"))
    thw = tuple(int(v) for v in a.target.split("x"))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = pathlib.Path(a.splits)

    rows, seen = [], set()
    for name in a.manifests.split(","):
        for r in csv.DictReader(open(splits / f"{name}.csv")):
            if r["cam"] not in seen:
                seen.add(r["cam"])
                rows.append(r)
    n = len(rows)
    need = n * (3 * hw[0] * hw[1] + thw[0] * thw[1]) / 1e9
    print(f"{n} unique frames -> {need:.1f} GB in {out}")

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "split", "vid", "frame", "cam", "gt"])
        for j, r in enumerate(rows):
            w.writerow([j, r["split"], r["vid"], r["frame"], r["cam"], r["gt"]])
    np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8, shape=(n, 3, *hw)).flush()
    np.lib.format.open_memmap(out / "gt.npy", mode="w+", dtype=np.uint8, shape=(n, *thw)).flush()

    items = [(j, r["cam"], r["gt"]) for j, r in enumerate(rows)]
    chunks = [items[i:i + a.chunk] for i in range(0, n, a.chunk)]
    t0, done = time.time(), 0
    with Pool(a.jobs, initializer=init, initargs=(out, a.root, hw, thw)) as p:
        for k in p.imap_unordered(work, chunks):
            done += k
            if done % (a.chunk * 100) < a.chunk or done == n:
                el = time.time() - t0
                print(f"{done}/{n}  {done/el:.0f} fr/s  eta {(n-done)/max(done/el,1e-9)/60:.1f} min", flush=True)

    import torch
    import torchvision
    meta = {
        "n": n, "frames_shape": [n, 3, *hw], "gt_shape": [n, *thw], "dtype": "uint8",
        "frame_transform": f"PIL RGB -> v2.ToImage -> v2.Resize({hw}, antialias=True)",
        "gt_transform": f"PIL L -> resize(({thw[1]},{thw[0]}), BILINEAR)",
        "manifests": {m: sha256(splits / f"{m}.csv") for m in a.manifests.split(",")},
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "build_seconds": round(time.time() - t0),
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print("done", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
