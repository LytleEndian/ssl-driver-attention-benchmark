#!/usr/bin/env python
"""
extract_vjepa_features.py - run the frozen V-JEPA 2.1 encoder ONCE over a manifest and store the
4 readout feature maps of each clip's last tubelet, so the decoder can be trained for 30 epochs
without re-running the backbone. The backbone is frozen, in eval mode and sees no augmentation,
so this is numerically the on-the-fly pipeline (fp16 storage of fp16-autocast activations);
--verify re-computes random samples and reports the difference.

  python tools/extract_vjepa_features.py --ckpt /srv/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
      --cache /srv/cache/bdda_288x512 --manifest train --stride 10 --clip-frames 8 --clip-step 7 \
      --out /srv/cache/vjepa21L_bdda_c8s7/train_s10 --gpu 0

Writes: feats.npy (N x 4 x D x h x w, float16), gt.npy (N x th x tw, uint8, copied from the frame
cache), index.csv (row, vid, frame), meta.json ("complete": true only after a full pass).
"""
import argparse
import csv
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from clipdata import CachedClips, prep_clip  # noqa: E402
from vjepa_backbone import FrozenVJEPA21  # noqa: E402
from data import vid_key


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="vit_large")
    ap.add_argument("--feat-norm", default="none", choices=["none", "model"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--splits", default="meta/splits")
    ap.add_argument("--manifest", required=True, help="manifest name, e.g. train / val")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--clip-frames", type=int, required=True)
    ap.add_argument("--clip-step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", type=int, default=32, help="re-check N random samples after writing")
    ap.add_argument("--finalize", action="store_true",
                    help="features already fully written by an earlier run (no meta.json): skip extraction, "
                         "run only verification + meta.json")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    if (out / "meta.json").exists():
        sys.exit(f"{out} already has meta.json; use a new --out")
    if a.finalize and not (out / "feats.npy").exists():
        sys.exit(f"--finalize: {out}/feats.npy not found")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{a.gpu}")
    man = pathlib.Path(a.splits) / f"{a.manifest}.csv"

    enc = FrozenVJEPA21(a.ckpt, arch=a.arch, feat_norm=a.feat_norm, fp16=True).to(dev)
    if enc.missing_keys or enc.unexpected_keys:
        print("WARNING key mismatch", enc.missing_keys, enc.unexpected_keys)
    ds = CachedClips(man, a.cache, a.clip_frames, a.clip_step, stride=a.stride, with_gt=True)
    cmeta = json.load(open(pathlib.Path(a.cache) / "meta.json"))
    H, W = cmeta["frames_shape"][2:]
    th, tw = cmeta["gt_shape"][1:]
    h, w = H // enc.patch_size, W // enc.patch_size
    L, D, N = len(enc.layers), enc.embed_dim, len(ds)
    print(f"{N} clips ({ds.n_padded} padded) | {a.clip_frames} frames step {a.clip_step} | "
          f"feats {L}x{D}x{h}x{w} fp16 = {N * L * D * h * w * 2 / 1e9:.1f} GB")

    t0, j = time.time(), 0
    if a.finalize:
        feats = np.load(out / "feats.npy", mmap_mode="r")
        gts = np.load(out / "gt.npy", mmap_mode="r")
        if feats.shape != (N, L, D, h, w) or gts.shape != (N, th, tw):
            sys.exit(f"--finalize: shapes {feats.shape} / {gts.shape} do not match this dataset")
        idx_rows = [(vid_key(r["vid"]), int(r["frame"])) for r in csv.DictReader(open(out / "index.csv"))]
        if idx_rows != [(v, fr) for v, fr, _, _ in ds.items]:
            sys.exit("--finalize: index.csv does not match this dataset's sample order")
        if not np.any(feats[N - 1, 0, 0]):
            sys.exit("--finalize: last row is empty; extraction did not finish")
        print("--finalize: skipping extraction")
        j = N
    else:
        feats = np.lib.format.open_memmap(out / "feats.npy", mode="w+", dtype=np.float16, shape=(N, L, D, h, w))
        gts = np.lib.format.open_memmap(out / "gt.npy", mode="w+", dtype=np.uint8, shape=(N, th, tw))
        with open(out / "index.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["row", "vid", "frame"])
            for jj, (v, fr, _, _) in enumerate(ds.items):
                wr.writerow([jj, v, fr])

    dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
    with torch.no_grad():
        for k, b in enumerate([] if a.finalize else dl):
            fs = enc(prep_clip(b["x"], dev))
            n = fs[0].shape[0]
            feats[j:j + n] = torch.stack(fs, 1).half().cpu().numpy()
            gts[j:j + n] = np.rint(b["gt"].numpy() * 255.0).astype(np.uint8)
            j += n
            if k % 200 == 0:
                el = time.time() - t0
                print(f"  {j}/{N}  {j / el:.1f} clips/s  eta {(N - j) / max(j / el, 1e-9) / 60:.0f} min", flush=True)
    if not a.finalize:
        feats.flush()
        gts.flush()
    assert j == N, (j, N)

    # ---- verification: recompute random samples, compare to what was stored -------------------
    rng = np.random.default_rng(0)
    worst_cos, worst_rel, gt_ok = 1.0, 0.0, True
    for i in rng.choice(N, size=min(a.verify, N), replace=False):
        it = ds[int(i)]
        with torch.no_grad():
            f = torch.stack(enc(prep_clip(it["x"][None], dev)), 1)[0].half().float().cpu()
        s = torch.from_numpy(np.array(feats[i])).float()
        cos = torch.nn.functional.cosine_similarity(f.flatten(), s.flatten(), dim=0).item()
        worst_cos = min(worst_cos, cos)
        worst_rel = max(worst_rel, ((f - s).abs().max() / s.abs().max()).item())
        gt_ok &= bool(np.array_equal(np.rint(it["gt"].numpy() * 255).astype(np.uint8), gts[i]))
    print(f"verify {min(a.verify, N)} samples: min cos {worst_cos:.6f}  max rel diff {worst_rel:.2e}  gt identical {gt_ok}")

    meta = {
        "complete": True, "n": N, "feats_shape": [N, L, D, h, w], "dtype": "float16",
        "manifest": a.manifest, "manifest_sha256": sha256(man), "stride": a.stride,
        "clip_frames": a.clip_frames, "clip_step": a.clip_step, "n_padded": ds.n_padded,
        "readout": {"arch": a.arch, "layers": enc.layers, "feat_norm": a.feat_norm, "slice": "last tubelet"},
        "ckpt": a.ckpt, "ckpt_sha256": sha256(a.ckpt),
        "missing_keys": enc.missing_keys, "unexpected_keys": enc.unexpected_keys,
        "vjepa2_commit": subprocess.run(["git", "-C", "/opt/vjepa2/src", "rev-parse", "HEAD"],
                                        capture_output=True, text=True).stdout.strip(),
        "frame_cache": a.cache, "frame_cache_meta": cmeta,
        "code_sha256": {p: sha256(pathlib.Path(__file__).resolve().parents[1] / p)
                        for p in ("vjepa_backbone.py", "clipdata.py", "tools/extract_vjepa_features.py")},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev),
        "verify": {"n": min(a.verify, N), "min_cos": worst_cos, "max_rel_diff": worst_rel, "gt_identical": gt_ok},
        "seconds": round(time.time() - t0),
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"done in {meta['seconds'] / 60:.0f} min -> {out}")


if __name__ == "__main__":
    main()
