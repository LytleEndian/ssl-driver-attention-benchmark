"""
data.py - frame dataset driven by the frozen split manifests (meta/splits/*.csv).

Each item: x (3 x H x W, ImageNet-normalized), gt (h x w in [0,1], optional), vid, frame.
GT is the SCOUT-cropped 1024x576 gaze-map PNG, converted to L and downsampled with
PIL BILINEAR (PIL antialiases on downscale) to target_hw. Normalization to a
probability distribution happens in the loss / metric, not here.
"""
from __future__ import annotations

import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gazehead import make_transform


def vid_key(s):
    """Video id from a manifest/index cell. BDD-A ids are ints ('123') and stay ints, so those
    batches collate to a tensor exactly as before; CoCAtt ids are strings ('A_0') and stay strings
    (they collate to a list, which train_gaze.predict handles)."""
    s = str(s)
    return int(s) if s.isdigit() else s


class CachedFrames(Dataset):
    """Same items as ManifestFrames, read from tools/build_frame_cache.py output.
    x is uint8 3 x H x W (resized, not yet scaled/normalized) -> finish on GPU with
    train_gaze.prep(). gt is float32 in [0,1], identical to ManifestFrames."""

    def __init__(self, manifest: str, cache_dir: str, stride: int = 1, with_gt: bool = True):
        rows = list(csv.DictReader(open(manifest)))
        if stride > 1:
            rows = [r for r in rows if int(r["frame"]) % stride == 0]
        idx = {r["cam"]: int(r["row"]) for r in csv.DictReader(open(f"{cache_dir}/index.csv"))}
        missing = [r["cam"] for r in rows if r["cam"] not in idx]
        if missing:
            raise ValueError(f"{len(missing)} frames not in cache, e.g. {missing[:3]}")
        self.rows = [(vid_key(r["vid"]), int(r["frame"]), idx[r["cam"]]) for r in rows]
        self.cache_dir, self.with_gt = cache_dir, with_gt
        self.X = self.Y = None  # opened lazily inside each worker

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.X is None:
            self.X = np.load(f"{self.cache_dir}/frames.npy", mmap_mode="r")
            self.Y = np.load(f"{self.cache_dir}/gt.npy", mmap_mode="r")
        v, f, j = self.rows[i]
        item = {"x": torch.from_numpy(np.array(self.X[j])), "vid": v, "frame": f}
        if self.with_gt:
            item["gt"] = torch.from_numpy(self.Y[j].astype(np.float32) / 255.0)
        return item


class ManifestFrames(Dataset):
    def __init__(self, manifest: str, root: str, input_hw=(288, 512), target_hw=(72, 128),
                 stride: int = 1, with_gt: bool = True):
        rows = list(csv.DictReader(open(manifest)))
        if stride > 1:  # temporal subsampling inside each video (stride 10 ~ 2.9 Hz)
            rows = [r for r in rows if int(r["frame"]) % stride == 0]
        self.rows, self.root = rows, root.rstrip("/")
        self.tf = make_transform(tuple(input_hw))
        self.target_hw = tuple(target_hw)
        self.with_gt = with_gt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(f"{self.root}/{r['cam']}").convert("RGB")
        item = {"x": self.tf(img), "vid": vid_key(r["vid"]), "frame": int(r["frame"])}
        if self.with_gt:
            g = Image.open(f"{self.root}/{r['gt']}").convert("L")
            g = g.resize((self.target_hw[1], self.target_hw[0]), Image.BILINEAR)
            item["gt"] = torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)
        return item
