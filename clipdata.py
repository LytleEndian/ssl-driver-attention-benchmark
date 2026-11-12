"""
clipdata.py - clip and cached-feature datasets for video backbones (V-JEPA 2.1).

Samples are EXACTLY the manifest rows ManifestFrames / CachedFrames would use (same stride rule:
frame % stride == 0), so DINOv3 and V-JEPA train/validate/predict on identical GT frames.

CachedClips      clip of n_frames ending at the GT frame, read from the frame cache
                 (tools/build_frame_cache.py): x uint8 T x 3 x H x W -> finish with prep_clip().
CachedFeatures   precomputed frozen-backbone features (tools/extract_vjepa_features.py):
                 f float16 L x D x h x w + the same GT as the frame cache.
"""
from __future__ import annotations

import collections
import csv
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from gazehead import IMNET_MEAN, IMNET_STD
from vjepa_backbone import clip_frame_indices
from data import vid_key

_MEAN = _STD = None


def prep_clip(x, dev):
    """uint8 B x T x 3 x H x W -> float B x 3 x T x H x W, same arithmetic as train_gaze.prep()."""
    global _MEAN, _STD
    x = x.to(dev, non_blocking=True)
    if _MEAN is None or _MEAN.device != x.device:
        _MEAN = torch.tensor(IMNET_MEAN, device=x.device).view(1, 1, 3, 1, 1)
        _STD = torch.tensor(IMNET_STD, device=x.device).view(1, 1, 3, 1, 1)
    x = (x.float() / 255.0 - _MEAN) / _STD
    return x.permute(0, 2, 1, 3, 4).contiguous()


def manifest_rows(manifest, stride=1):
    rows = list(csv.DictReader(open(manifest)))
    if stride > 1:
        rows = [r for r in rows if int(r["frame"]) % stride == 0]
    return rows


class CachedClips(Dataset):
    def __init__(self, manifest, cache_dir, n_frames: int, step: int, stride: int = 1, with_gt: bool = True):
        by_video = collections.defaultdict(dict)          # (split, vid) -> {frame: cache row}
        for r in csv.DictReader(open(f"{cache_dir}/index.csv")):
            by_video[(r["split"], vid_key(r["vid"]))][int(r["frame"])] = int(r["row"])
        avail = {k: sorted(v) for k, v in by_video.items()}

        self.items = []
        self.n_padded = 0
        for r in manifest_rows(manifest, stride):
            key, fr = (r["split"], vid_key(r["vid"])), int(r["frame"])
            if key not in by_video or fr not in by_video[key]:
                raise ValueError(f"{key} frame {fr} not in cache {cache_dir}")
            idx = clip_frame_indices(fr, n_frames, step, avail[key])
            if fr - (n_frames - 1) * step < avail[key][0]:
                self.n_padded += 1
            self.items.append((key[1], fr, [by_video[key][f] for f in idx], by_video[key][fr]))
        self.cache_dir, self.with_gt = cache_dir, with_gt
        self.n_frames, self.step = n_frames, step
        self.X = self.Y = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.X is None:
            self.X = np.load(f"{self.cache_dir}/frames.npy", mmap_mode="r")
            self.Y = np.load(f"{self.cache_dir}/gt.npy", mmap_mode="r")
        v, f, rows, gt_row = self.items[i]
        item = {"x": torch.from_numpy(np.stack([self.X[j] for j in rows])), "vid": v, "frame": f}
        if self.with_gt:
            item["gt"] = torch.from_numpy(self.Y[gt_row].astype(np.float32) / 255.0)
        return item


class CachedFeatures(Dataset):
    def __init__(self, feat_dir, with_gt: bool = True):
        self.dir = str(feat_dir)
        self.meta = json.load(open(f"{self.dir}/meta.json"))
        if not self.meta.get("complete"):
            raise ValueError(f"{self.dir}: feature extraction not complete")
        self.rows = [(vid_key(r["vid"]), int(r["frame"])) for r in csv.DictReader(open(f"{self.dir}/index.csv"))]
        self.with_gt = with_gt
        self.F = self.Y = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.F is None:
            self.F = np.load(f"{self.dir}/feats.npy", mmap_mode="r")
            self.Y = np.load(f"{self.dir}/gt.npy", mmap_mode="r")
        v, f = self.rows[i]
        item = {"f": torch.from_numpy(np.array(self.F[i])), "vid": v, "frame": f}
        if self.with_gt:
            item["gt"] = torch.from_numpy(self.Y[i].astype(np.float32) / 255.0)
        return item
