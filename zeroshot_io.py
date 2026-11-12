"""Shared helpers for the zero-shot baseline predictors (ML-Net, U²-Net).

Intended path: /opt/gazehead/zeroshot_io.py

Provides: manifest reading, per-video npz writing in the gazehead prediction contract
(<run>/pred_test_common/<vid>.npz with `frames` + one float32 (N,h,w) array), overlay PNGs for
eyeballing, and run.json provenance (weight/manifest hashes, git commits, env versions).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def env_versions():
    import torch

    d = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
    }
    for m in ("cv2", "skimage", "h5py", "torchvision"):
        try:
            d[m] = __import__(m).__version__
        except Exception:
            d[m] = None
    if torch.cuda.is_available():
        d["gpus_visible"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        d["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return d


def read_manifest(path):
    """Rows of a gazehead split manifest (split,vid,frame,cam,gt), sorted by (vid, frame)."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({"vid": row["vid"], "frame": int(row["frame"]), "cam": row["cam"]})
    rows.sort(key=lambda r: (r["vid"], r["frame"]))
    return rows


def pick_check_rows(rows, n):
    """n rows spread evenly through the manifest (different videos, different times)."""
    if n >= len(rows):
        return list(rows)
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [rows[i] for i in idx]


class VideoWriter:
    """Accumulates per-frame maps and writes one <vid>.npz per video.

    Rows must be fed in (vid, frame) order; call flush() after every batch to write the videos that
    are complete (everything except the last vid seen), and flush(keep_last=False) at the end.
    """

    def __init__(self, out_dir, key="sal", compress=True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.key = key
        self.compress = compress
        self.buf = OrderedDict()
        self.written = 0
        self.frames_written = 0

    def add(self, vid, frame, arr):
        if vid not in self.buf:
            self.buf[vid] = ([], [])
        self.buf[vid][0].append(int(frame))
        self.buf[vid][1].append(np.asarray(arr, dtype=np.float32))

    def flush(self, keep_last=True):
        vids = list(self.buf)
        if keep_last and vids:
            vids = vids[:-1]
        for v in vids:
            frames, maps = self.buf.pop(v)
            self._write(v, frames, maps)

    def _write(self, vid, frames, maps):
        frames = np.asarray(frames, dtype=np.int64)
        order = np.argsort(frames, kind="stable")
        frames = frames[order]
        stack = np.stack(maps, 0)[order]
        assert stack.ndim == 3 and stack.dtype == np.float32, (vid, stack.shape, stack.dtype)
        assert np.isfinite(stack).all() and (stack >= 0).all(), f"{vid}: non-finite or negative values"
        assert len(np.unique(frames)) == len(frames), f"{vid}: duplicate frames"
        saver = np.savez_compressed if self.compress else np.savez
        tmp = self.out_dir / f"{vid}.npz.tmp"
        with open(tmp, "wb") as f:
            saver(f, frames=frames, **{self.key: stack})
        os.replace(tmp, self.out_dir / f"{vid}.npz")
        self.written += 1
        self.frames_written += len(frames)


def overlay_png(frame_bgr, sal, path, scale=0.5):
    """frame | heat overlay | raw map, side by side, written as one PNG."""
    import cv2

    h, w = frame_bgr.shape[:2]
    s = np.asarray(sal, np.float32)
    s = s / (s.max() + 1e-12)
    s = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    s8 = (s * 255).astype(np.uint8)
    heat = cv2.applyColorMap(s8, cv2.COLORMAP_JET)
    blend = cv2.addWeighted(frame_bgr, 0.5, heat, 0.5, 0)
    raw = cv2.cvtColor(s8, cv2.COLOR_GRAY2BGR)
    trip = np.concatenate([frame_bgr, blend, raw], 1)
    if scale != 1:
        trip = cv2.resize(trip, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), trip)


def write_run_json(out_dir, info):
    info = dict(info)
    info.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    info.setdefault("argv", sys.argv)
    info.setdefault("env", env_versions())
    p = Path(out_dir) / "run.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return p
