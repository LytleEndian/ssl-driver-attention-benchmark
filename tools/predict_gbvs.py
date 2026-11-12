#!/usr/bin/env python3
"""
/opt/gazehead/tools/predict_gbvs.py

GBVS (Python port of Harel et al. 2006; shreelock/gbvs @ 4c7e3c5, sped up in gbvs_fast.py)
for any manifest with columns split,vid,frame,cam[,gt]. CPU only, one process per video.

Writes <out>/pred_<manifest stem>/<vid>.npz with
    frames : int64 (N,)
    sal    : float32 (N, 28, 32)   min-max normalised GBVS map (evaluate.py resizes to GT size)
plus <out>/run.json and <out>/per_video.csv.

Timing test (1 video, 1 worker):
  python /opt/gazehead/tools/predict_gbvs.py --manifest /opt/gazehead/meta/splits/test_common.csv \
      --root /srv/datasets/BDDA --out /srv/results/gbvs/timing_$(hostname -s) --workers 1 --limit-videos 1

Full run:
  python /opt/gazehead/tools/predict_gbvs.py --manifest ... --root ... --out /srv/results/gbvs/bdda_v1 --workers 22
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # one BLAS thread per worker, or N workers oversubscribe the box

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import scipy

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gbvs_fast  # noqa: E402

PORT_COMMIT = "4c7e3c58e786d053ec4ae6fb9d03bdfeea2c539a"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def port_commit(port_dir):
    try:
        return subprocess.check_output(["git", "-C", str(port_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"



def _init(mat):
    gbvs_fast.set_graph_mat(mat)


def run_video(vid, frames, paths, root, outdir):
    t0 = time.perf_counter()
    sal = np.empty((len(frames), gbvs_fast.GRAPH_H, gbvs_fast.GRAPH_W), np.float32)
    shapes = set()
    for i, p in enumerate(paths):
        img = cv2.imread(str(Path(root) / p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {p}")
        shapes.add(f"{img.shape[1]}x{img.shape[0]}")
        sal[i] = np.maximum(gbvs_fast.compute(img), 0.0)   # min-max round-off gives ~-1e-10 at the minimum
    if not np.isfinite(sal).all() or (sal < 0).any():
        raise RuntimeError(f"{vid}: non-finite or negative values")
    tmp = outdir / f".{vid}.tmp.npz"
    np.savez(tmp, frames=np.asarray(frames, np.int64), sal=sal)
    os.replace(tmp, outdir / f"{vid}.npz")
    return vid, len(frames), time.perf_counter() - t0, sorted(shapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--port-dir", default="/opt/gbvs")
    ap.add_argument("--limit-videos", type=int, default=0, help="timing test: only the first N videos")
    ap.add_argument("--resume", action="store_true", help="allow existing --out; skip finished videos")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.resume:
        sys.exit(f"{out} exists; use a new --out (or --resume to continue an interrupted run)")
    manifest = Path(a.manifest)
    preddir = out / f"pred_{manifest.stem}"
    preddir.mkdir(parents=True, exist_ok=True)
    mat = Path(a.port_dir) / "saliency_models/resources/28__32__m__2.mat"

    vids = OrderedDict()
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            vids.setdefault(r["vid"], []).append((int(r["frame"]), r["cam"]))
    if a.limit_videos:
        vids = OrderedDict(list(vids.items())[: a.limit_videos])
    todo = [v for v in vids if not (preddir / f"{v}.npz").exists()]
    n_frames = sum(len(vids[v]) for v in todo)

    meta = {
        "model": "GBVS (Python port, Harel/Koch/Perona 2006)",
        "port": {"repo": "https://github.com/shreelock/gbvs", "expected_commit": PORT_COMMIT,
                 "found_commit": port_commit(a.port_dir), "graph_mat": str(mat), "graph_mat_sha256": sha256(mat)},
        "code_sha256": {"gbvs_fast.py": sha256(gbvs_fast.__file__), "predict_gbvs.py": sha256(__file__)},
        "args": vars(a),
        "manifest_sha256": sha256(manifest),
        "videos_total": len(vids), "videos_todo": len(todo), "frames_todo": n_frames,
        "output": {"key": "sal", "shape_per_frame": [gbvs_fast.GRAPH_H, gbvs_fast.GRAPH_W],
                   "note": "port's min-max normalised master map before its final cubic upsampling; ~1e-10 round-off negatives clipped to 0"},
        "input": "native frame, BGR uint8 via cv2.imread, no resizing (port defaults, untuned)",
        "params": {k: v for k, v in gbvs_fast.PARAMS.items() if k != "gaborparams"} | {"gaborparams": {
            k: (float(v) if isinstance(v, float) else v) for k, v in gbvs_fast.PARAMS["gaborparams"].items()}},
        "env": {"host": socket.gethostname(), "cpu": cpu_model(), "cpu_count": os.cpu_count(),
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "opencv": cv2.__version__},
    }
    if meta["port"]["found_commit"] not in (PORT_COMMIT, "unknown"):
        print(f"WARNING: {a.port_dir} is at {meta['port']['found_commit']}, expected {PORT_COMMIT}", flush=True)
    print(f"{len(todo)} videos / {n_frames} frames to do, {a.workers} workers, host {meta['env']['host']}", flush=True)
    if not todo:
        print("nothing to do")
        return

    t0 = time.perf_counter()
    done_f = 0
    shapes = set()
    log = open(out / "per_video.csv", "a", newline="")
    lw = csv.writer(log)
    if log.tell() == 0:
        lw.writerow(["vid", "frames", "seconds", "s_per_frame"])
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(mat,)) as ex:
        futs = [ex.submit(run_video, v, [f for f, _ in vids[v]], [p for _, p in vids[v]], a.root, preddir)
                for v in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            vid, nf, secs, shp = fut.result()
            shapes.update(shp)
            done_f += nf
            lw.writerow([vid, nf, f"{secs:.2f}", f"{secs / nf:.4f}"]); log.flush()
            el = time.perf_counter() - t0
            eta = el / done_f * (n_frames - done_f)
            print(f"[{k}/{len(todo)}] {vid}: {nf} fr {secs / nf:.3f} s/fr/worker | "
                  f"{done_f / el:.1f} fr/s total | ETA {eta / 60:.1f} min", flush=True)
    log.close()

    el = time.perf_counter() - t0
    meta["frame_sizes_seen"] = sorted(shapes)
    meta["timing"] = {"wall_s": round(el, 1), "frames_per_s": round(done_f / max(el, 1e-9), 3),
                      "s_per_frame_per_worker": round(el * a.workers / max(done_f, 1), 4)}
    full = 84831
    meta["timing"]["est_hours_for_84831_frames_at_these_workers"] = round(full / max(done_f / el, 1e-9) / 3600, 2)
    with open(out / ("run.json" if not (out / "run.json").exists() else f"run_{int(time.time())}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["timing"], indent=2))
    print(f"wrote {preddir}")


if __name__ == "__main__":
    main()
