#!/usr/bin/env python
"""Zero-shot U²-Net (Qin et al., PR 2020) predictions in the gazehead prediction contract.

Intended path: /opt/gazehead/tools/predict_u2net.py

Writes <out>/pred_test_common/<vid>.npz (frames + float32 `sal` (N,320,320) — the fused output d1 after
the model's own sigmoid, at the network's 320x320 input resolution, BEFORE normPRED) and <out>/run.json.
The scorer's resize to 576x1024 undoes the 16:9 -> square squash.

Preprocessing replicates u2net_test.py exactly: RescaleT(320) = skimage.transform.resize to 320x320
(mode='constant', float in [0,1]); ToTensorLab(flag=0) = image / image.max(), then per-channel
(x - mean) / std with ImageNet mean/std on RGB. The model file is imported from the cloned repo
(--repo, default /opt/u2net) and the import path is asserted.

Check mode (no predictions): --check-frames 12 --check-dir /srv/results/u2net/check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/opt/gazehead")
import zeroshot_io as Z  # noqa: E402

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
N_PARAMS_PAPER = 44.0e6  # U²-Net full, 44.0M in the paper


def import_u2net(repo):
    repo = str(Path(repo).resolve())
    sys.path.insert(0, repo)
    from model.u2net import U2NET  # noqa: E402
    import model.u2net as mod
    assert Path(mod.__file__).resolve().is_relative_to(repo), f"U2NET imported from {mod.__file__}, not {repo}"
    return U2NET, mod.__file__


def preprocess_rgb(img_rgb, size=320):
    from skimage import transform

    im = transform.resize(img_rgb, (size, size), mode="constant")  # float64 in [0,1]
    im = im / max(float(im.max()), 1e-12)                          # ToTensorLab(flag=0): image/np.max(image)
    out = np.empty((3, size, size), np.float32)
    for c in range(3):
        out[c] = (im[:, :, c] - MEAN[c]) / STD[c]
    return out


class Frames(Dataset):
    def __init__(self, rows, root, size):
        self.rows, self.root, self.size = rows, Path(root), size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(self.root / r["cam"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.root / r["cam"])
        return torch.from_numpy(preprocess_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.size)), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="run dir, e.g. /srv/results/u2net/bdda_v1")
    ap.add_argument("--weights", default="/srv/models/u2net/u2net.pth")
    ap.add_argument("--repo", default="/opt/u2net")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=0, help="smoke test: only the first N videos")
    ap.add_argument("--check-frames", type=int, default=0)
    ap.add_argument("--check-dir", default=None)
    args = ap.parse_args()

    U2NET, model_file = import_u2net(args.repo)
    print("U2NET from", model_file, "| zeroshot_io from", Z.__file__)
    dev = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = True

    rows = Z.read_manifest(args.manifest)
    vids = sorted({r["vid"] for r in rows})
    if args.max_videos:
        keep = set(vids[:args.max_videos])
        rows = [r for r in rows if r["vid"] in keep]
        vids = sorted(keep)
    print(f"{len(rows)} frames / {len(vids)} videos from {args.manifest}")

    net = U2NET(3, 1)
    sd = torch.load(args.weights, map_location="cpu")
    missing, unexpected = net.load_state_dict(sd, strict=True)
    npar = sum(p.numel() for p in net.parameters())
    print(f"loaded {len(sd)} tensors from {args.weights}; missing {missing} unexpected {unexpected}; "
          f"params {npar:,} (paper: 44.0M)")
    assert abs(npar - N_PARAMS_PAPER) / N_PARAMS_PAPER < 0.01, npar
    net = net.to(dev).eval()

    def forward(x):
        outs = net(x)                       # 7 sigmoid maps; outs[0] is d1 = the fused output
        return outs[0][:, 0]

    if args.check_frames:
        cdir = Path(args.check_dir or Path(args.out).parent / "check")
        for r in Z.pick_check_rows(rows, args.check_frames):
            img = cv2.imread(str(Path(args.root) / r["cam"]), cv2.IMREAD_COLOR)
            x = torch.from_numpy(preprocess_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), args.size))[None]
            with torch.no_grad():
                sal = forward(x.to(dev))[0].cpu().numpy()
            Z.overlay_png(img, sal, cdir / f"{r['vid']}_{r['frame']:05d}.png")
            print(f"{r['vid']} {r['frame']:5d} saved {sal.shape} min {sal.min():.3f} max {sal.max():.3f} "
                  f"mean {sal.mean():.3f}")
        print(f"overlays in {cdir}")
        return

    out = Path(args.out)
    pred_dir = out / "pred_test_common"
    if pred_dir.exists() and any(pred_dir.iterdir()):
        raise SystemExit(f"{pred_dir} is not empty — use a new run directory")
    writer = Z.VideoWriter(pred_dir, key="sal")
    dl = DataLoader(Frames(rows, args.root, args.size), batch_size=args.bs, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for x, idx in dl:
            sal = forward(x.to(dev, non_blocking=True)).cpu().numpy()
            for s, i in zip(sal, idx.tolist()):
                writer.add(rows[i]["vid"], rows[i]["frame"], s)
            writer.flush()
            done += len(idx)
            if done % (args.bs * 50) < args.bs:
                el = time.time() - t0
                print(f"{done}/{len(rows)} frames, {writer.written} videos written, "
                      f"{done / el:.1f} fps, eta {(len(rows) - done) / (done / el) / 60:.1f} min", flush=True)
    writer.flush(keep_last=False)
    el = time.time() - t0
    print(f"done: {writer.frames_written} frames / {writer.written} videos in {el / 60:.1f} min")

    Z.write_run_json(out, {
        "model": "U²-Net full (Qin et al., PR 2020), zero-shot, DUTS-TR salient-object weights (u2net.pth)",
        "code": {"repo": args.repo, "commit": Z.git_commit(args.repo), "model_file": model_file,
                 "gazehead_commit": Z.git_commit("/opt/gazehead")},
        "weights": {"path": args.weights, "sha256": Z.sha256_file(args.weights), "tensors": len(sd), "n_params": npar},
        "manifest": {"path": args.manifest, "sha256": Z.sha256_file(args.manifest),
                     "frames": len(rows), "videos": len(vids)},
        "decisions": {
            "input": f"squash to {args.size}x{args.size} via skimage.transform.resize(mode='constant'), "
                     f"RGB, image/image.max(), ImageNet mean/std (u2net_test.py RescaleT + ToTensorLab(flag=0))",
            "output": f"d1 (fused head) after the model's sigmoid, {args.size}x{args.size}, before normPRED; "
                      f"no blur, no normalisation; evaluate.py resizes to 576x1024 (restores 16:9) + blurs",
            "head": "outs[0] of U2NET.forward", "precision": "fp32",
            "label": "salient-OBJECT segmentation baseline (objectness vs gaze), not an attention model",
        },
        "timing": {"seconds": el, "fps": len(rows) / el, "batch_size": args.bs, "gpu_index": args.gpu},
        "outputs": {"pred_dir": str(pred_dir), "npz_key": "sal", "videos_written": writer.written,
                    "frames_written": writer.frames_written, "npz": "savez_compressed"},
    })
    print(f"run.json written to {out / 'run.json'}")


if __name__ == "__main__":
    main()
