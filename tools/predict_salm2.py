#!/usr/bin/env python3
"""
Predict a gazehead test list with a trained SalM² checkpoint and write gazehead-format predictions.
Dataset-generic successor of predict_salm2_cocatt.py (kept as the CoCAtt record).

Intended path: /opt/gazehead/tools/predict_salm2.py
Run from the SalM² venv, e.g.
    cd /opt/SaliencyMamba && source .venv/bin/activate
    python /opt/gazehead/tools/predict_salm2.py --ckpt /srv/results/salm2/bdda_<date>/model_best_salmm.tar \
        --json-root /srv/datasets/BDDA/salm2 --out /srv/results/salm2/bdda_<date>/pred_bdda_test_scored --gpu 1

Output per video: <out>/<vid>.npz with `frames` (int64) and `sal` (N,256,256) float32 — the raw
sigmoid map at the model's native size, no normalisation, no blur.  Refuses a non-empty --out.
"""
import os, sys, json, hashlib, argparse, time
import numpy as np
import torch
import cv2
from pathlib import Path

SALM = '/opt/SaliencyMamba'
sys.path.insert(0, SALM)
os.chdir(SALM)
from salmm_predict import build_salmm, to_tensor, CLIP_PREFIX  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--json-root', required=True, help='dir with test.json + data_root.txt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f'refusing to write into non-empty {args.out}')
    os.makedirs(args.out, exist_ok=True)

    data_root = open(os.path.join(args.json_root, 'data_root.txt')).read().strip()
    test_json = os.path.join(args.json_root, 'test.json')
    by_vid = {}
    for l in open(test_json):
        r = json.loads(l); by_vid.setdefault(r['vid'], []).append(r)
    vids = sorted(by_vid, key=lambda v: (0, int(v)) if v.isdigit() else (1, v))

    dev = torch.device(f'cuda:{args.gpu}')
    model, ckpt = build_salmm(Path(args.ckpt), dev, None)
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(CLIP_PREFIX))
    print(f'ckpt {args.ckpt} epoch {ckpt.get("epoch")} val_loss {ckpt.get("valid_loss")} '
          f'params {n_params} sha {sha256(args.ckpt)[:16]}; {len(vids)} videos, '
          f'{sum(len(v) for v in by_vid.values())} frames')

    counts = {}
    h = hashlib.sha256()
    t0 = time.time(); n = 0
    with torch.inference_mode():
        for vi, vid in enumerate(vids):
            rows = sorted(by_vid[vid], key=lambda r: r['frame'])
            frames, maps = [], []
            for i in range(0, len(rows), args.batch):
                chunk = rows[i:i + args.batch]
                imgs = [cv2.cvtColor(cv2.imread(os.path.join(data_root, r['cam']), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                        for r in chunk]
                x = torch.stack([to_tensor(im) for im in imgs]).to(dev)
                maps.append(model(x)[0][:, 0].float().cpu().numpy())
                frames.extend(r['frame'] for r in chunk); n += len(chunk)
            sal = np.concatenate(maps).astype(np.float32)
            assert sal.min() >= 0
            p = os.path.join(args.out, f'{vid}.npz')
            np.savez(p, frames=np.asarray(frames, dtype=np.int64), sal=sal)
            h.update(open(p, 'rb').read())
            counts[vid] = len(frames)
            if (vi + 1) % 20 == 0 or vi + 1 == len(vids):
                print(f'{vi+1}/{len(vids)} videos, {n} frames, {n/(time.time()-t0):.1f} fr/s', flush=True)

    run = {
        'model': 'SalM2 (SaliencyMamba, AAAI 2025), frames only; frozen CLIP ViT-B/32 image branch',
        'ckpt': args.ckpt, 'ckpt_sha256': sha256(args.ckpt), 'epoch': ckpt.get('epoch'),
        'valid_loss': float(ckpt.get('valid_loss', float('nan'))), 'trainable_params': n_params,
        'test_json': test_json, 'test_json_sha256': sha256(test_json),
        'build': json.load(open(os.path.join(args.json_root, 'build.json'))) if os.path.exists(os.path.join(args.json_root, 'build.json')) else None,
        'input_size': [256, 256], 'output_size': [256, 256],
        'note': 'raw sigmoid map at native 256x256 (frame squashed to square); no blur/normalisation',
        'frames_per_vid': counts, 'total_frames': int(sum(counts.values())),
        'pred_sha256': h.hexdigest(), 'seconds': time.time() - t0,
    }
    json.dump(run, open(os.path.join(args.out, 'run.json'), 'w'), indent=2)
    print('total', run['total_frames'])


if __name__ == '__main__':
    main()
