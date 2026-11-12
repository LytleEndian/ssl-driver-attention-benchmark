#!/usr/bin/env python3
"""
Predict CoCAtt test with a trained SalM² checkpoint and write gazehead-format predictions.

Intended path: /opt/gazehead/tools/predict_salm2_cocatt.py
Run from the SalM² venv:
    cd /opt/SaliencyMamba && source .venv/bin/activate
    python /opt/gazehead/tools/predict_salm2_cocatt.py --ckpt <ckpt>/model_best_salmm.tar \
        --out /srv/results/salm2/cocatt_<date>/pred_cocatt_test --gpu 1

Reuses build_salmm() from /opt/SaliencyMamba/salmm_predict.py (fused Mamba kernels on CUDA).
Output per video: <out>/<vid>.npz with `frames` (int64) and `sal` (N,256,256) float32 — the raw
sigmoid map at the model's native 256x256, no normalisation, no blur.  The scorer resizes it to
the 256x512 GT (restoring the 2:1 aspect) and sum-normalises.  Refuses a non-empty --out.
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

ROOT_JSON = '/srv/datasets/CoCAtt/salm2'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out):
        sys.exit(f'refusing to write into non-empty {args.out}')
    os.makedirs(args.out, exist_ok=True)

    data_root = open(os.path.join(ROOT_JSON, 'data_root.txt')).read().strip()
    test_json = os.path.join(ROOT_JSON, 'test.json')
    recs = [json.loads(l) for l in open(test_json)]
    by_vid = {}
    for r in recs:
        by_vid.setdefault(r['vid'], []).append(r)

    dev = torch.device(f'cuda:{args.gpu}')
    model, ckpt = build_salmm(Path(args.ckpt), dev, None)
    n_params = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(CLIP_PREFIX))
    print(f'ckpt {args.ckpt} epoch {ckpt.get("epoch")} val_loss {ckpt.get("valid_loss")} '
          f'params {n_params} sha {sha256(args.ckpt)[:16]}')

    counts = {}
    t0 = time.time(); n = 0
    with torch.inference_mode():
        for vid in sorted(by_vid):
            rows = sorted(by_vid[vid], key=lambda r: r['frame'])
            frames, maps = [], []
            for i in range(0, len(rows), args.batch):
                chunk = rows[i:i + args.batch]
                imgs = [cv2.cvtColor(cv2.imread(os.path.join(data_root, r['cam']), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                        for r in chunk]
                x = torch.stack([to_tensor(im) for im in imgs]).to(dev)
                sal = model(x)[0][:, 0].float().cpu().numpy()
                maps.append(sal); frames.extend(r['frame'] for r in chunk)
                n += len(chunk)
            sal = np.concatenate(maps).astype(np.float32)
            assert sal.min() >= 0
            np.savez(os.path.join(args.out, f'{vid}.npz'), frames=np.asarray(frames, dtype=np.int64), sal=sal)
            counts[vid] = len(frames)
            print(f'{vid}: {len(frames)} frames, {n/(time.time()-t0):.1f} fr/s', flush=True)

    h = hashlib.sha256()
    for vid in sorted(by_vid):
        h.update(open(os.path.join(args.out, f'{vid}.npz'), 'rb').read())
    run = {
        'model': 'SalM2 (SaliencyMamba, AAAI 2025), frames only; frozen CLIP ViT-B/32 image branch',
        'ckpt': args.ckpt, 'ckpt_sha256': sha256(args.ckpt), 'epoch': ckpt.get('epoch'),
        'valid_loss': float(ckpt.get('valid_loss', float('nan'))), 'trainable_params': n_params,
        'test_json': test_json, 'test_json_sha256': sha256(test_json),
        'input_size': [256, 256], 'output_size': [256, 256],
        'note': 'raw sigmoid map at native 256x256 (2:1 frame squashed); no blur/normalisation',
        'frames_per_vid': counts, 'total_frames': int(sum(counts.values())),
        'pred_sha256': h.hexdigest(), 'seconds': time.time() - t0,
    }
    json.dump(run, open(os.path.join(args.out, 'run.json'), 'w'), indent=2)
    print('total', run['total_frames'], 'expected 15029')


if __name__ == '__main__':
    main()
