#!/usr/bin/env python3
"""
Predict a gazehead test manifest with a trained SCOUT (w/o task) run and write gazehead-format
predictions.  Dataset-generic successor of predict_scout_cocatt.py (kept as the CoCAtt record).

Intended path: /opt/gazehead/tools/predict_scout.py
Run from the SCOUT venv, e.g.
    cd /opt/SCOUT && source .venv/bin/activate
    python /opt/gazehead/tools/predict_scout.py --run /srv/results/scout/bdda_wo_task_<date> --dataset bdda --gpu 0

Writes <run>/pred_<test manifest>/<vid>.npz with `frames` (int64) and `sal` (N,h,w) float32 — the
native decoder output before SCOUT's own blur/resize — plus run.json.  Refuses a non-empty output dir.
"""
import os, sys, json, hashlib, argparse, time
import numpy as np
import torch
import yaml

SCOUT = '/opt/SCOUT'
sys.path.insert(0, SCOUT)
os.chdir(SCOUT)

import model as scout_model                              # noqa: E402
from dataloader_gazehead import GazeheadDataset, DATASETS  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--out', default=None, help='default <run>/pred_<test manifest>')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    cfgd = DATASETS[args.dataset]
    out = args.out or os.path.join(args.run, f"pred_{cfgd['manifests']['test']}")
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f'refusing to write into non-empty {out}')
    os.makedirs(out, exist_ok=True)

    cfg = yaml.safe_load(open(os.path.join(args.run, 'config.yaml')))
    weights = os.path.join(args.run, cfg['best_weights'])
    print('weights', weights, sha256(weights)[:16])

    mp = dict(cfg['model_params'])
    mp['pretrained_backbone'] = False
    net = getattr(scout_model, cfg['model_class'])(**mp)
    sd = torch.load(weights, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    dev = torch.device(f'cuda:{args.gpu}')
    net.to(dev).eval()
    n_params = sum(p.numel() for p in net.parameters())

    ds = GazeheadDataset(mp['clip_size'], img_size=mp['img_size'], mode='test', dataset=args.dataset)
    ds.setup()
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    per_vid = {i: ([], []) for i in range(len(ds.vid_names))}
    t0 = time.time(); n = 0
    with torch.no_grad():
        for sample in loader:
            x = sample[0].to(dev).permute(0, 2, 1, 3, 4)
            pred = net(x, None)
            if pred.dim() == 4:
                pred = pred[:, 0]
            pred = pred.float().cpu().numpy()
            for v, f, p in zip(sample[4][0].tolist(), sample[4][1].tolist(), pred):
                per_vid[v][0].append(f); per_vid[v][1].append(p)
            n += len(pred)
            if n % (args.batch * 250) == 0:
                print(f'{n}/{len(ds)} {n/(time.time()-t0):.1f} fr/s', flush=True)

    counts, shape = {}, None
    h = hashlib.sha256()
    for i, name in enumerate(ds.vid_names):
        fr, maps = per_vid[i]
        order = np.argsort(fr)
        fr = np.asarray(fr, dtype=np.int64)[order]
        sal = np.stack(maps)[order].astype(np.float32)
        assert sal.min() >= 0
        p = os.path.join(out, f'{name}.npz')
        np.savez(p, frames=fr, sal=sal)
        h.update(open(p, 'rb').read())
        counts[name] = int(len(fr)); shape = list(sal.shape[1:])

    run = {
        'model': 'SCOUT w/o task (frames only)', 'dataset': args.dataset, 'run_dir': args.run,
        'weights': weights, 'weights_sha256': sha256(weights), 'params': n_params, 'config': cfg,
        'manifest': ds.manifest, 'manifest_sha256': sha256(ds.manifest),
        'clip_size': mp['clip_size'], 'clip_fps': cfgd['fps'], 'clip_seconds': mp['clip_size'] / cfgd['fps'],
        'clip_padding': 'repeat frame 0 at video start', 'input_size': mp['img_size'], 'output_size': shape,
        'frames_per_vid': counts, 'total_frames': int(sum(counts.values())),
        'note': 'native decoder output, before SCOUT blur/resize; scorer applies protocol blur',
        'pred_sha256': h.hexdigest(), 'seconds': time.time() - t0,
    }
    json.dump(run, open(os.path.join(out, 'run.json'), 'w'), indent=2)
    print(json.dumps({k: run[k] for k in ('total_frames', 'output_size')}), f'videos {len(counts)}')


if __name__ == '__main__':
    main()
