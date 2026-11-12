#!/usr/bin/env python3
"""
Export gazehead benchmark manifests as SalM²-style JSON-lines lists (dataset-generic successor of
build_salm2_cocatt_json.py, which stays as the CoCAtt record).

Intended path: /opt/gazehead/tools/build_salm2_json.py
Example:
    python /opt/gazehead/tools/build_salm2_json.py --dataset bdda --train-stride 10 --val-stride 5
Output: /srv/datasets/<Dataset>/salm2/{train,valid,test}.json + data_root.txt + build.json + SHA256SUMS
(pass the dir as SalM² --root).  Rows carry vid, frame, cam, gt only.  Strides subsample each video's
sorted frames from offset 0 (same rule as the SSL recipe's --stride); test is never subsampled.
"""
import csv, json, os, hashlib, sys, argparse

SPLITS = '/opt/gazehead/meta/splits'
DATASETS = {
    'bdda':   dict(root='/srv/datasets/BDDA',   manifests={'train': 'bdda_train', 'val': 'bdda_val', 'test': 'bdda_test_scored'}),
    'cocatt': dict(root='/srv/datasets/CoCAtt', manifests={'train': 'cocatt_train', 'val': 'cocatt_val', 'test': 'cocatt_test'}),
}
NAMES = {'train': 'train', 'val': 'valid', 'test': 'test'}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
ap.add_argument('--train-stride', type=int, default=1)
ap.add_argument('--val-stride', type=int, default=1)
ap.add_argument('--out', default=None, help='default <root>/salm2')
args = ap.parse_args()
cfg = DATASETS[args.dataset]
out = args.out or os.path.join(cfg['root'], 'salm2')
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'data_root.txt'), 'w') as f:
    f.write(cfg['root'] + '\n')

strides = {'train': args.train_stride, 'val': args.val_stride, 'test': 1}
info = {'dataset': args.dataset, 'strides': strides, 'manifests': {}, 'rows': {}}
sums = []
for split, name in NAMES.items():
    src = os.path.join(SPLITS, cfg['manifests'][split] + '.csv')
    dst = os.path.join(out, f'{name}.json')
    by_vid = {}
    for r in csv.DictReader(open(src)):
        by_vid.setdefault(r['vid'], []).append(r)
    n = missing = 0
    with open(dst, 'w') as fo:
        for vid in sorted(by_vid, key=lambda v: (0, int(v)) if v.isdigit() else (1, v)):
            rows = sorted(by_vid[vid], key=lambda r: int(r['frame']))[::strides[split]]
            for r in rows:
                rec = {'vid': r['vid'], 'frame': int(r['frame']), 'cam': r['cam'], 'gt': r['gt']}
                for k in ('cam', 'gt'):
                    if not os.path.exists(os.path.join(cfg['root'], rec[k])):
                        missing += 1
                fo.write(json.dumps(rec) + '\n'); n += 1
    print(f'{split:5s} {src} -> {dst}: {len(by_vid)} videos, {n} rows (stride {strides[split]}), {missing} missing')
    if missing:
        sys.exit(f'ERROR: {missing} missing files in {split}')
    info['manifests'][split] = {'path': src, 'sha256': sha256(src)}
    info['rows'][split] = {'videos': len(by_vid), 'rows': n}
    sums.append(f'{sha256(dst)}  {dst}')
json.dump(info, open(os.path.join(out, 'build.json'), 'w'), indent=2)
open(os.path.join(out, 'SHA256SUMS'), 'w').write('\n'.join(sums) + '\n')
