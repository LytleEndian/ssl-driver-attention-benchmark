#!/usr/bin/env python3
"""
Export the frozen CoCAtt benchmark manifests as SalM²-style JSON-lines lists.

Intended path: /opt/gazehead/tools/build_salm2_cocatt_json.py
Output dir:    /srv/datasets/CoCAtt/salm2/   (train.json, valid.json, test.json,
               data_root.txt, SHA256SUMS) — pass this dir as SalM² --root.

Rows are copied verbatim from meta/splits/cocatt_{train,val,test}.csv
(vid, frame, cam, gt only — the metadata columns are never exported, so they
cannot leak into the model).  Counts are checked against the manifests and
cocatt_manifest_summary.json.
"""
import csv, json, os, hashlib, sys

SPLITS = '/opt/gazehead/meta/splits'
OUT = '/srv/datasets/CoCAtt/salm2'
DATA_ROOT = '/srv/datasets/CoCAtt'
NAMES = {'train': 'train', 'val': 'valid', 'test': 'test'}

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'data_root.txt'), 'w') as f:
    f.write(DATA_ROOT + '\n')

sums = []
for split, name in NAMES.items():
    src = os.path.join(SPLITS, f'cocatt_{split}.csv')
    dst = os.path.join(OUT, f'{name}.json')
    n = 0
    missing = 0
    with open(src) as fi, open(dst, 'w') as fo:
        for r in csv.DictReader(fi):
            rec = {'vid': r['vid'], 'frame': int(r['frame']), 'cam': r['cam'], 'gt': r['gt']}
            for k in ('cam', 'gt'):
                if not os.path.exists(os.path.join(DATA_ROOT, rec[k])):
                    missing += 1
            fo.write(json.dumps(rec) + '\n')
            n += 1
    print(f'{split:5s} -> {dst}: {n} rows, {missing} missing files, manifest sha {sha256(src)[:12]}')
    if missing:
        sys.exit(f'ERROR: {missing} missing files in {split}')
    sums.append(f'{sha256(dst)}  {dst}')

with open(os.path.join(OUT, 'SHA256SUMS'), 'w') as f:
    f.write('\n'.join(sums) + '\n')
print('expected: train 58529, val 15351, test 15029')
