"""
Dataset-generic manifest loader for SCOUT (frames-only, w/o task), driven by the gazehead
benchmark manifests.  Successor of dataloader_cocatt.py (kept as the record of the CoCAtt run).

Intended path: /opt/SCOUT/dataloader_gazehead.py

train.py builds it as  dataset_class(clip_size, old_gt=..., mode=..., weight_type=..., img_size=...,
task_attributes=..., map_params=...)  where dataset_class is functools.partial(GazeheadDataset, dataset=NAME).
Items: (obs_img[T,3,H,W], gt[Hg,Wg] float in [0,1], {}, [], (vid_idx, frame, idx)).

DATASETS[name] gives manifest prefix, data root, fps and the train/val/test strides.  Frame file
naming is inferred from each manifest row's `cam` path: the file index may be offset from the
manifest `frame` (CoCAtt: file = frame+1 .png; BDD-A: file = frame .jpg), zero-padding and
extension are read from the path.  Clips are obs_length frames ending at the target frame, read
from the raw frame directory (every camera frame exists there), clamped at frame 0 (repeat).
GT: convert('L'), /255 if max > 1 (what SCOUT's own BDD-A loader does).
"""
import os
import re
import csv
import functools
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

MANIFEST_DIR = '/opt/gazehead/meta/splits'

DATASETS = {
    # BDD-A, 29 fps.  Strides are SCOUT's own: obs_length//2 = 8 for train/val, 1 for test.
    # test manifest = bdda_test_scored (the scored set; SCOUT's own sample list minus empty-GT rows).
    'bdda': dict(root='/srv/datasets/BDDA', fps=29,
                 manifests={'train': 'bdda_train', 'val': 'bdda_val', 'test': 'bdda_test_scored'},
                 step={'train': 8, 'val': 8, 'test': 1}),
    # CoCAtt, 4 fps (see cocatt_SOTA_DECISION.txt for the stride-2 rationale).
    'cocatt': dict(root='/srv/datasets/CoCAtt', fps=4,
                   manifests={'train': 'cocatt_train', 'val': 'cocatt_val', 'test': 'cocatt_test'},
                   step={'train': 2, 'val': 2, 'test': 1}),
}

_FILE_RE = re.compile(r'^(\d+)(\.[A-Za-z0-9]+)$')


def _vid_sort_key(v):
    return (0, int(v)) if v.isdigit() else (1, v)


class GazeheadDataset(Dataset):
    def __init__(self, obs_length=16, img_size=(224, 224), mode='train', dataset='bdda',
                 task_attributes=None, map_params=None, step=None, **kwargs):
        self.cfg = DATASETS[dataset]
        self.dataset = dataset
        self.obs_length = obs_length
        self.img_size = tuple(img_size)
        self.mode = mode
        self.step = self.cfg['step'][mode] if step is None else step
        self.task_attributes = task_attributes
        self.map_params = map_params
        self.sample_list = []
        self.vid_names = []
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @classmethod
    def for_dataset(cls, name):
        return functools.partial(cls, dataset=name)

    # ---- setup -------------------------------------------------------------
    def setup(self, vid_id=-1):
        self.manifest = os.path.join(MANIFEST_DIR, self.cfg['manifests'][self.mode] + '.csv')
        rows_by_vid = {}
        with open(self.manifest) as f:
            for r in csv.DictReader(f):
                rows_by_vid.setdefault(r['vid'], []).append((int(r['frame']), r['cam'], r['gt']))
        self.vid_names = sorted(rows_by_vid, key=_vid_sort_key)
        self.vid2idx = {v: i for i, v in enumerate(self.vid_names)}
        self.sample_list = []
        for v in self.vid_names:
            rows = sorted(rows_by_vid[v])
            for frame, cam, gt in rows[::self.step]:
                d, fn = os.path.split(cam)
                m = _FILE_RE.match(fn)
                if m is None:
                    raise ValueError(f'cannot parse frame file name {cam}')
                offset = int(m.group(1)) - frame
                self.sample_list.append((self.vid2idx[v], frame, d, len(m.group(1)), m.group(2), offset, gt))
        print(f'Loading {self.mode} data [{self.dataset}]: {len(self.vid_names)} videos, '
              f'{sum(len(r) for r in rows_by_vid.values())} manifest frames, '
              f'{len(self.sample_list)} samples (step {self.step}, clip {self.obs_length}, '
              f'{self.obs_length / self.cfg["fps"]:.2f} s)')

    def get_sample_weights(self):
        return None

    def get_frame_weights(self):
        return None

    # ---- items -------------------------------------------------------------
    def clip_frames(self, frame):
        return [max(0, frame - self.obs_length + 1 + i) for i in range(self.obs_length)]

    def frame_path(self, img_dir, width, ext, offset, frame):
        return os.path.join(self.cfg['root'], img_dir, f'{frame + offset:0{width}d}{ext}')

    def load_image(self, path):
        try:
            return Image.open(path).convert('RGB')
        except OSError:
            print('ERROR: could not load', path)
            raise

    def get_gt(self, gt_rel):
        gt = np.array(Image.open(os.path.join(self.cfg['root'], gt_rel)).convert('L')).astype('float')
        if gt.max() > 1.0:
            gt = gt / 255.0
        return gt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        vid_idx, frame, img_dir, width, ext, offset, gt_rel = self.sample_list[idx]
        imgs = [self.img_transform(self.load_image(self.frame_path(img_dir, width, ext, offset, k)))
                for k in self.clip_frames(frame)]
        obs_img = torch.FloatTensor(torch.stack(imgs, dim=0))
        gt = torch.FloatTensor(self.get_gt(gt_rel))
        return obs_img, gt, {}, [], (vid_idx, frame, idx)
