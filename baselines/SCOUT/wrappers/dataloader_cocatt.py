"""
CoCAtt dataset for SCOUT (Kotseruba & Tsotsos, IV 2024), frames-only (w/o task).

Intended path: /opt/SCOUT/dataloader_cocatt.py

Mirrors the BDDADataset contract used by /opt/SCOUT/train.py:
    ds = CoCAttDataset(clip_size, old_gt=False, mode=..., weight_type=..., img_size=..., task_attributes=..., map_params=...)
    ds.setup(); ds.get_sample_weights()
    ds[i] -> (obs_img[T,3,H,W], gt[Hg,Wg] float in [0,1], {}, [], (vid_idx, frame, i))

Differences from the BDD-A loader, all deliberate (see cocatt_SOTA_DECISION.txt):
- Samples come from the frozen gazehead manifests (meta/splits/cocatt_{train,val,test}.csv),
  one sample per manifest row (target = that frame).  No vehicle data, no task/map inputs.
- Clips are 16 frames ending at the target frame, read from the raw images_4hz PNGs, which
  contain every camera frame (eye-tracker dropout frames included), so history is real.
  Frames before the session start repeat frame 0 (same padding rule as the V-JEPA runs).
- Manifest `frame` is 0-based; the PNG name is frame+1.
- `vid` is a string ("A_0"); the collate path in train.py needs ints, so samples carry a
  vid index and `self.vid_names` maps it back.
- GT: 512x256 heatmap stored as RGB -> convert('L') -> /255 (per-frame max-normalised
  already, so max is 1).  SCOUT resizes its prediction to this size for the loss.
- Stride: train/val step 2 (matches the DINOv3/V-JEPA recipe: 29,266 / 7,687 samples at 4 fps,
  comparable to BDD-A sample counts), test step 1 (all 15,029 frames).  SCOUT's own
  obs_length//2 = 8-frame step was tuned for ~30 fps video and would be 2 s at 4 fps.
"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

MANIFEST_DIR = '/opt/gazehead/meta/splits'
DATA_ROOT = '/srv/datasets/CoCAtt'
DEFAULT_STEP = {'train': 2, 'val': 2, 'test': 1}


class CoCAttDataset(Dataset):
    def __init__(self, obs_length=16, img_size=(224, 224), mode='train',
                 task_attributes=None, map_params=None, step=None, **kwargs):
        self.obs_length = obs_length
        self.img_size = tuple(img_size)
        self.mode = mode
        self.step = DEFAULT_STEP[mode] if step is None else step
        self.task_attributes = task_attributes
        self.map_params = map_params
        self.sample_list = []
        self.vid_names = []
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # ---- setup -------------------------------------------------------------
    def setup(self, vid_id=-1):
        self.manifest = os.path.join(MANIFEST_DIR, f'cocatt_{self.mode}.csv')
        rows_by_vid = {}
        with open(self.manifest) as f:
            for r in csv.DictReader(f):
                rows_by_vid.setdefault(r['vid'], []).append((int(r['frame']), r['cam'], r['gt']))
        self.vid_names = sorted(rows_by_vid)
        self.vid2idx = {v: i for i, v in enumerate(self.vid_names)}
        self.sample_list = []
        for v in self.vid_names:
            rows = sorted(rows_by_vid[v])
            for frame, cam, gt in rows[::self.step]:
                self.sample_list.append((self.vid2idx[v], frame, os.path.dirname(cam), gt))
        print(f'Loading {self.mode} data: {len(self.vid_names)} videos, '
              f'{sum(len(r) for r in rows_by_vid.values())} manifest frames, '
              f'{len(self.sample_list)} samples (step {self.step}, clip {self.obs_length})')

    def get_sample_weights(self):
        return None

    def get_frame_weights(self):
        return None

    # ---- items -------------------------------------------------------------
    def clip_frames(self, frame):
        return [max(0, frame - self.obs_length + 1 + i) for i in range(self.obs_length)]

    def load_image(self, img_dir, frame):
        p = os.path.join(DATA_ROOT, img_dir, f'{frame + 1:05d}.png')
        try:
            return Image.open(p).convert('RGB')
        except OSError:
            print('ERROR: could not load', p)
            raise

    def get_images(self, img_dir, frame):
        return [self.img_transform(self.load_image(img_dir, k)) for k in self.clip_frames(frame)]

    def get_gt(self, gt_rel):
        gt = np.array(Image.open(os.path.join(DATA_ROOT, gt_rel)).convert('L')).astype('float')
        if gt.max() > 1.0:
            gt = gt / 255.0
        return gt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        vid_idx, frame, img_dir, gt_rel = self.sample_list[idx]
        obs_img = torch.FloatTensor(torch.stack(self.get_images(img_dir, frame), dim=0))
        gt = torch.FloatTensor(self.get_gt(gt_rel))
        return obs_img, gt, {}, [], (vid_idx, frame, idx)
