"""
CoCAtt dataset for SalM² (SaliencyMamba).

Intended path: /opt/SaliencyMamba/utils/datasets/CoCAtt.py

Reads the gazehead benchmark manifests exported as JSON lines by
/opt/gazehead/tools/build_salm2_cocatt_json.py.  Each line is a dict:
    {"vid": "A_0", "frame": 12, "cam": "<rel path>", "gt": "<rel path>"}
with paths relative to args.root's `data_root` (see below).

Frames-only: only the camera frame is an input.  GT is the authors' per-frame
eye-tracker heatmap, read as grayscale (stored as RGB, so cv2 flag 0), resized
to img_shape and renormalised to max 1 — same contract as ImageList_BDDA.

Resolution: CoCAtt frames are 512x256 (2:1).  Like the authors do for BDD-A
(1280x720 -> 256x256), the frame is squashed to args.img_shape (256x256);
the scorer restores the aspect ratio when it resizes predictions to GT size.
"""
import os
import cv2
import torch
import numpy as np
import imageio.v2 as io
from torch.utils.data import Dataset


class ImageList_CoCAtt(Dataset):
    def __init__(self, args, imgs, for_train=False):
        # args.root holds the json lists; data_root.txt inside it points at the
        # dataset root (/srv/datasets/CoCAtt) so --root stays self-contained.
        self.root = args.root
        with open(os.path.join(args.root, 'data_root.txt')) as f:
            self.data_root = f.read().strip()
        self.imgs = imgs
        self.img_shape = tuple(args.img_shape)
        self.for_train = for_train

    def _read_gt(self, rel):
        lab = cv2.imread(os.path.join(self.data_root, rel), 0)
        if lab is None:
            raise FileNotFoundError(rel)
        lab = cv2.resize(lab, self.img_shape, interpolation=cv2.INTER_CUBIC)
        lab = lab.astype('float32') / 255.0
        m = lab.max()
        if m > 0:
            lab = lab / m
        return lab

    def __getitem__(self, index):
        rec = self.imgs[index]
        img = io.imread(os.path.join(self.data_root, rec['cam']))
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0

        lab = self._read_gt(rec['gt'])

        if self.for_train and np.random.uniform() < 0.5:   # same aug as BDDA
            img = img[:, ::-1]
            lab = lab[:, ::-1]

        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        lab = np.ascontiguousarray(lab[None, ...])
        return torch.from_numpy(img), torch.from_numpy(lab)

    def __len__(self):
        return len(self.imgs)

    # used by the predictor to name outputs
    def meta(self, index):
        rec = self.imgs[index]
        return rec['vid'], int(rec['frame'])


# SalM² uses seq_len == 1 for salmm; keep the name the builder expects.
ImageList_CoCAtt_Continuous = ImageList_CoCAtt
