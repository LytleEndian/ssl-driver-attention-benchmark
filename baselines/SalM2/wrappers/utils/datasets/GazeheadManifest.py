"""
Dataset-generic gazehead-manifest loader for SalM² (SaliencyMamba).

Intended path: /opt/SaliencyMamba/utils/datasets/GazeheadManifest.py
Register in utils/build_datasets.py as  'BDDA_gh': ImageList_BDDA_gh  (both dicts).

Reads the JSON lines written by /opt/gazehead/tools/build_salm2_json.py (vid, frame, cam, gt; paths
relative to data_root.txt inside args.root).  Same contract as the authors' ImageList_BDDA:
RGB frame, cv2 INTER_CUBIC resize to args.img_shape (the 16:9 frame is squashed to 256x256, as
their BDD-A loader does), /255, no mean/std; GT read grayscale, resized, /255, horizontal-flip
augmentation in train.  `renorm` divides the GT by its per-frame max; the authors' BDD-A loader
does NOT (TrafficGaze/DrFixD do), so ImageList_BDDA_gh keeps renorm=False for fidelity.
"""
import os
import cv2
import torch
import numpy as np
import imageio.v2 as io
from torch.utils.data import Dataset


class ImageList_Gazehead(Dataset):
    renorm = False

    def __init__(self, args, imgs, for_train=False):
        self.root = args.root
        with open(os.path.join(args.root, 'data_root.txt')) as f:
            self.data_root = f.read().strip()
        self.imgs = imgs
        self.img_shape = tuple(args.img_shape)
        self.for_train = for_train

    def __getitem__(self, index):
        rec = self.imgs[index]
        img = io.imread(os.path.join(self.data_root, rec['cam']))
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32') / 255.0

        lab = cv2.imread(os.path.join(self.data_root, rec['gt']), 0)
        if lab is None:
            raise FileNotFoundError(rec['gt'])
        lab = cv2.resize(lab, self.img_shape, interpolation=cv2.INTER_CUBIC)
        lab = lab.astype('float32') / 255.0
        if self.renorm and lab.max() > 0:
            lab = lab / lab.max()

        if self.for_train and np.random.uniform() < 0.5:
            img = img[:, ::-1]
            lab = lab[:, ::-1]

        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        lab = np.ascontiguousarray(lab[None, ...])
        return torch.from_numpy(img), torch.from_numpy(lab)

    def __len__(self):
        return len(self.imgs)

    def meta(self, index):
        rec = self.imgs[index]
        return rec['vid'], int(rec['frame'])


class ImageList_BDDA_gh(ImageList_Gazehead):
    renorm = False


ImageList_BDDA_gh_Continuous = ImageList_BDDA_gh
