"""PyTorch port of ML-Net (Cornia, Baraldi, Serra, Cucchiara, ICPR 2016) for zero-shot inference from
the authors' released Keras-1/Theano weights (mlnet_salicon_weights.pkl — an HDF5 file despite the name).

Intended path: /opt/gazehead/mlnet_torch.py

Architecture, from model.py of marcellacornia/mlnet (master):
  VGG-16 conv1_1..conv5_3 (3x3, 'same', relu); pool1-3 are 2x2 stride 2, pool4 is 2x2 stride 1, no pool5.
  concat[conv3_pool, conv4_pool, conv5_3] (256+512+512 = 1280 ch at 1/8) -> Dropout(0.5)
  -> Conv 3x3, 64, relu -> Conv 1x1, 1, relu -> x * bilinear_upsampling(1 + W_prior[6x8], 10) -> relu.
  Input 3 x 480 x 640 (BGR, Caffe means 103.939/116.779/123.68, no scaling); output 1 x 60 x 80.
  Parameter count: 15,452,145 (14,714,688 VGG convs + 737,344 + 65 + 48).

Preprocessing, from utilities.py: aspect-preserving resize of the image to fit 480x640, zero-pad to
480x640 (centered), then mean subtraction. Postprocessing (original): resize the 60x80 map back to the
image size, crop the padding, divide by max, x255. For the benchmark we save the NATIVE 60x80 map with
the padding removed (crop_native) and let evaluate.py do the resize + blur.

Two things the Keras-1/Theano stack does differently from PyTorch, which the released weights depend on:
  * theano.tensor.nnet.conv2d is a true convolution (kernel flipped); PyTorch is cross-correlation
    -> kernel_flip=True flips every conv kernel on load.
  * Keras-1 theano_backend.pool2d(border_mode='same') pads every side by 1 (with -inf for max), pools with
    ignore_border=True, then keeps the first ceil(n/stride) outputs. Net effect: output (i, j) covers input
    rows i*s-1..i*s, i.e. the window is shifted up/left by one compared with PyTorch/TF 'same'
    -> pool='theano'.  pool='standard' is the TF/PyTorch convention (pad bottom/right).
Both are switches so /opt/gazehead/tools/check_mlnet_port.py can pick the variant that reproduces the
authors' published maps; the chosen values are recorded in run.json by predict_mlnet.py.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPE_R, SHAPE_C = 480, 640          # config.py shape_r, shape_c
NET_DS, PRIOR_DS = 8, 10             # downsampling_factor_net, downsampling_factor_product
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)
N_PARAMS_EXPECTED = 15_452_145

_VGG_CHANS = [(3, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256), (256, 256),
              (256, 512), (512, 512), (512, 512), (512, 512), (512, 512), (512, 512)]


def _maxpool_same(x, stride, variant):
    neg = float("-inf")
    if variant == "theano":
        x = F.pad(x, (1, 0, 1, 0), value=neg)
        return F.max_pool2d(x, 2, stride)
    if variant == "standard":
        if stride == 2:
            return F.max_pool2d(x, 2, 2, ceil_mode=True)
        x = F.pad(x, (0, 1, 0, 1), value=neg)
        return F.max_pool2d(x, 2, 1)
    raise ValueError(variant)


class MLNet(nn.Module):
    def __init__(self, pool="theano"):
        super().__init__()
        assert pool in ("theano", "standard")
        self.pool = pool
        self.convs = nn.ModuleList([nn.Conv2d(i, o, 3, padding=1) for i, o in _VGG_CHANS])
        self.int_conv = nn.Conv2d(1280, 64, 3, padding=1)
        self.pre_final = nn.Conv2d(64, 1, 1)
        self.prior = nn.Parameter(torch.zeros(SHAPE_R // NET_DS // PRIOR_DS, SHAPE_C // NET_DS // PRIOR_DS))

    def forward(self, x):
        c, r, p = self.convs, F.relu, self.pool
        x = r(c[1](r(c[0](x))))
        x = _maxpool_same(x, 2, p)
        x = r(c[3](r(c[2](x))))
        x = _maxpool_same(x, 2, p)
        x = r(c[6](r(c[5](r(c[4](x))))))
        p3 = _maxpool_same(x, 2, p)
        x = r(c[9](r(c[8](r(c[7](p3))))))
        p4 = _maxpool_same(x, 1, p)
        c5 = r(c[12](r(c[11](r(c[10](p4))))))
        y = torch.cat([p3, p4, c5], 1)
        y = r(self.int_conv(y))          # dropout is identity at inference
        y = r(self.pre_final(y))
        prior = F.interpolate((1 + self.prior)[None, None], scale_factor=PRIOR_DS,
                              mode="bilinear", align_corners=False)
        assert prior.shape[-2:] == y.shape[-2:], (prior.shape, y.shape)
        return r(y * prior)              # (B, 1, 60, 80), non-negative


# ----------------------------------------------------------------------------- weights

def read_keras_weights(path):
    """All weight arrays of a Keras-1 save_weights() HDF5 file, as (name, array), in model-layer order."""
    import h5py

    with open(path, "rb") as f:
        magic = f.read(8)
    if not magic.startswith(b"\x89HDF"):
        raise ValueError(f"{path}: not an HDF5 file (first bytes {magic!r}); Keras-1 save_weights() "
                         f"writes HDF5 whatever the extension, so this is probably a bad download")
    arrays = []

    def dec(s):
        return s.decode() if isinstance(s, bytes) else str(s)

    with h5py.File(path, "r") as h:
        f = h["model_weights"] if "model_weights" in h else h   # model.save() vs save_weights()
        if "layer_names" in f.attrs:                       # Keras >= 1.0 format
            for ln in f.attrs["layer_names"]:
                g = f[dec(ln)]
                for wn in g.attrs.get("weight_names", []):
                    arrays.append((dec(wn), np.array(g[dec(wn)])))
        else:                                              # Keras 0.x format
            for i in range(int(f.attrs["nb_layers"])):
                g = f[f"layer_{i}"]
                for p in range(int(g.attrs["nb_params"])):
                    arrays.append((f"layer_{i}/param_{p}", np.array(g[f"param_{p}"])))
    return arrays


def load_keras_weights(model, path, kernel_flip=True):
    """Fill `model` from the released checkpoint. Returns the list of weight names consumed."""
    arrays = read_keras_weights(path)
    convs = [(n, a) for n, a in arrays if a.ndim == 4]
    biases = [(n, a) for n, a in arrays if a.ndim == 1]
    priors = [(n, a) for n, a in arrays if a.ndim == 2]
    targets = list(model.convs) + [model.int_conv, model.pre_final]
    if len(convs) != len(targets) or len(biases) != len(targets) or len(priors) != 1:
        raise ValueError(f"unexpected checkpoint layout: {len(convs)} conv kernels, {len(biases)} biases, "
                         f"{len(priors)} 2-D arrays; names: {[n for n, _ in arrays]}")
    with torch.no_grad():
        for layer, (wn, w), (bn, b) in zip(targets, convs, biases):
            if kernel_flip:
                w = w[:, :, ::-1, ::-1]
            w = np.array(w, dtype=np.float32, order="C", copy=True)  # size-1 flipped axes keep negative strides otherwise
            if tuple(layer.weight.shape) != w.shape:
                raise ValueError(f"{wn}: shape {w.shape} != {tuple(layer.weight.shape)}")
            layer.weight.copy_(torch.from_numpy(w))
            layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))
        pw = priors[0][1].astype(np.float32)
        if tuple(model.prior.shape) != pw.shape:
            raise ValueError(f"prior shape {pw.shape} != {tuple(model.prior.shape)}")
        model.prior.copy_(torch.from_numpy(pw))
    return [n for n, _ in arrays]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- pre/post

def pad_resize_bgr(img):
    """utilities.padding() for a BGR uint8 image. Returns the 480x640x3 uint8 canvas and the content box
    (r0, nr, c0, nc) = rows/cols of the canvas that hold image content."""
    import cv2

    h, w = img.shape[:2]
    out = np.zeros((SHAPE_R, SHAPE_C, 3), np.uint8)
    if h / SHAPE_R > w / SHAPE_C:
        nc = (w * SHAPE_R) // h
        img = cv2.resize(img, (nc, SHAPE_R))
        nc = min(nc, SHAPE_C)
        c0 = (SHAPE_C - nc) // 2
        out[:, c0:c0 + nc] = img[:, :nc]
        box = (0, SHAPE_R, c0, nc)
    else:
        nr = (h * SHAPE_C) // w
        img = cv2.resize(img, (SHAPE_C, nr))
        nr = min(nr, SHAPE_R)
        r0 = (SHAPE_R - nr) // 2
        out[r0:r0 + nr] = img[:nr]
        box = (r0, nr, 0, SHAPE_C)
    return out, box


def preprocess_bgr(img):
    """BGR uint8 image -> float32 (3,480,640) network input, plus the content box."""
    padded, box = pad_resize_bgr(img)
    x = padded.astype(np.float32) - MEAN_BGR
    return np.ascontiguousarray(x.transpose(2, 0, 1)), box


def _resample_axis(a, axis, start, n):
    """Linear resample of `a` along `axis` at pixel-index coordinates start + k, k = 0..n-1."""
    coords = start + np.arange(n)
    lo = np.floor(coords).astype(int)
    frac = (coords - lo).astype(np.float32)
    last = a.shape[axis] - 1
    hi = np.clip(lo + 1, 0, last)
    lo = np.clip(lo, 0, last)
    A = np.take(a, lo, axis)
    B = np.take(a, hi, axis)
    shape = [1] * a.ndim
    shape[axis] = n
    frac = frac.reshape(shape)
    return A * (1 - frac) + B * frac


def crop_native(out, box):
    """Remove the padding from a native (60x80) ML-Net map without upsampling it.

    Content rows r0..r0+nr-1 of the 480x640 input occupy native rows r0/8..(r0+nr)/8. For 16:9 input
    (720p -> r0 = 60, nr = 360) that is 7.5..52.5, so the map is linearly resampled at the half-pixel
    offset; for pixel-aligned boxes this is a plain slice.
    """
    r0, nr, c0, nc = box
    out = _resample_axis(np.asarray(out, np.float32), 0, r0 / NET_DS, int(round(nr / NET_DS)))
    out = _resample_axis(out, 1, c0 / NET_DS, int(round(nc / NET_DS)))
    return np.ascontiguousarray(out, dtype=np.float32)


def postprocess_original(pred, shape_r, shape_c):
    """Exact port of utilities.postprocess_predictions(): resize to the image, crop padding, /max*255.
    Only used by the port check (compares against the authors' published maps)."""
    import cv2

    ph, pw = pred.shape
    if shape_r / ph > shape_c / pw:
        new_cols = (pw * shape_r) // ph
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (ph * shape_c) // pw
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]
    return img / np.max(img) * 255
