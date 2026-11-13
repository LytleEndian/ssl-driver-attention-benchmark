"""Saliency metrics, batched torch — exact ports of SCOUT's loss.py (the code behind SCOUT's
published BDD-A / DR(eye)VE numbers; NOT saliency_metrics.py, which test.py does not use).

All functions take (B, H, W) float32 tensors: s_map = prediction, gt = ground truth, both at
the scoring resolution, and return a (B,) tensor of per-frame scores.

Deliberately replicated quirks (document in Methods):
  * KLD:  KL(G||P) with eps = float32 eps (1.19e-7) inside the log AND added to the sums.
  * NSS:  continuous-map NSS = sum(z(P) * G) / sum(G), z uses the unbiased std (+eps).
  * SIM:  prediction is min-max normalised then sum-normalised; GT is ONLY sum-normalised
          (SCOUT computes a min-max GT and then discards it).
  * CC:   Pearson with unbiased std, no eps (constant maps -> NaN).

Pre-processing (SCOUT test.py): prediction resized to GT size with torchvision Resize
(bilinear, antialias=True), then GaussianBlur(kernel 11, sigma 2). See `scout_postprocess`.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF

EPS = float(np.finfo(np.float32).eps)  # 1.1920929e-07, as in loss.py


def _flat(x):
    return x.reshape(x.shape[0], -1)


def kldiv(s_map, gt):
    s = _flat(s_map)
    g = _flat(gt)
    s = s / (s.sum(1, keepdim=True) + EPS)
    g = g / (g.sum(1, keepdim=True) + EPS)
    return (g * torch.log(EPS + g / (s + EPS))).sum(1)


def cc(s_map, gt):
    s = _flat(s_map)
    g = _flat(gt)
    s = (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True)
    g = (g - g.mean(1, keepdim=True)) / g.std(1, keepdim=True)
    ab = (s * g).sum(1)
    aa = (s * s).sum(1)
    bb = (g * g).sum(1)
    return ab / torch.sqrt(aa * bb)


def sim(s_map, gt):
    s = _flat(s_map)
    g = _flat(gt)
    smin = s.min(1, keepdim=True)[0]
    smax = s.max(1, keepdim=True)[0]
    s = (s - smin) / (smax - smin)
    s = s / s.sum(1, keepdim=True)
    g = g / g.sum(1, keepdim=True)          # raw GT, not min-max (SCOUT quirk)
    return torch.minimum(s, g).sum(1)


def nss(s_map, gt):
    s = _flat(s_map)
    g = _flat(gt)
    s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + EPS)
    return (s * g).sum(1) / g.sum(1)


METRICS = {"KLD": kldiv, "CC": cc, "SIM": sim, "NSS": nss}
LOWER_IS_BETTER = {"KLD"}


def all_metrics(s_map, gt):
    return {k: f(s_map, gt) for k, f in METRICS.items()}


def resize_to(pred, hw):
    """(B,h,w) -> (B,H,W), torchvision Resize semantics (bilinear, antialias=True)."""
    return TF.resize(pred.unsqueeze(1), list(hw), antialias=True).squeeze(1)


def blur(pred):
    """SCOUT's transforms.GaussianBlur(11, 2): sigma fixed at 2, reflect padding."""
    return TF.gaussian_blur(pred.unsqueeze(1), [11, 11], [2.0, 2.0]).squeeze(1)


def gt_from_uint8(gt_u8):
    """SCOUT get_gt(): float64, divide by 255 only if the frame's max > 1, then float32."""
    g = gt_u8.double()
    mx = g.amax(dim=(1, 2), keepdim=True)
    g = torch.where(mx > 1.0, g / 255.0, g)
    return g.float()
