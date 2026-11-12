"""
/opt/gazehead/tools/gbvs_fast.py

Speed-only rewrite of the GBVS Python port by shreelock
(https://github.com/shreelock/gbvs, commit 4c7e3c58e786d053ec4ae6fb9d03bdfeea2c539a),
which is itself a port of Harel, Koch & Perona's MATLAB GBVS (NIPS 2006).

The algorithm is the port's, operation for operation. Only these things change:
  * the 28x32 graph distance matrix is loaded once from an absolute path (the port reloads
    it from a cwd-relative path 42 times per frame), and exp(-D / 2 sigma^2) is cached per sigma;
  * Markov power iteration uses ndarray.sum() and a vector 2-norm instead of Python's builtin
    sum() over rows and np.linalg.norm(ord=2), which runs an SVD on an (n,1) matrix;
  * sklearn.preprocessing.normalize(axis=0, norm='l1') is replaced by the same column division;
  * numpy.matlib.repmat is replaced by np.tile (same values);
  * compute() returns the port's min-max normalised 28x32 master map BEFORE its final cubic
    upsampling (evaluate.py resizes predictions to GT size itself); compute_full() returns
    the port's full-resolution output for equivalence checks.

Equivalence with the pristine port is checked by /opt/gazehead/tools/check_gbvs_port.py.
Differences are floating-point summation order only.

Known deviations of the PORT from Harel's MATLAB GBVS (inherited, not changed here):
  * activation dissimilarity is |a - b| (the paper uses |log(a / b)|);
  * colour features are |b - min(r,g)| / L and |r - g| / L with L = max(r,g,b);
  * every pyramid level is squashed to a fixed 28x32 graph regardless of aspect ratio;
  * no final blur or border attenuation; output is min-max normalised.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import scipy.io

DEFAULT_MAT = Path("/opt/gbvs/saliency_models/resources/28__32__m__2.mat")
GRAPH_H, GRAPH_W = 28, 32

PARAMS = {
    "gaborparams": {"stddev": 2, "elongation": 2, "filterSize": -1, "filterPeriod": np.pi},
    "sigma_frac_act": 0.15,
    "sigma_frac_norm": 0.06,
    "max_level": 4,
    "thetas": [0, 45, 90, 135],
}

_D = None
_FAB = {}
_GABOR = None


def set_graph_mat(path):
    """Load the port's 28x32 graph distance matrix (call once per process)."""
    global _D, _FAB
    f = scipy.io.loadmat(str(path))
    _D = np.array(f["grframe"])[0][0][0]
    _FAB = {}


def _fab(sigma):
    if _D is None:
        set_graph_mat(DEFAULT_MAT)
    if sigma not in _FAB:
        denom = 2 * pow(sigma, 2)
        _FAB[sigma] = np.exp(-np.divide(_D, denom))
    return _FAB[sigma]


# ---------------------------------------------------------------- Gabor (port: gaborKernelCalculator.py)
def _gabor_kernel(gp, angle, phase):
    major_sd = gp["stddev"]
    minor_sd = major_sd * gp["elongation"]
    max_sd = max(major_sd, minor_sd)
    sz = gp["filterSize"]
    sz = math.ceil(max_sd * math.sqrt(10)) if sz == -1 else math.floor(sz / 2)
    psi = np.pi / 180 * phase
    rt = np.pi / 180 * angle
    omega = 2 * np.pi / gp["filterPeriod"]
    co, si = math.cos(rt), -math.sin(rt)
    major_sigq, minor_sigq = 2 * pow(major_sd, 2), 2 * pow(minor_sd, 2)
    vec = range(-int(sz), int(sz) + 1)
    vlen = len(vec)
    vco = [i * co for i in vec]
    vsi = [i * si for i in vec]
    major = np.tile(np.asarray(vco).transpose(), (vlen, 1)).transpose() + np.tile(np.asarray(vsi), (vlen, 1))
    minor = np.tile(np.asarray(vsi).transpose(), (vlen, 1)).transpose() + np.tile(np.asarray(vco), (vlen, 1))
    result = np.multiply(np.cos(omega * major + psi),
                         np.exp(-np.power(major, 2) / major_sigq - np.power(minor, 2) / minor_sigq))
    f = np.subtract(result, np.mean(result.reshape(-1)))
    return np.divide(f, np.sqrt(np.sum(np.power(f.reshape(-1), 2))))


def _gabor_kernels():
    global _GABOR
    if _GABOR is None:
        gp = PARAMS["gaborparams"]
        _GABOR = {th: (_gabor_kernel(gp, th, 0), _gabor_kernel(gp, th, 90)) for th in PARAMS["thetas"]}
    return _GABOR


# ---------------------------------------------------------------- features (port: colorFeatureMaps / orientationFeatureMaps)
def _color_maps(r, g, b, L):
    cby = np.divide(np.abs(np.subtract(b, np.minimum(r, g))), L, out=np.zeros_like(L), where=L != 0)
    crg = np.divide(np.abs(np.subtract(r, g)), L, out=np.zeros_like(L), where=L != 0)
    return cby, crg, L


def _orientation_maps(L):
    out = []
    for th in PARAMS["thetas"]:
        k0, k90 = _gabor_kernels()[th]
        o1 = cv2.filter2D(L, -1, k0, borderType=cv2.BORDER_REPLICATE)
        o2 = cv2.filter2D(L, -1, k90, borderType=cv2.BORDER_REPLICATE)
        out.append(np.add(abs(o1), abs(o2)))
    return out


def _pyramid(ch, max_level):
    pyr = [cv2.pyrDown(ch)]
    for i in range(1, max_level):
        pyr.append(cv2.pyrDown(pyr[i - 1]))
    return pyr[1:]


# ---------------------------------------------------------------- graph (port: markovChain.py / graphBasedActivation.py)
def _col_l1_normalize(w):
    s = np.abs(w).sum(axis=0)
    s[s == 0.0] = 1.0
    return w / s


def _markov_solve(mat, tol):
    n = mat.shape[0]
    v = np.divide(np.ones((n, 1), dtype=np.float32), n)
    diff = 1
    while diff > tol:
        oldv = v
        v = np.dot(mat, v)
        d = oldv - v
        diff = math.sqrt(float((d * d).sum()))
        s = v.sum()
        if not (s >= 0 and s < np.inf):
            v = oldv          # port: v = oldoldv, which always equals oldv there
            break
    return np.divide(v, v.sum())


def _activation(m, sigma):
    a = np.ravel(m, order="F")
    w = _fab(sigma) * np.abs(a[:, None] - a[None, :])
    e = _markov_solve(_col_l1_normalize(w), 0.0001)
    return np.reshape(e, m.shape, order="F")


def _normalization(m, sigma):
    a = np.ravel(m, order="F")
    w = _fab(sigma) * np.abs(a)[:, None]
    e = _markov_solve(_col_l1_normalize(w), 0.0001)
    return np.reshape(e, m.shape, order="F")


# ---------------------------------------------------------------- driver (port: gbvs.run)
def _master_map(image_bgr_uint8):
    image = image_bgr_uint8 / 255.0
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    L = np.maximum(np.maximum(r, g), b)
    ml = PARAMS["max_level"]
    b_p, g_p, r_p, L_p = (_pyramid(c, ml) for c in (b, g, r, L))

    feats = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(b_p)):
        col = _color_maps(r_p[i], g_p[i], b_p[i], L_p[i])
        for k in range(3):
            feats[k].append(cv2.resize(col[k], (GRAPH_W, GRAPH_H), interpolation=cv2.INTER_CUBIC))
        for o in _orientation_maps(L_p[i]):
            feats[3].append(cv2.resize(o, (GRAPH_W, GRAPH_H), interpolation=cv2.INTER_CUBIC))

    s_act = PARAMS["sigma_frac_act"] * np.mean([GRAPH_W, GRAPH_H])
    s_norm = PARAMS["sigma_frac_norm"] * np.mean([GRAPH_W, GRAPH_H])
    acts = [_activation(m, s_act) for k in range(4) for m in feats[k]]
    norms = [_normalization(m, s_norm) for m in acts]

    master = norms[0]
    for i in range(1, len(norms)):
        master = np.add(norms[i], master)
    return cv2.normalize(master, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def compute(image_bgr_uint8):
    """28x32 float32 saliency in [0, 1] (the port's map before its final upsampling)."""
    return _master_map(image_bgr_uint8)


def compute_full(image_bgr_uint8):
    """Same as the port's gbvs.compute_saliency(): cubic-upsampled to input size, x255."""
    gray = _master_map(image_bgr_uint8)
    h, w = image_bgr_uint8.shape[:2]
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_CUBIC) * 255.0
