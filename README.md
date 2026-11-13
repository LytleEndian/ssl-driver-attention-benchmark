# ssl-driver-attention-benchmark

Code, split lists, decision files and per-video scores needed to replicate the
driver-attention benchmark in *Self-Supervised Visual Representations in
Driving Scenes* (Matthew Booker Mason Lytle, PhD dissertation, University of
Arkansas at Little Rock, 2026). This repository is the deposit described in
Appendix E of the dissertation.

The benchmark attaches a small gaze decoder to frozen self-supervised backbones
(DINOv3, V-JEPA 2.1) and compares it with driver-attention models (SCOUT, SalM²)
and saliency baselines (GBVS, ML-Net, U²-Net, center bias) on BDD-A and CoCAtt.

## What is here, and what is not

| Path | Contents |
|---|---|
| `*.py` | Data loading, backbones, gaze head, training (`train_gaze.py`) and the shared scorer (`evaluate.py`: CC, SIM, KLD, NSS, AUC-Judd) |
| `tools/` | Frame extraction, split and center-bias builders, baseline prediction, cross-dataset transfer, pairwise agreement and the shuffled-frame control |
| `meta/splits/` | Frozen split lists (`split,vid,frame,cam,gt`) for BDD-A and CoCAtt |
| `meta/centerbias/` | Center-bias maps for both datasets |
| `meta/decisions/` | Decision criteria used in the experiments, as written |
| `results/in-domain/`, `results/cross-dataset/`, `results/agreement/` | Per run: per-video scores, summary or pairwise tables and `run.json`, to check a replication against |
| `results/training/`, `results/prediction/` | `run.json` (settings) and scores for each gaze-head training run and baseline prediction run |
| `baselines/` | For each upstream project: repository, commit used, and the patch or wrapper files written for this benchmark |
| `weights/` | Trained gaze-decoder heads for the reported DINOv3 ViT-L/16 and V-JEPA 2.1 ViT-L/16 runs on BDD-A and CoCAtt (decoder only, about 0.6M parameters each), with `SHA256SUMS` |
| `COMMANDS.md` | The command for every deposited run, rebuilt from its `run.json` |
| `requirements.txt` | Pinned environment (Python 3.12.3, PyTorch 2.7.1) |

**Not included, by design:** datasets, backbone and other upstream weights,
trained SCOUT and SalM² checkpoints, prediction maps, and any unmodified code
from other projects. The sections below say where to get each of them and
where to put it.

## Datasets

| Dataset | Where to get it | Put it in |
|---|---|---|
| BDD-A (BDD-Attention; Xia et al., ACCV 2018) | http://bdd-data.berkeley.edu/download.html | `/srv/datasets/BDDA` |
| CoCAtt (Shen et al., ITSC 2022) | https://cocatt-dataset.github.io/ (data link on that page) | `/srv/datasets/CoCAtt` |

CoCAtt: the authors' processed release, manual-drive sessions only. The split
lists in `meta/splits/` identify every frame used from both datasets.

## Upstream repositories and weights

Clone each repository at the commit shown, into the folder shown. Then, where
`baselines/<name>/` has a `patch.diff`, apply it in the clone
(`git apply patch.diff`), and where it has a `wrappers/` folder, copy its
contents over the clone. `baselines/<name>/UPSTREAM.md` repeats these details
and adds run notes.

| Project | Repository | Commit | Clone to |
|---|---|---|---|
| SCOUT | https://github.com/ykotseruba/SCOUT | `771100cbda71b2401499933123b92bfd99faadf5` | `/opt/SCOUT` |
| SalM² | https://github.com/zhao-chunyu/SaliencyMamba | `b2812d1a9c7c272f7648fbbd3b5d77c295f990e6` | `/opt/SaliencyMamba` |
| DINOv3 | https://github.com/facebookresearch/dinov3 | `6876159a11b4df116f30f667f8c9888617df0751` | `/opt/dinov3` |
| V-JEPA 2.1 | https://github.com/facebookresearch/vjepa2 | `204698b45b3712590f06245fbfba32d3be539812` | `/opt/vjepa2` |
| ML-Net | https://github.com/marcellacornia/mlnet | `bf6e4a9905915a2bddad2cbe45ac35524fd788c5` | `/opt/mlnet` |
| U²-Net | https://github.com/xuebinqin/U-2-Net | `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29` | `/opt/u2net` |
| GBVS | https://github.com/shreelock/gbvs | `4c7e3c58e786d053ec4ae6fb9d03bdfeea2c539a` | `/opt/gbvs` |

**Upstream weights.** None are included here; get them as follows.
- **SCOUT:** Trained for this benchmark from the configs in wrappers/configs/ (no task input, pretrained backbone as in upstream). Each prediction run.json records the checkpoint used and its sha256.
- **SalM²:** Trained for this benchmark (frames only, frozen CLIP ViT-B/32 image branch). Each prediction run.json records the checkpoint, epoch and sha256.
- **DINOv3:** ViT-L/16 distilled (dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth) and ViT-B/16 distilled (dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth), LVD-1689M, placed in /srv/models/dinov3/. Meta distributes these under the DINOv3 License through an access request form: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ .
- **V-JEPA 2.1:** V-JEPA 2.1 ViT-L/16 checkpoint vjepa2_1_vitl_dist_vitG_384.pt (sha256 7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6) from https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt, placed in /srv/models/vjepa2/.
- **ML-Net:** Authors' SALICON weights (see the upstream README), used zero-shot, stored as /srv/models/mlnet/mlnet_salicon_weights.pkl (sha256 6975e6ef98174b4c0fffd1ce11087d82bed53bdaf6dece8cf172b28c479709de).
- **U²-Net:** u2net.pth (full model, DUTS-TR; download link in the upstream README), used zero-shot, stored as /srv/models/u2net/u2net.pth (sha256 10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58).
- **GBVS:** None needed (no trained weights).

Please cite the upstream projects when you use them; each repository's README
gives its reference.

### Using the deposited decoder heads

Each file in `weights/` holds the decoder only (`decoder`, plus the epoch,
validation scores and arguments of the run that produced it). The frozen
backbone is not in the file: download it as listed above. To predict without
retraining, take that run's command from `COMMANDS.md`, give it a new `--out`,
and add:

```bash
--predict-only --ckpt weights/vjepa21/bdda_vitl16_light_s10_c8s7.pt
```

These are the heads behind the reported DINOv3 and V-JEPA 2.1 numbers.

## Directory layout

The scripts were run on one Linux host with this layout, and their defaults and
docstrings use these paths. Either recreate it (symlinks are enough) or pass the
corresponding command-line options.

| Path | Holds |
|---|---|
| `/opt/gazehead` | This repository (the scripts expect this folder name) |
| `/opt/<Repo>` | Each upstream clone, as listed above |
| `/srv/datasets/BDDA`, `/srv/datasets/CoCAtt` | Datasets |
| `/srv/models/<model>` | Pretrained weights, as listed above |
| `/srv/cache` | Frame and feature caches built by `tools/build_frame_cache.py` and `tools/extract_vjepa_features.py` |
| `/srv/results` | Run outputs; `results/` here mirrors its structure |

```bash
git clone https://github.com/LytleEndian/ssl-driver-attention-benchmark.git /opt/gazehead
```

In `run.json` and `COMMANDS.md` these paths appear as variables:

```bash
export DATASETS=/srv/datasets MODELS=/srv/models CACHE=/srv/cache RESULTS=/srv/results BASELINES=/opt
```

## Replicating the results

The steps follow Appendices A and B of the dissertation. `COMMANDS.md` gives the
exact command for each run.

1. **Environment.**
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
   SCOUT, SalM² and U²-Net run in their own environments; follow each upstream
   README.
2. **Datasets, repositories and weights.** Get them as described above.
3. **Frames and splits.** Extract BDD-A frames with `tools/extract_bdda_frames.sh`.
   The frozen lists in `meta/splits/` are the ones used; `tools/build_bdda_splits.py`
   and `tools/build_cocatt_splits.py` show how they were built.
4. **Train** the gaze heads on the frozen DINOv3 and V-JEPA 2.1 backbones with
   `train_gaze.py`, and train SCOUT and SalM² as described in their `UPSTREAM.md`.
5. **Predict** with each baseline using the `tools/predict_*.py` scripts.
6. **Score** every prediction set with `evaluate.py` against the center-bias maps
   in `meta/centerbias/`, applying the criteria in `meta/decisions/`.
7. **Cross-dataset and agreement** results: `tools/transfer_gaps.py`,
   `tools/pairwise_agreement.py`, `tools/shuffled_agreement.py`.

Compare your result files with the matching run in `results/`. Scoring an
existing prediction set reproduces the deposited tables exactly; retrained
models will give close but not identical numbers.

## Acknowledgements

The BDD-A test list and exclusion list follow those published with SCOUT
(Kotseruba & Tsotsos, IV 2024). They are included in `meta/splits/` (bdda_excluded_videos.csv bdda_scout_test_samples.csv bdda_scout_eval_frames.csv); SCOUT is MIT-licensed and its license is reproduced in `baselines/SCOUT/LICENSE.upstream`.

`tools/gbvs_fast.py` is adapted from the GBVS implementation listed above, and
`mlnet_torch.py` is a PyTorch port of ML-Net. Files under `baselines/*/wrappers/`
and each `patch.diff` were written for this benchmark against the upstream
repositories. Where an upstream license applies to them it is reproduced as
`LICENSE.upstream` beside them; apart from those license notices, no unmodified
upstream file is included.

Built with DINOv3. The DINOv3 backbone is used under the DINOv3 License, a copy
of which is in `baselines/DINOv3/LICENSE.upstream`; V-JEPA 2.1 is used under
its MIT license.

## License

- Code: MIT (`LICENSE`).
- Scores, split lists, center-bias maps and decision files: CC BY 4.0.
- Decoder weights in `weights/vjepa21/`: CC BY 4.0.
- Decoder weights in `weights/dinov3/`: these were trained on DINOv3 features.
  They contain no DINOv3 weights or code, but to the extent they are a
  derivative work of the DINO Materials they are provided under the terms of
  the DINOv3 License (`baselines/DINOv3/LICENSE.upstream`), not CC BY 4.0.
- Datasets and upstream code and weights remain under their owners' terms.

## Citation

If you use this code or these scores, please cite the dissertation:

```bibtex
@phdthesis{lytle2026svr,
  author  = {Lytle, Matthew Booker Mason},
  title   = {Self-Supervised Visual Representations in Driving Scenes},
  school  = {University of Arkansas at Little Rock},
  address = {Little Rock, AR},
  year    = {2026},
  month   = dec,
  type    = {{PhD} dissertation}
}
```

To cite the code and scores themselves:

```bibtex
@software{lytle2026sdab,
  author  = {Lytle, Matthew Booker Mason},
  title   = {ssl-driver-attention-benchmark: Code, Split Lists and Scores
             for a Self-Supervised Driver-Attention Benchmark},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/LytleEndian/ssl-driver-attention-benchmark}
}
```

`CITATION.cff` carries the same information for GitHub's "Cite this repository".
