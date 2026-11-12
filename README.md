# gazehead: driver-attention benchmark

Code, split lists, decision files and per-video scores needed to replicate the
driver-attention benchmark in *[DISSERTATION TITLE]* (Matthew [SURNAME],
University of Arkansas at Little Rock, [YEAR]). This repository is the deposit
described in Appendix E.

DOI: [DOI pending]

## Contents

| Path | Contents |
|---|---|
| `*.py` | Data loading, backbones, gaze head, training (`train_gaze.py`) and the shared scorer (`evaluate.py`: CC, SIM, KLD, NSS, AUC-Judd) |
| `tools/` | Frame extraction, split and center-bias builders, baseline prediction, cross-dataset transfer, pairwise agreement and the shuffled-frame control |
| `meta/splits/` | Frozen split lists (`split,vid,frame,cam,gt`) for BDD-A and CoCAtt |
| `meta/centerbias/` | Center-bias maps for both datasets |
| `meta/decisions/` | Decision criteria used in the experiments, as written |
| `results/in-domain/`, `results/cross-dataset/`, `results/agreement/` | Per run: `scores_per_video.csv`, `summary.csv` or `pairwise.csv`, `run.json`, to check a replication against |
| `results/training/`, `results/prediction/` | `run.json` (settings) and scores for each gaze-head training run and baseline prediction run |
| `baselines/` | For each baseline: upstream repository, commit used, and patch or wrapper files |
| `requirements.txt` | Pinned environment (Python 3.12.3, PyTorch 2.7.1) |

Not included: dataset frames and gaze maps (obtain BDD-A and CoCAtt from their
authors; the split lists identify every frame used), prediction maps, and other
projects' code or weights (`baselines/` links to each at the commit used).

## Directory layout

The scripts were run on one Linux host with this layout, and their defaults and
docstrings use these paths. Either recreate it (symlinks are enough) or pass the
corresponding command-line options.

| Path | Holds |
|---|---|
| `/opt/gazehead` | This repository |
| `/opt/<Repo>` | Each baseline's clone (`/opt/SCOUT`, `/opt/SaliencyMamba`, `/opt/dinov3`, `/opt/vjepa2`, `/opt/mlnet`, `/opt/u2net`, `/opt/gbvs`) |
| `/srv/datasets/BDDA`, `/srv/datasets/CoCAtt` | Datasets |
| `/srv/models/<model>` | Pretrained weights |
| `/srv/cache` | Frame caches built by `tools/build_frame_cache.py` |
| `/srv/results` | Run outputs; `results/` here mirrors its structure |

In `run.json` these appear as `$DATASETS`, `$MODELS`, `$RESULTS` and `$BASELINES`.

## Replicating the results

The steps follow Appendices A and B.

1. **Environment.**
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. **Datasets.** Download BDD-A and CoCAtt (manual-drive sessions only; the
   authors' processed release) and place them under one root, `$DATASETS`.
   - BDD-A: [URL]
   - CoCAtt: [URL]
3. **Frames and splits.** Extract BDD-A frames with `tools/extract_bdda_frames.sh`.
   The frozen lists in `meta/splits/` are the ones used; `tools/build_bdda_splits.py`
   and `tools/build_cocatt_splits.py` show how they were built.
4. **Train** the gaze heads on the frozen DINOv3 and V-JEPA 2.1 backbones with
   `train_gaze.py` ([ARGS]).
5. **Predict** with each baseline using `baselines/<name>/UPSTREAM.md` and the
   `tools/predict_*.py` scripts.
6. **Score** every prediction set with `evaluate.py` against the center-bias maps
   in `meta/centerbias/`, applying the criteria in `meta/decisions/`.
7. **Cross-dataset and agreement** results: `tools/transfer_gaps.py`,
   `tools/pairwise_agreement.py`, `tools/shuffled_agreement.py`.

Compare your `summary.csv` and `pairwise.csv` with the matching run in `results/`.

## Acknowledgements

The BDD-A test list and exclusion list follow those published with SCOUT
(Kotseruba & Tsotsos, IV 2024), [URL]. [State whether the lists are included
here under SCOUT's license or must be obtained from SCOUT's repository.]

## License

Code: MIT (`LICENSE`). Scores, split lists, center-bias maps and decision files:
CC BY 4.0. Datasets and baseline code and weights remain under their owners' terms.

## Citation

See `CITATION.cff`, or cite the dissertation:

> Matthew [SURNAME]. *[DISSERTATION TITLE]*. PhD dissertation, University of
> Arkansas at Little Rock, [YEAR]. DOI: [DOI pending]
