# SCOUT

- Upstream: https://github.com/ykotseruba/SCOUT.git
- Commit: `771100cbda71b2401499933123b92bfd99faadf5`
- Changes to upstream files: `patch.diff` (`git apply patch.diff` in the clone)
- Files added for this benchmark: `wrappers/`
- Notes: Pretrained Video Swin backbone: run pretrained_weights/download_weights.sh in the SCOUT clone. Apply patch.diff and copy wrappers/ into the clone, then train with upstream's command, python3 train.py --config configs/bdda_wo_task.yaml --save_dir <run dir> (configs/cocatt_wo_task.yaml for CoCAtt). Predictions are written by tools/predict_scout.py and tools/predict_scout_cocatt.py in this repository, not by SCOUT's test.py.
- Upstream license: `LICENSE.upstream` (applies to `patch.diff` and files adapted from upstream)
- Weights: Trained for this benchmark from the configs in wrappers/configs/ (no task input, pretrained backbone as in upstream). Each prediction run.json records the checkpoint used and its sha256.
