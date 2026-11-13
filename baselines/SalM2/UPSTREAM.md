# SalM2

- Upstream: https://github.com/zhao-chunyu/SaliencyMamba.git
- Commit: `b2812d1a9c7c272f7648fbbd3b5d77c295f990e6`
- Changes to upstream files: `patch.diff` (`git apply patch.diff` in the clone)
- Files added for this benchmark: `wrappers/`
- Notes: Install as in upstream's deployment notes (requirements.txt, then pip install -e utils/models/causal-conv1d and utils/models/mamba). Apply patch.diff and copy wrappers/ into the clone. Build the JSON lists with tools/build_salm2_json.py or tools/build_salm2_cocatt_json.py, train with upstream's train.py (--network salmm), and predict with wrappers/salmm_predict.py through tools/predict_salm2.py or tools/predict_salm2_cocatt.py.
- Upstream license: `LICENSE.upstream` (applies to `patch.diff` and files adapted from upstream)
- Weights: Trained for this benchmark (frames only, frozen CLIP ViT-B/32 image branch). Checkpoints are not included; each prediction run.json records the checkpoint, epoch and sha256.
