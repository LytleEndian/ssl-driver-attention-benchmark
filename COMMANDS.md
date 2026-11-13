# Commands

One command per deposited run, rebuilt from the arguments recorded in its
`run.json` (false and null arguments are omitted). Run them from the
repository root, in this order: training, prediction, then scoring.

```bash
export DATASETS=/srv/datasets MODELS=/srv/models CACHE=/srv/cache RESULTS=/srv/results BASELINES=/opt
```

Arguments such as `--gpu` and `--workers` reflect the original host; adjust
them. Flags were checked against each script's `--help` unless noted.

**Manifest names.** Earlier BDD-A runs refer to manifests without the
`bdda_` prefix. These are byte-identical files, so use the prefixed one in
`meta/splits/`: `excluded_videos` = `bdda_excluded_videos`, `scout_eval_frames` = `bdda_scout_eval_frames`, `scout_test_samples` = `bdda_scout_test_samples`, `test_common` = `bdda_test_common`, `test_salm2_own` = `bdda_test_salm2_own`, `test_scored` = `bdda_test_scored`, `train` = `bdda_train`, `val` = `bdda_val`.

## training

### `results/training/dinov3/bdda_vitl16_light_s10`

```bash
python train_gaze.py \
    --hub dinov3_vitl16 \
    --weights "$MODELS/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --repo "$BASELINES/dinov3" \
    --root "$DATASETS/BDDA" \
    --splits meta/splits \
    --test-manifests test_common \
    --out "$RESULTS/dinov3/bdda_vitl16_light_s10" \
    --input 288 512 \
    --target 72 128 \
    --pred-size 72 128 \
    --width 128 \
    --epochs 30 \
    --bs 32 \
    --lr 0.001 \
    --wd 0.0001 \
    --warmup 500 \
    --stride 10 \
    --val-stride 5 \
    --workers 4 \
    --gpu 1 \
    --seed 0 \
    --max-steps 0 \
    --log-every 100 \
    --cache "$CACHE/bdda_288x512"
```

### `results/training/dinov3/cocatt_vitl16_light_s2`

```bash
python train_gaze.py \
    --hub dinov3_vitl16 \
    --weights "$MODELS/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    --repo "$BASELINES/dinov3" \
    --root "$DATASETS/CoCAtt" \
    --splits meta/splits \
    --train-manifest cocatt_train \
    --val-manifest cocatt_val \
    --test-manifests cocatt_test \
    --out "$RESULTS/dinov3/cocatt_vitl16_light_s2" \
    --input 256 512 \
    --target 64 128 \
    --pred-size 64 128 \
    --width 128 \
    --epochs 30 \
    --bs 32 \
    --lr 0.001 \
    --wd 0.0001 \
    --warmup 500 \
    --stride 2 \
    --val-stride 2 \
    --workers 5 \
    --gpu 1 \
    --seed 0 \
    --max-steps 0 \
    --log-every 100 \
    --cache "$CACHE/cocatt_256x512_all" \
    --backbone dinov3 \
    --vjepa-arch vit_large \
    --feat-norm none \
    --clip-frames 0 \
    --clip-step 0
```

### `results/training/vjepa21/bdda_vitl16_light_s10_c16s7`

```bash
python train_gaze.py \
    --hub dinov3_vitb16 \
    --weights "$MODELS/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt" \
    --repo "$BASELINES/dinov3" \
    --root "$DATASETS/BDDA" \
    --splits meta/splits \
    --test-manifests test_common \
    --out "$RESULTS/vjepa21/bdda_vitl16_light_s10_c16s7" \
    --input 288 512 \
    --target 72 128 \
    --pred-size 72 128 \
    --width 128 \
    --epochs 30 \
    --bs 32 \
    --lr 0.001 \
    --wd 0.0001 \
    --warmup 500 \
    --stride 10 \
    --val-stride 5 \
    --workers 6 \
    --gpu 1 \
    --seed 0 \
    --max-steps 0 \
    --log-every 100 \
    --no-predict \
    --cache "$CACHE/bdda_288x512" \
    --backbone vjepa21 \
    --vjepa-arch vit_large \
    --feat-norm none \
    --clip-frames 16 \
    --clip-step 7 \
    --train-features "$CACHE/vjepa21L_bdda_c16s7/train_s10" \
    --val-features "$CACHE/vjepa21L_bdda_c16s7/val_s5"
```

### `results/training/vjepa21/bdda_vitl16_light_s10_c8s7`

```bash
python train_gaze.py \
    --hub dinov3_vitb16 \
    --weights "$MODELS/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt" \
    --repo "$BASELINES/dinov3" \
    --root "$DATASETS/BDDA" \
    --splits meta/splits \
    --test-manifests test_common \
    --out "$RESULTS/vjepa21/bdda_vitl16_light_s10_c8s7" \
    --input 288 512 \
    --target 72 128 \
    --pred-size 72 128 \
    --width 128 \
    --epochs 30 \
    --bs 32 \
    --lr 0.001 \
    --wd 0.0001 \
    --warmup 500 \
    --stride 10 \
    --val-stride 5 \
    --workers 6 \
    --gpu 0 \
    --seed 0 \
    --max-steps 0 \
    --log-every 100 \
    --no-predict \
    --cache "$CACHE/bdda_288x512" \
    --backbone vjepa21 \
    --vjepa-arch vit_large \
    --feat-norm none \
    --clip-frames 8 \
    --clip-step 7 \
    --train-features "$CACHE/vjepa21L_bdda_c8s7/train_s10" \
    --val-features "$CACHE/vjepa21L_bdda_c8s7/val_s5"
```

### `results/training/vjepa21/cocatt_vitl16_light_s2_c8s1`

```bash
python train_gaze.py \
    --hub dinov3_vitb16 \
    --weights "$MODELS/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt" \
    --repo "$BASELINES/dinov3" \
    --root "$DATASETS/CoCAtt" \
    --splits meta/splits \
    --train-manifest cocatt_train \
    --val-manifest cocatt_val \
    --test-manifests cocatt_test \
    --out "$RESULTS/vjepa21/cocatt_vitl16_light_s2_c8s1" \
    --input 256 512 \
    --target 64 128 \
    --pred-size 64 128 \
    --width 128 \
    --epochs 30 \
    --bs 32 \
    --lr 0.001 \
    --wd 0.0001 \
    --warmup 500 \
    --stride 2 \
    --val-stride 2 \
    --workers 5 \
    --gpu 0 \
    --seed 0 \
    --max-steps 0 \
    --log-every 100 \
    --cache "$CACHE/cocatt_256x512_all" \
    --backbone vjepa21 \
    --vjepa-arch vit_large \
    --feat-norm none \
    --clip-frames 8 \
    --clip-step 1 \
    --train-features "$CACHE/vjepa21L_cocatt_c8s1/train_s2" \
    --val-features "$CACHE/vjepa21L_cocatt_c8s1/val_s2"
```

## prediction

### `results/prediction/gbvs/bdda_v1`

```bash
python tools/predict_gbvs.py \
    --manifest ./meta/splits/test_common.csv \
    --root "$DATASETS/BDDA" \
    --out "$RESULTS/gbvs/bdda_v1" \
    --workers 19 \
    --port-dir "$BASELINES/gbvs" \
    --limit-videos 0
```

### `results/prediction/gbvs/cocatt_v1`

```bash
python tools/predict_gbvs.py \
    --manifest ./meta/splits/cocatt_test.csv \
    --root "$DATASETS/CoCAtt" \
    --out "$RESULTS/gbvs/cocatt_v1" \
    --workers 19 \
    --port-dir "$BASELINES/gbvs" \
    --limit-videos 0
```

### `results/prediction/mlnet/bdda_v1`

Produced by `tools/predict_mlnet.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

```text
usage: predict_mlnet.py [-h] --manifest MANIFEST --root ROOT --out OUT
                        [--weights WEIGHTS] --kernel-flip {yes,no} --pool
                        {theano,standard} [--gpu GPU] [--bs BS]
                        [--workers WORKERS] [--max-videos MAX_VIDEOS]
                        [--check-frames CHECK_FRAMES] [--check-dir CHECK_DIR]
```

### `results/prediction/mlnet/cocatt_v1`

Produced by `tools/predict_mlnet.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

Usage of `tools/predict_mlnet.py` is shown above.

### `results/prediction/salm2/bdda_20261028/pred_bdda_test_scored`

Produced by `tools/predict_salm2.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

```text
usage: predict_salm2.py [-h] --ckpt CKPT --json-root JSON_ROOT --out OUT
                        [--gpu GPU] [--batch BATCH]
```

### `results/prediction/salm2/bdda_20261028/pred_cocatt_test`

Produced by `tools/predict_salm2_cocatt.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

```text
usage: predict_salm2_cocatt.py [-h] --ckpt CKPT --out OUT [--gpu GPU]
                               [--batch BATCH]
```

### `results/prediction/salm2/cocatt_20261027/pred_bdda_test_scored`

Produced by `tools/predict_salm2.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

Usage of `tools/predict_salm2.py` is shown above.

### `results/prediction/salm2/cocatt_20261027/pred_cocatt_test`

Produced by `tools/predict_salm2_cocatt.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

Usage of `tools/predict_salm2_cocatt.py` is shown above.

### `results/prediction/scout/bdda_wo_task_20261028/pred_bdda_test_scored`

Produced by `tools/predict_scout.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

```text
usage: predict_scout.py [-h] --run RUN --dataset {bdda,cocatt} [--out OUT]
                        [--gpu GPU] [--batch BATCH] [--workers WORKERS]
```

### `results/prediction/scout/bdda_wo_task_20261028/pred_cocatt_test`

Produced by `tools/predict_scout_cocatt.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

```text
usage: predict_scout_cocatt.py [-h] --run RUN [--out OUT] [--gpu GPU]
                               [--batch BATCH] [--workers WORKERS]
```

### `results/prediction/scout/cocatt_wo_task_20261027/pred_bdda_test_scored`

Produced by `tools/predict_scout.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

Usage of `tools/predict_scout.py` is shown above.

### `results/prediction/scout/cocatt_wo_task_20261027/pred_cocatt_test`

Produced by `tools/predict_scout_cocatt.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

Usage of `tools/predict_scout_cocatt.py` is shown above.

### `results/prediction/u2net/bdda_v1`

Produced by `tools/predict_u2net.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

```text
usage: predict_u2net.py [-h] --manifest MANIFEST --root ROOT --out OUT
                        [--weights WEIGHTS] [--repo REPO] [--size SIZE]
                        [--gpu GPU] [--bs BS] [--workers WORKERS]
                        [--max-videos MAX_VIDEOS]
                        [--check-frames CHECK_FRAMES] [--check-dir CHECK_DIR]
```

### `results/prediction/u2net/cocatt_v1`

Produced by `tools/predict_u2net.py`. Its command line was not recorded; the settings that determine the output (weights and their hash, manifest, input and output handling) are in this run's `run.json`.

Usage of `tools/predict_u2net.py` is shown above.

## in-domain

### `results/in-domain/eval/bdda_test_gbvs_v1`

```bash
python evaluate.py \
    --manifest meta/splits/test_scored.csv \
    --root "$DATASETS/BDDA" \
    --model "center_bias=$RESULTS/baselines/bdda_center_bias_72x128.npy" \
    --model "gbvs=$RESULTS/gbvs/bdda_v1/pred_test_common" \
    --model "dinov3_l=$RESULTS/dinov3/bdda_vitl16_light_s10/pred_test_common" \
    --out "$RESULTS/eval/bdda_test_gbvs_v1" \
    --gpu 0 \
    --workers 6 \
    --chunk 48 \
    --n-boot 10000 \
    --seed 0 \
    --limit-videos 0
```

### `results/in-domain/eval/bdda_test_v4`

```bash
python evaluate.py \
    --manifest meta/splits/bdda_test_scored.csv \
    --root "$DATASETS/BDDA" \
    --model "center=$RESULTS/baselines/bdda_center_bias_72x128.npy" \
    --model "gbvs=$RESULTS/gbvs/bdda_v1/pred_test_common" \
    --model "mlnet=$RESULTS/mlnet/bdda_v1/pred_test_common" \
    --model "u2net=$RESULTS/u2net/bdda_v1/pred_test_common" \
    --model "dinov3_l=$RESULTS/dinov3/bdda_vitl16_light_s10/pred_test_common" \
    --model "vjepa21_l=$RESULTS/vjepa21/bdda_vitl16_light_s10_c8s7/pred_test_common" \
    --model "scout_wo_task=$RESULTS/scout/bdda_wo_task_20261028/pred_bdda_test_scored" \
    --model "salm2=$RESULTS/salm2/bdda_20261028/pred_bdda_test_scored" \
    --out "$RESULTS/eval/bdda_test_v4" \
    --gpu 1 \
    --workers 6 \
    --chunk 48 \
    --n-boot 10000 \
    --seed 0 \
    --limit-videos 0
```

### `results/in-domain/eval/cocatt_test_v4`

```bash
python evaluate.py \
    --manifest meta/splits/cocatt_test.csv \
    --root "$DATASETS/CoCAtt" \
    --model "center=$RESULTS/baselines/cocatt_center_bias_64x128.npy" \
    --model "gbvs=$RESULTS/gbvs/cocatt_v1/pred_cocatt_test" \
    --model "mlnet=$RESULTS/mlnet/cocatt_v1/pred_test_common" \
    --model "u2net=$RESULTS/u2net/cocatt_v1/pred_test_common" \
    --model "dinov3_l=$RESULTS/dinov3/cocatt_vitl16_light_s2/pred_cocatt_test" \
    --model "vjepa21_l=$RESULTS/vjepa21/cocatt_vitl16_light_s2_c8s1/pred_cocatt_test" \
    --model "scout_wo_task=$RESULTS/scout/cocatt_wo_task_20261027/pred_cocatt_test" \
    --model "salm2=$RESULTS/salm2/cocatt_20261027/pred_cocatt_test" \
    --out "$RESULTS/eval/cocatt_test_v4" \
    --gpu 1 \
    --workers 6 \
    --chunk 48 \
    --n-boot 10000 \
    --seed 0 \
    --limit-videos 0
```

## cross-dataset

### `results/cross-dataset/eval/bdda_crossdataset_v1`

```bash
python evaluate.py \
    --manifest meta/splits/bdda_test_scored.csv \
    --root "$DATASETS/BDDA" \
    --model "center_bdda=$RESULTS/baselines/bdda_center_bias_72x128.npy" \
    --model "center_cocatt=$RESULTS/baselines/cocatt_center_bias_64x128.npy" \
    --model "dinov3_l_bdda=$RESULTS/dinov3/bdda_vitl16_light_s10/pred_test_common" \
    --model "dinov3_l_cocatt=$RESULTS/dinov3/cocatt_vitl16_light_s2_on_bdda/pred_bdda_test_scored" \
    --model "vjepa21_l_bdda=$RESULTS/vjepa21/bdda_vitl16_light_s10_c8s7/pred_test_common" \
    --model "vjepa21_l_cocatt=$RESULTS/vjepa21/cocatt_vitl16_light_s2_c8s1_on_bdda/pred_bdda_test_scored" \
    --model "scout_wo_task_bdda=$RESULTS/scout/bdda_wo_task_20261028/pred_bdda_test_scored" \
    --model "scout_wo_task_cocatt=$RESULTS/scout/cocatt_wo_task_20261027/pred_bdda_test_scored" \
    --model "salm2_bdda=$RESULTS/salm2/bdda_20261028/pred_bdda_test_scored" \
    --model "salm2_cocatt=$RESULTS/salm2/cocatt_20261027/pred_bdda_test_scored" \
    --out "$RESULTS/eval/bdda_crossdataset_v1" \
    --gpu 1 \
    --workers 6 \
    --chunk 48 \
    --n-boot 10000 \
    --seed 0 \
    --limit-videos 0
```

### `results/cross-dataset/eval/cocatt_crossdataset_v1`

```bash
python evaluate.py \
    --manifest meta/splits/cocatt_test.csv \
    --root "$DATASETS/CoCAtt" \
    --model "center_cocatt=$RESULTS/baselines/cocatt_center_bias_64x128.npy" \
    --model "center_bdda=$RESULTS/baselines/bdda_center_bias_72x128.npy" \
    --model "dinov3_l_cocatt=$RESULTS/dinov3/cocatt_vitl16_light_s2/pred_cocatt_test" \
    --model "dinov3_l_bdda=$RESULTS/dinov3/bdda_vitl16_light_s10_on_cocatt/pred_cocatt_test" \
    --model "vjepa21_l_cocatt=$RESULTS/vjepa21/cocatt_vitl16_light_s2_c8s1/pred_cocatt_test" \
    --model "vjepa21_l_bdda=$RESULTS/vjepa21/bdda_vitl16_light_s10_c8s7_on_cocatt/pred_cocatt_test" \
    --model "scout_wo_task_cocatt=$RESULTS/scout/cocatt_wo_task_20261027/pred_cocatt_test" \
    --model "scout_wo_task_bdda=$RESULTS/scout/bdda_wo_task_20261028/pred_cocatt_test" \
    --model "salm2_cocatt=$RESULTS/salm2/cocatt_20261027/pred_cocatt_test" \
    --model "salm2_bdda=$RESULTS/salm2/bdda_20261028/pred_cocatt_test" \
    --out "$RESULTS/eval/cocatt_crossdataset_v1" \
    --gpu 0 \
    --workers 6 \
    --chunk 48 \
    --n-boot 10000 \
    --seed 0 \
    --limit-videos 0
```

### `results/cross-dataset/eval/transfer_gaps_v2`

```bash
python tools/transfer_gaps.py \
    --run "bdda=$RESULTS/eval/bdda_crossdataset_v1" \
    --run "cocatt=$RESULTS/eval/cocatt_crossdataset_v1" \
    --protocol scout \
    --n-boot 10000 \
    --seed 0 \
    --out "$RESULTS/eval/transfer_gaps_v2"
```

## agreement

### `results/agreement/eval/bdda_pairwise_v2`

```bash
python tools/pairwise_agreement.py \
    --manifest meta/splits/bdda_test_scored.csv \
    --root "$DATASETS/BDDA" \
    --model "center_bdda=$RESULTS/baselines/bdda_center_bias_72x128.npy" "center_cocatt=$RESULTS/baselines/cocatt_center_bias_64x128.npy" "gbvs=$RESULTS/gbvs/bdda_v1/pred_test_common" "mlnet=$RESULTS/mlnet/bdda_v1/pred_test_common" "u2net=$RESULTS/u2net/bdda_v1/pred_test_common" "dinov3_l_bdda=$RESULTS/dinov3/bdda_vitl16_light_s10/pred_test_common" "dinov3_l_cocatt=$RESULTS/dinov3/cocatt_vitl16_light_s2_on_bdda/pred_bdda_test_scored" "vjepa21_l_bdda=$RESULTS/vjepa21/bdda_vitl16_light_s10_c8s7/pred_test_common" "vjepa21_l_cocatt=$RESULTS/vjepa21/cocatt_vitl16_light_s2_c8s1_on_bdda/pred_bdda_test_scored" "scout_wo_task_bdda=$RESULTS/scout/bdda_wo_task_20261028/pred_bdda_test_scored" "scout_wo_task_cocatt=$RESULTS/scout/cocatt_wo_task_20261027/pred_bdda_test_scored" "salm2_bdda=$RESULTS/salm2/bdda_20261028/pred_bdda_test_scored" "salm2_cocatt=$RESULTS/salm2/cocatt_20261027/pred_bdda_test_scored" \
    --center center_bdda \
    --size 72 128 \
    --out "$RESULTS/eval/bdda_pairwise_v2" \
    --gpu 1 \
    --workers 6 \
    --chunk 512 \
    --n-boot 10000 \
    --seed 0 \
    --limit-videos 0
```

### `results/agreement/eval/bdda_shuffle_v1`

```bash
python tools/shuffled_agreement.py \
    --from-run "$RESULTS/eval/bdda_pairwise_v2/run.json" \
    --repo . \
    --exclude  \
    --n-derange 5 \
    --frame-stride 1 \
    --reference "$RESULTS/eval/bdda_pairwise_v2/agreement_per_video.csv" \
    --limit-videos 0 \
    --gpu 1 \
    --workers 6 \
    --chunk 512 \
    --n-boot 10000 \
    --seed 0 \
    --out "$RESULTS/eval/bdda_shuffle_v1"
```

### `results/agreement/eval/cocatt_pairwise_v2`

```bash
python tools/pairwise_agreement.py \
    --manifest meta/splits/cocatt_test.csv \
    --root "$DATASETS/CoCAtt" \
    --model "center_cocatt=$RESULTS/baselines/cocatt_center_bias_64x128.npy" "center_bdda=$RESULTS/baselines/bdda_center_bias_72x128.npy" "gbvs=$RESULTS/gbvs/cocatt_v1/pred_cocatt_test" "mlnet=$RESULTS/mlnet/cocatt_v1/pred_test_common" "u2net=$RESULTS/u2net/cocatt_v1/pred_test_common" "dinov3_l_cocatt=$RESULTS/dinov3/cocatt_vitl16_light_s2/pred_cocatt_test" "dinov3_l_bdda=$RESULTS/dinov3/bdda_vitl16_light_s10_on_cocatt/pred_cocatt_test" "vjepa21_l_cocatt=$RESULTS/vjepa21/cocatt_vitl16_light_s2_c8s1/pred_cocatt_test" "vjepa21_l_bdda=$RESULTS/vjepa21/bdda_vitl16_light_s10_c8s7_on_cocatt/pred_cocatt_test" "scout_wo_task_cocatt=$RESULTS/scout/cocatt_wo_task_20261027/pred_cocatt_test" "scout_wo_task_bdda=$RESULTS/scout/bdda_wo_task_20261028/pred_cocatt_test" "salm2_cocatt=$RESULTS/salm2/cocatt_20261027/pred_cocatt_test" "salm2_bdda=$RESULTS/salm2/bdda_20261028/pred_cocatt_test" \
    --center center_cocatt \
    --size 64 128 \
    --out "$RESULTS/eval/cocatt_pairwise_v2" \
    --gpu 0 \
    --workers 6 \
    --chunk 512 \
    --n-boot 10000 \
    --seed 0 \
    --limit-videos 0
```

### `results/agreement/eval/cocatt_shuffle_v1`

```bash
python tools/shuffled_agreement.py \
    --from-run "$RESULTS/eval/cocatt_pairwise_v2/run.json" \
    --repo . \
    --exclude  \
    --n-derange 5 \
    --frame-stride 1 \
    --group-by-subject \
    --reference "$RESULTS/eval/cocatt_pairwise_v2/agreement_per_video.csv" \
    --limit-videos 0 \
    --gpu 0 \
    --workers 2 \
    --chunk 512 \
    --n-boot 10000 \
    --seed 0 \
    --out "$RESULTS/eval/cocatt_shuffle_v1"
```
