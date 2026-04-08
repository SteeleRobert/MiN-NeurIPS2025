# CLAUDE.md — MiN-NeurIPS2025

Mixture of Noise (MiN) implementation for NeurIPS 2025. This directory is used to benchmark MiN against the Bayesian streaming classifiers in `~/qz-compcont-learning/` on the same 7 EASE benchmarks.

## Setup

```bash
cd ~/MiN-NeurIPS2025
uv sync          # installs torch==2.0.0+cu118, Python 3.11
source .venv/bin/activate
```

## Running Experiments

### Single benchmark

```bash
cd ~/MiN-NeurIPS2025
python MiN/main.py \
  --base_configs  MiN/configs/base_configs/<base>.json \
  --model_configs MiN/configs/model_configs/<model>.json
```

### All EASE benchmarks (matches qz-compcont-learning protocols)

```bash
bash run_benchmarks.sh
# GPU override:
GPU0=0 GPU1=1 bash run_benchmarks.sh
```

`run_benchmarks.sh` runs all 7 benchmarks in parallel pairs, skipping any whose data is missing. It calls `prepare_data.py` automatically if the ImageNet-R/A train-test splits don't exist yet.

### Running with DINOv2 or DINOv3 backbones

Set the `BACKBONE` env var to swap the backbone for all benchmarks in one invocation:

```bash
# DINOv2 ViT-B/14 — downloads automatically via torch.hub, no auth needed
BACKBONE=dinov2_vitb14 bash run_benchmarks.sh

# DINOv3 ViT-B/16 — requires HuggingFace authentication (see Backbone Model Locations)
BACKBONE=dinov3_vitb16 bash run_benchmarks.sh

# Long SSH session:
BACKBONE=dinov2_vitb14 nohup bash run_benchmarks.sh > ~/min_dinov2_bench.log 2>&1 &
strings ~/min_dinov2_bench.log | grep FINAL
```

To run a single benchmark with a DINO backbone, pass `backbone_type` directly in the model config or use a backbone-specific config:

```bash
# Using a backbone-specific model config
python MiN/main.py \
  --base_configs  MiN/configs/base_configs/cifar224.json \
  --model_configs MiN/configs/model_configs/MiN-cifar-10steps-dinov2vitb14.json

# Using the cifar_ease base config (as run_benchmarks.sh does)
python MiN/main.py \
  --base_configs  MiN/configs/base_configs/cifar_ease.json \
  --model_configs MiN/configs/model_configs/MiN-cifar-10steps-dinov2vitb14.json
```

**Supported `BACKBONE` values for `run_benchmarks.sh`:**

| `BACKBONE` value                    | Source      | Auth needed |
| ----------------------------------- | ----------- | ----------- |
| `pretrained_vit_b16_224_in21k_min`  | timm (default) | No       |
| `dinov2_vitb14`                     | torch.hub   | No          |
| `dinov2_vits14`                     | torch.hub   | No          |
| `dinov2_vitb14_reg`                 | torch.hub   | No          |
| `dinov3_vitb16`                     | HuggingFace | Yes         |
| `dinov3_vits16`                     | HuggingFace | Yes         |
| `dinov3_vitl16`                     | HuggingFace | Yes         |
| `dinov3_vith16`                     | HuggingFace | Yes         |
| `dinov3_vit7b16`                    | HuggingFace | Yes         |

Log files are tagged with the backbone name (e.g. `cifar100_ease_20T_dinov2_vitb14.log`) so runs with different backbones don't overwrite each other. Results are saved to `~/qz-compcont-learning/results/<benchmark>/MiN_<backbone_type>.json`.

### Config pairs per benchmark

Base configs are the same for all backbones. Model configs follow a naming convention: `<base-config>[-<backbone-suffix>].json`. The `BACKBONE` env var selects the right suffix automatically.

| Benchmark     | Base config          | Model config (original)  | Model config (DINOv2-B/14)          | Model config (DINOv3-B/16)          | Protocol       |
| ------------- | -------------------- | ------------------------ | ----------------------------------- | ----------------------------------- | -------------- |
| CIFAR-100     | `cifar_ease`         | `MiN-cifar-10steps`      | `MiN-cifar-10steps-dinov2vitb14`    | `MiN-cifar-10steps-dinov3vitb16`    | B0-Inc5 (20T)  |
| ImageNet-R    | `imagenetr_ease`     | `MiN-inr-10steps`        | `MiN-inr-10steps-dinov2vitb14`      | `MiN-inr-10steps-dinov3vitb16`      | B0-Inc10 (20T) |
| ImageNet-A    | `imageneta_ease`     | `MiN-imageneta`          | `MiN-imageneta-dinov2vitb14`        | `MiN-imageneta-dinov3vitb16`        | B0-Inc20 (10T) |
| CUB-200       | `cub_ease`           | `MiN-cub-10steps`        | `MiN-cub-10steps-dinov2vitb14`      | `MiN-cub-10steps-dinov3vitb16`      | B0-Inc10 (20T) |
| OmniBenchmark | `omnibenchmark_ease` | `MiN-omni-10steps`       | `MiN-omni-10steps-dinov2vitb14`     | `MiN-omni-10steps-dinov3vitb16`     | B0-Inc30 (10T) |
| VTAB          | `vtab_ease`          | `MiN-vtab-5steps`        | `MiN-vtab-5steps-dinov2vitb14`      | `MiN-vtab-5steps-dinov3vitb16`      | B0-Inc10 (5T)  |
| ObjectNet     | `objectnet_ease`     | `MiN-inr-10steps`        | `MiN-inr-10steps-dinov2vitb14`      | `MiN-inr-10steps-dinov3vitb16`      | B0-Inc10 (20T) |


## Data

All benchmarks share data with `~/qz-compcont-learning/data/` (base configs set `data_root` to that path).

**ImageNet-R and ImageNet-A** are stored flat in `qz-compcont-learning/data/` (no train/test split). MiN requires separate `train/` and `test/` dirs, so `prepare_data.py` creates symlink-based 80/20 splits:

```bash
python prepare_data.py --data-root ~/qz-compcont-learning/data
# Creates: ~/qz-compcont-learning/data/imagenet-r-split/{train,test}/
#          ~/qz-compcont-learning/data/imagenet-a-split/{train,test}/
```

CIFAR-100 auto-downloads. CUB-200, OmniBenchmark, VTAB, and ObjectNet require manual download — see `~/qz-compcont-learning/docs/running-experiments.md` for instructions.

## Output and Metrics

After each run, results are saved to `~/qz-compcont-learning/results/<benchmark>/MiN_<backbone>.json` in the same format used by `compcont-bench`, so results are directly comparable via `compcont-bench results`.

Metrics match `~/qz-compcont-learning/src/kanerva_sdm/eval/metrics.py` exactly:


| Metric                  | Key           | Description                                                             |
| ----------------------- | ------------- | ----------------------------------------------------------------------- |
| Ā (Average incremental) | `avg_inc_acc` | Mean of per-step overall accuracy. EASE headline metric.                |
| A_B (Final)             | `avg_acc`     | Accuracy on all classes after the last task.                            |
| BWT                     | `bwt`         | Backward transfer: avg per-task accuracy drop from when learned to end. |
| T0 Final                | `task0_final` | Task 0 accuracy after all tasks. Measures forgetting of first task.     |


`FINAL:` lines are printed to stdout at the end of each run for quick inspection:

```
FINAL: avg_acc=XX.X%  avg_inc_acc=XX.X%  bwt=+X.XXX  task0=XX.X%
```

Logs go to `MiN/logs/<dataset>/<model>/<init>_<increment>/<timestamp>/`.

## Architecture

```
MiN/
  main.py              # entry point — merges base + model configs, calls train()
  trainer/
    BaseTrainer.py     # training loop, _compute_cil_metrics(), _save_compcont_results()
  data_process/
    data_manger.py     # DataManger: task splits, DataLoader construction
    data.py            # Dataset classes (iCIFAR224, iImageNet_R, iImageNet_A, ...)
  models/              # MiN model implementation
  backbones/           # ViT-B/16 backbone (pretrained_vit_b16_224_in21k_min)
  configs/
    base_configs/      # Per-benchmark settings (dataset, init_class, increment, data_root)
    model_configs/     # Per-model hyperparameters (epochs, lr, buffer_size, etc.)
prepare_data.py        # Creates train/test symlink splits for ImageNet-R/A
run_benchmarks.sh      # Runs all 7 EASE benchmarks, saves results to qz-compcont-learning
```

The training loop in `BaseTrainer._train()` collects `all_task_accy` after each task stage into `task_accs_history` (a list of `{task_id: accuracy}` dicts), then passes it to `_compute_cil_metrics()` and `_save_compcont_results()`.

## Backbone Model Locations

### DINOv2 (torch.hub)

Auto-downloaded to `~/.cache/torch/hub/facebookresearch_dinov2_main/` on first use. No authentication needed.

### DINOv3 (HuggingFace)

Auto-downloaded to `~/.cache/huggingface/hub/` on first use. Requires:

1. HuggingFace account
2. License acceptance per model (e.g., [https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m))
3. Authentication: `python -c "from huggingface_hub import login; login(token='TOKEN')"`

1 (open_clip)

Auto-downloaded on first use. No authentication needed.

## GPU Memory Requirements (Frozen Inference)


| Model         | Params | ~VRAM (batch=64)             |
| ------------- | ------ | ---------------------------- |
| DINOv2-S/14   | 22M    | 1-2 GB                       |
| DINOv2-B/14   | 86M    | 2-3 GB                       |
| DINOv3-S/16   | 21M    | 1-2 GB                       |
| DINOv3-B/16   | 86M    | 2-3 GB                       |
| DINOv3-L/16   | 300M   | 4-6 GB                       |
| DINOv3-H+/16  | 840M   | 8-12 GB                      |
| DINOv3-7B/16  | 6.7B   | 14-16 GB (won't fit on 3090) |
| CLIP ViT-B/16 | 86M    | 2-3 GB                       |
| CLIP ViT-L/14 | 304M   | 4-6 GB                       |


## Comparison Method Codebases


| Method             | Repo                                                                  | Framework | Notes                     |
| ------------------ | --------------------------------------------------------------------- | --------- | ------------------------- |
| EASE (CVPR 2024)   | [sun-hailong/CVPR24-Ease](https://github.com/sun-hailong/CVPR24-Ease) | PyTorch   | Primary baseline          |
| MiN (NeurIPS 2025) | [ASCIIJK/MiN-NeurIPS2025](https://github.com/ASCIIJK/MiN-NeurIPS2025) | PyTorch   | Strongest competitor      |
| MOS (AAAI 2025)    | [sun-hailong/AAAI25-MOS](https://github.com/sun-hailong/AAAI25-MOS)   | PyTorch   | Same author as EASE       |
| TOSCA (arXiv 2025) | No public code                                                        | --        | Use reported numbers only |


EASE and MOS share the same first author and likely share codebase structure.

## Infrastructure

### OLab3 (`nyu-gpu`)

- **SSH:** `nyu-gpu` (requires NYU VPN, proxied through BigPurple)
- **GPUs:** 2x RTX 3090 (24GB each)
- **Use for:** Prototyping and individual benchmark runs
- **Datasets ready:** ImageNet-R, ImageNet-A, CIFAR-100, CUB-200, OmniBenchmark, VTAB (all in `~/qz-compcont-learning/data/`)
- **Datasets missing:** ObjectNet

```bash
ssh nyu-gpu
cd ~/MiN-NeurIPS2025
source .venv/bin/activate
```

For long SSH runs, wrap with `nohup`:

```bash
nohup bash run_benchmarks.sh > ~/min_bench.log 2>&1 &
strings ~/min_bench.log | grep FINAL
```

### H100 Cluster (3 nodes)

- **GPUs:** 3 nodes × 8 H100 (80GB each) = 24 H100s
- **Use for:** Full benchmark sweeps
- **Status:** Environment and dataset setup TBD

### GPU Memory (MiN ViT-B/16 backbone)

MiN uses `pretrained_vit_b16_224_in21k_min` (86M params, ~2-3GB VRAM at batch=64). Both RTX 3090s can run it comfortably. If OOM, reduce `batch_size` or `buffer_batch` in the model config.

## Benchmark Notes

- **Seed**: all `_ease` base configs use `seed=1993` with shuffled class order — matches the random order used in `compcont-bench` (which also shuffles by default).
- **Binary log output**: use `strings <logfile> | grep FINAL` to extract results from logs with binary CIFAR progress bars.

