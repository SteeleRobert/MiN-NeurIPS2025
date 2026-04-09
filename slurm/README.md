# DINO benchmark sweeps (Slurm)

This directory contains Slurm scripts to run all MiN EASE-protocol benchmarks with DINO backbones (`dinov2_vitb14`, `dinov3_vitb16`). Each benchmark runs as a separate array task on its own GPU.

## Prerequisites

### Repository and environment

From the `MiN-NeurIPS2025` repository root:

```bash
cd /path/to/MiN-NeurIPS2025
uv sync
```

Slurm jobs require `.venv/bin/python` to exist. The scripts exit with an error if the virtualenv is missing.

### Data

By default, `DATA_ROOT` is `/gpfs/data/oermannlab/public_data/continual_learning` (bigpurple shared data). The scripts run `prepare_data.py` automatically when ImageNet-R / ImageNet-A train–test splits are missing. Benchmarks whose data directories are absent or empty exit cleanly (code 0) without failing the array job. Override the location with `DATA_ROOT` if your data live elsewhere.

### DINOv3 (Hugging Face)

DINOv3 is gated. Authenticate once on the cluster, for example:

```bash
huggingface-cli login
```

Or pass a token into the job:

```bash
sbatch --export=ALL,BACKBONE=dinov3_vitb16,HF_TOKEN="$HF_TOKEN" slurm/run_dino_benchmarks.slurm
```

## Submit from the repository root

Always `cd` into `MiN-NeurIPS2025` before `sbatch` so `SLURM_SUBMIT_DIR` is the repo root (the scripts `cd` there on startup).

```bash
cd /path/to/MiN-NeurIPS2025
```

## Ways to run

### Option A: two per-backbone arrays (DINOv2 and DINOv3)

```bash
./slurm/submit_dino_backbones.sh
```

Submits `run_dino_benchmarks.slurm` as a 7-task array twice: once for `BACKBONE=dinov2_vitb14`, once for `BACKBONE=dinov3_vitb16`. Both arrays are submitted immediately and run concurrently (subject to scheduler availability).

Optional Slurm overrides (example):

```bash
SLURM_EXTRA="--partition=gpu4_short --time=12:00:00" ./slurm/submit_dino_backbones.sh
```

### Option B: single backbone array

```bash
sbatch --export=ALL,BACKBONE=dinov2_vitb14 slurm/run_dino_benchmarks.slurm
# or
sbatch --export=ALL,BACKBONE=dinov3_vitb16 slurm/run_dino_benchmarks.slurm
```

Submits a 7-task array (`--array=0-6`). Allowed values for `BACKBONE` are `dinov2_vitb14` and `dinov3_vitb16` only.

To run a single benchmark only (e.g. CIFAR-100):

```bash
sbatch --array=0 --export=ALL,BACKBONE=dinov2_vitb14 slurm/run_dino_benchmarks.slurm
```

### Option C: combined array (both backbones, one submission)

```bash
sbatch slurm/array_dino_benchmarks.slurm
```

Submits 14 array tasks (`--array=0-13`): tasks 0–6 run all 7 benchmarks with `dinov2_vitb14`; tasks 7–13 run the same benchmarks with `dinov3_vitb16`.

To run only one backbone:

```bash
sbatch --array=0-6  slurm/array_dino_benchmarks.slurm   # DINOv2 only
sbatch --array=7-13 slurm/array_dino_benchmarks.slurm   # DINOv3 only
```

## Array task layout

Each array task runs **one benchmark on one GPU**. The benchmark index (within a backbone) maps as follows:

| Task index (mod 7) | Benchmark     | Protocol       | Data check path                             |
|--------------------|---------------|----------------|---------------------------------------------|
| 0                  | CIFAR-100     | B0-Inc5 (20T)  | auto-downloads, always runs                 |
| 1                  | ImageNet-R    | B0-Inc10 (20T) | `$DATA_ROOT/imagenet-r/{train,test}/`       |
| 2                  | ImageNet-A    | B0-Inc20 (10T) | `$DATA_ROOT/imagenet-a/{train,test}/`       |
| 3                  | CUB-200       | B0-Inc10 (20T) | `$DATA_ROOT/cub/{train,test}/`              |
| 4                  | OmniBenchmark | B0-Inc30 (10T) | `$DATA_ROOT/omnibenchmark/{train,test}/`    |
| 5                  | VTAB          | B0-Inc10 (5T)  | `$DATA_ROOT/vtab/{train,test}/`             |
| 6                  | ObjectNet     | B0-Inc10 (20T) | `$DATA_ROOT/objectnet/{train,test}/`        |

For `array_dino_benchmarks.slurm`, task IDs 0–6 are `dinov2_vitb14` and 7–13 are `dinov3_vitb16`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `BACKBONE` | Required for `run_dino_benchmarks.slurm`; derived from task ID in `array_dino_benchmarks.slurm` |
| `DATA_ROOT` | Dataset root (default `/gpfs/data/oermannlab/public_data/continual_learning`). Passed to `main.py` via `--data_root`, overriding the hardcoded path in base config JSONs. |
| `LOG_DIR` | Benchmark logs (default `$HOME/min_benchmark_logs/<BACKBONE>`) |
| `REPO_ROOT` | Repo path; usually unnecessary (defaults to `SLURM_SUBMIT_DIR`) |
| `SLACK_WEBHOOK_URL` | Optional Slack notification on job completion |
| `slurm/.slack_webhook_url` | Alternative: one-line webhook file (see `slack_notify.sh`) |

Example with overrides:

```bash
sbatch --export=ALL,BACKBONE=dinov2_vitb14,DATA_ROOT=/path/to/data,LOG_DIR=/path/to/logs slurm/run_dino_benchmarks.slurm
```

## Slurm directives and site configuration

Each array task requests **1 node, 1 GPU, 8 CPUs, 32G memory, 3 days**, partition `a100_short`. Adjust `#SBATCH` lines to match your cluster.

If your site requires **account** or **QoS**, uncomment and set in the `.slurm` files:

```bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --qos=<YOUR_QOS>
```

## Outputs and monitoring

- **Slurm stdout/stderr**: `dino-bench-<arrayid>_<taskid>.out` / `.err` in the repository root.
- **Benchmark logs**: `$LOG_DIR/<benchmark>_<backbone>.log` for each run.
- **FINAL lines**: `strings <logfile> | grep FINAL` — printed to the task's stdout at completion.
- **Queue**: `squeue -u "$USER"`

## Files in this directory

| File | Role |
|------|------|
| `run_dino_benchmarks.slurm` | 7-task array (one per benchmark); set `BACKBONE` via `--export` |
| `array_dino_benchmarks.slurm` | 14-task array over both DINO backbones (7 benchmarks × 2 backbones) |
| `submit_dino_backbones.sh` | Submits two 7-task arrays (DINOv2 then DINOv3) |
| `slack_notify.sh` | Sourced by Slurm scripts for optional Slack notifications |
