# DINO benchmark sweeps (Slurm)

This directory contains Slurm scripts to run MiN EASE-protocol benchmarks with DINO backbones (`dinov2_vitb14`, `dinov3_vitb16`), using two GPUs in parallel via `run_benchmarks.sh` (`GPU0` / `GPU1`).

## Prerequisites

### Repository and environment

From the `MiN-NeurIPS2025` repository root:

```bash
cd /path/to/MiN-NeurIPS2025
uv sync
```

Slurm jobs require `.venv/bin/python` to exist. The scripts exit with an error if the virtualenv is missing.

### Data

By default, `DATA_ROOT` is `$HOME/qz-compcont-learning/data`. `run_benchmarks.sh` runs `prepare_data.py` when ImageNet-R / ImageNet-A train–test splits are missing. Some datasets (e.g. CUB-200, Omnibenchmark) must be laid out manually under that tree; see comments at the top of `run_benchmarks.sh`. Override the location with `DATA_ROOT` if your data live elsewhere.

### DINOv3 (Hugging Face)

DINOv3 is gated. Authenticate once on the cluster, for example:

```bash
huggingface-cli login
```

Or pass a token into the job (example for `dinov3_vitb16` only):

```bash
sbatch --export=ALL,BACKBONE=dinov3_vitb16,HF_TOKEN="$HF_TOKEN" slurm/run_dino_benchmarks.slurm
```

## Submit from the repository root

Always `cd` into `MiN-NeurIPS2025` before `sbatch` so `SLURM_SUBMIT_DIR` is the repo (the scripts `cd` there and require `run_benchmarks.sh`).

```bash
cd /path/to/MiN-NeurIPS2025
```

## Ways to run

### Option A: two separate jobs (DINOv2 and DINOv3)

```bash
./slurm/submit_dino_backbones.sh
```

This submits `run_dino_benchmarks.slurm` twice: first `BACKBONE=dinov2_vitb14`, then `BACKBONE=dinov3_vitb16`.

Optional Slurm overrides (example):

```bash
SLURM_EXTRA="--partition=gpu4_short --time=12:00:00" ./slurm/submit_dino_backbones.sh
```

### Option B: single backbone

```bash
sbatch --export=ALL,BACKBONE=dinov2_vitb14 slurm/run_dino_benchmarks.slurm
# or
sbatch --export=ALL,BACKBONE=dinov3_vitb16 slurm/run_dino_benchmarks.slurm
```

Allowed values for `BACKBONE` are `dinov2_vitb14` and `dinov3_vitb16` only.

### Option C: job array (both backbones, one submission)

```bash
sbatch slurm/array_dino_benchmarks.slurm
```

- Array index `0` runs `dinov2_vitb14`; index `1` runs `dinov3_vitb16`.
- The default `#SBATCH --array=0-1%1` runs one backbone at a time on one 2-GPU node. Edit to `%2` if you want both tasks in parallel on two nodes (and your allocation allows it).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DATA_ROOT` | Dataset root (default `$HOME/qz-compcont-learning/data`) |
| `LOG_DIR` | Benchmark logs (default `$HOME/min_benchmark_logs/<BACKBONE>`) |
| `REPO_ROOT` | Repo path; usually unnecessary (defaults to the directory from which you ran `sbatch`) |
| `SLACK_WEBHOOK_URL` | Optional Slack notification on job completion |
| `slurm/.slack_webhook_url` | Alternative: one-line webhook file (see `slack_notify.sh`) |

Example with overrides:

```bash
sbatch --export=ALL,BACKBONE=dinov2_vitb14,DATA_ROOT=/path/to/data,LOG_DIR=/path/to/logs slurm/run_dino_benchmarks.slurm
```

## Slurm directives and site configuration

`run_dino_benchmarks.slurm` and `array_dino_benchmarks.slurm` request **1 node, 2 GPUs, 16 CPUs, 64G memory, 3 days**, partition `gpu4_medium`. Adjust `#SBATCH` lines to match your cluster.

If your site requires **account** or **QoS**, uncomment and set in the `.slurm` files:

```bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --qos=<YOUR_QOS>
```

## Outputs and monitoring

- **Slurm stdout/stderr**: `dino-bench-<jobid>.out` and `dino-bench-<jobid>.err` in the repository root. For array jobs, names use `dino-bench-%A_%a.out` / `.err`.
- **Benchmark logs**: under `LOG_DIR` (per backbone by default).
- **Queue**: `squeue -u "$USER"`

## Files in this directory

| File | Role |
|------|------|
| `run_dino_benchmarks.slurm` | Single job; set `BACKBONE` via `--export` |
| `array_dino_benchmarks.slurm` | Array over both DINO backbones |
| `submit_dino_backbones.sh` | Submits two jobs (v2 then v3) |
| `slack_notify.sh` | Sourced by the Slurm scripts for optional Slack notifications |
