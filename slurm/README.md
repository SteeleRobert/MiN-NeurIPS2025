# DINO benchmark sweeps (Slurm)

This directory contains Slurm scripts to run all MiN EASE-protocol benchmarks with DINO backbones (`dinov2_vitb14`, `dinov3_vitb16`). All benchmarks run in parallel — one per GPU — on a single node with 7 GPUs.

## Prerequisites

### Repository and environment

From the `MiN-NeurIPS2025` repository root:

```bash
cd /path/to/MiN-NeurIPS2025
uv sync
```

Slurm jobs require `.venv/bin/python` to exist. The scripts exit with an error if the virtualenv is missing.

### Data

By default, `DATA_ROOT` is `/gpfs/data/oermannlab/public_data/continual_learning` (bigpurple shared data). The scripts run `prepare_data.py` automatically when ImageNet-R / ImageNet-A train–test splits are missing. Benchmarks whose data directories are absent or empty are skipped. Override the location with `DATA_ROOT` if your data live elsewhere.

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

### Option A: two separate jobs (DINOv2 and DINOv3)

```bash
./slurm/submit_dino_backbones.sh
```

Submits `run_dino_benchmarks.slurm` twice: once for `BACKBONE=dinov2_vitb14`, once for `BACKBONE=dinov3_vitb16`. Both jobs run concurrently (subject to scheduler availability).

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

- Array index `0` → `dinov2_vitb14`; index `1` → `dinov3_vitb16`.
- `#SBATCH --array=0-1%2` runs both array tasks in parallel on two separate nodes.
- To run only one backbone: `sbatch --array=0 slurm/array_dino_benchmarks.slurm` (DINOv2) or `--array=1` (DINOv3).

## Parallel execution

Each job requests **7 GPUs** (one per benchmark). All available benchmarks are launched simultaneously as background processes; `wait` collects them at the end. GPU assignment is sequential: CIFAR-100 always gets GPU 0, then each subsequent benchmark that passes the data check gets the next GPU index.

Benchmarks and their protocols:

| Benchmark     | Protocol       | Data check path                                  |
|---------------|----------------|--------------------------------------------------|
| CIFAR-100     | B0-Inc5 (20T)  | auto-downloads, always runs                      |
| ImageNet-R    | B0-Inc10 (20T) | `$DATA_ROOT/imagenet-r-split/{train,test}/`      |
| ImageNet-A    | B0-Inc20 (10T) | `$DATA_ROOT/imagenet-a-split/{train,test}/`      |
| CUB-200       | B0-Inc10 (20T) | `$DATA_ROOT/cub/{train,test}/`                   |
| OmniBenchmark | B0-Inc30 (10T) | `$DATA_ROOT/omnibenchmark/{train,test}/`         |
| VTAB          | B0-Inc10 (5T)  | `$DATA_ROOT/vtab/{train,test}/`                  |
| ObjectNet     | B0-Inc10 (20T) | `$DATA_ROOT/objectnet/{train,test}/`             |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DATA_ROOT` | Dataset root (default `/gpfs/data/oermannlab/public_data/continual_learning`) |
| `LOG_DIR` | Benchmark logs (default `$HOME/min_benchmark_logs/<BACKBONE>`) |
| `REPO_ROOT` | Repo path; usually unnecessary (defaults to `SLURM_SUBMIT_DIR`) |
| `SLACK_WEBHOOK_URL` | Optional Slack notification on job completion |
| `slurm/.slack_webhook_url` | Alternative: one-line webhook file (see `slack_notify.sh`) |

Example with overrides:

```bash
sbatch --export=ALL,BACKBONE=dinov2_vitb14,DATA_ROOT=/path/to/data,LOG_DIR=/path/to/logs slurm/run_dino_benchmarks.slurm
```

## Slurm directives and site configuration

Both `run_dino_benchmarks.slurm` and `array_dino_benchmarks.slurm` request **1 node, 7 GPUs, 56 CPUs, 224G memory, 3 days**, partition `a100_short`. Adjust `#SBATCH` lines to match your cluster. If fewer than 7 datasets are available, some GPUs will be idle — you can lower `--gres=gpu` accordingly.

If your site requires **account** or **QoS**, uncomment and set in the `.slurm` files:

```bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --qos=<YOUR_QOS>
```

## Outputs and monitoring

- **Slurm stdout/stderr**: `dino-bench-<jobid>.out` / `.err` in the repository root. Array jobs use `dino-bench-%A_%a.out` / `.err`.
- **Benchmark logs**: `$LOG_DIR/<benchmark>_<backbone>.log` for each run.
- **FINAL lines**: `strings <logfile> | grep FINAL` — printed to the job's stdout at completion.
- **Queue**: `squeue -u "$USER"`

## Files in this directory

| File | Role |
|------|------|
| `run_dino_benchmarks.slurm` | Single job; set `BACKBONE` via `--export` |
| `array_dino_benchmarks.slurm` | Array over both DINO backbones (runs in parallel, `%2`) |
| `submit_dino_backbones.sh` | Submits two separate jobs (DINOv2 then DINOv3) |
| `slack_notify.sh` | Sourced by Slurm scripts for optional Slack notifications |
