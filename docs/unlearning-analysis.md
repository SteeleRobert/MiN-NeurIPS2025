# MiN Unlearning Analysis

Source: `MiN/unlearn.py`

## Overview

The unlearning analysis answers a diagnostic question: *where in the MiN model does task-specific knowledge live, and how completely can it be erased?*

After a model is fully trained on all tasks, the script holds the model fixed and applies three progressively more aggressive ablations to a chosen task. Each ablation zeroes a different subset of parameters; the difference in forgotten-task accuracy between consecutive ablations isolates the contribution of each component.

## MiN model structure

MiN stores task knowledge across three components per task `u`:

| Component | Parameters | Role |
|---|---|---|
| Mixture weight `ω[u]` | scalar per layer | Routes how much task `u`'s noise generator contributes to the mixed feature signal |
| Noise generator `P^u` | `mu[u]`, `sigma[u]` linear layers per backbone layer | Generates task-specific noise `ε_u` injected into the feature stream |
| Analytic classifier columns | columns of `_network.weight` for task `u`'s classes | Final logit computation; one column per class |

## The three ablations

Ablations are cumulative — each builds on the previous.

### A1 — Zero mixture weight (`ablation_a1_zero_weight`)

Sets `ω[task_to_forget] = 0` and renormalises the remaining weights to sum to 1. The noise generator `P^u` still exists but contributes nothing to the forward pass because its routing weight is zero. This is the weakest intervention: all parameters are intact, only routing is suppressed.

### A2 — Remove noise generator (`ablation_a2_remove_generator`)

Zeros all weights and biases of `mu[u]` and `sigmma[u]` for the forgotten task, ensuring `ε_u ≡ 0` for any input regardless of the mixture weight. Also zeros `ω[u]` for consistency. The model structure is preserved (the module is not removed from the `ModuleList`) so `state_dict` compatibility is maintained.

### A3 — Full removal (`ablation_a3_full_removal`)

Applies A2, then zeros the analytic classifier columns corresponding to the forgotten task's classes. After this, the model's logits for all forgotten-task classes are identically 0 — it can never predict those classes. This is the most complete intervention.

## Evaluation

For each of Baseline, A1, A2, A3, the evaluator (`UnlearningEvaluator`) computes:

**Forgotten task accuracy** — accuracy on the held-out test set for the forgotten task's classes only. Lower is better: strong unlearning should push this toward chance level.

**Mean retained accuracy** — mean accuracy across all other tasks. Should remain high; a drop indicates collateral damage.

**BWT (Backward Transfer)** — `mean(acc_after[t] - acc_before[t])` for all retained tasks, where `acc_before` is the baseline model. Negative BWT means unlearning degraded retained tasks; near-zero means the intervention was surgical.

**Confusion analysis** — records where forgotten-task samples are misclassified after ablation. A spread distribution across many classes indicates knowledge was redistributed; collapse to a few classes indicates migration to a surrogate.

**Feature separability probe** — trains a logistic regression classifier on raw backbone features (bypassing noise injection) to distinguish forgotten-task samples from retained-task samples. Probe accuracy above 0.85 means the backbone still encodes the forgotten task distinctly, i.e. unlearning erased routing/classification but not the underlying representation. Below 0.85 means the backbone itself has been altered. Run on A3 (or A2 if A3 is unavailable). Can be skipped with `--no_feature_probe`.

## Interpretation block

The interpretation block (`_print_interpretation`) decomposes the forgotten-task accuracy drop into three marginal effects:

```
Drop A1 vs baseline  = baseline_forgotten_acc − A1_forgotten_acc   (effect of weight routing)
Drop A2 vs A1        = A1_forgotten_acc       − A2_forgotten_acc   (effect of generator params)
Drop A3 vs A2        = A2_forgotten_acc       − A3_forgotten_acc   (effect of classifier columns)
```

Each drop quantifies how much forgotten-task accuracy fell when that component was removed. Thresholds for the printed commentary:

| Drop | Threshold | Interpretation |
|---|---|---|
| A1 vs baseline | > 10% | Mixture weights carry significant routing |
| A1 vs baseline | ≤ 10% | Mixture weights alone are insufficient |
| A2 vs A1 | > 10% | Noise generators store task-specific features |
| A2 vs A1 | ≤ 10% | P^u stores little beyond routing |
| A3 vs A2 | > 5% | Classifier columns add residual signal; A3 needed |
| A3 vs A2 | ≤ 5% | Classifier adds minimal signal once generator is removed |

Collateral damage commentary thresholds (based on worst BWT across A1/A2/A3):

| BWT | Interpretation |
|---|---|
| < −0.05 | Significant collateral damage |
| −0.05 to −0.01 | Minor collateral damage |
| ≥ −0.01 | Negligible collateral damage |

## `--all_tasks` mode

When run with `--all_tasks` (as `slurm/unlearn_benchmarks.slurm` does), the model is trained once and all ablations are applied for every task in sequence. After all tasks complete, `print_all_tasks_summary` prints three cross-task tables (forgotten accuracy, mean retained accuracy, BWT) and `save_all_tasks_plot` saves a heat-map PNG.

## `trained_model.pt` checkpoint

When run with `--save_checkpoint` (as the SLURM script always does), `unlearn.py` saves the fully-trained model to `<output_dir>/trained_model.pt` **before any ablations are applied**. This is the model that knows everything — the input to the unlearning analysis, not a result of it.

### Why a custom format

A standard `torch.save(model.state_dict())` only captures tensors registered with PyTorch (`nn.Parameter` or `register_buffer`). MiN's noise makers store two tensors as plain Python attributes — `weight_noise` (the mixture weights `ω`) and `w_down` (projection matrices) — that PyTorch's `state_dict` silently omits. Loading from a standard checkpoint would give wrong mixture weights, breaking the A1 ablation. The custom format (`save_unlearn_checkpoint` / `load_unlearn_checkpoint`) explicitly saves and restores these alongside the standard `state_dict` and `task_prototypes`.

### Purpose: skip retraining on reruns

Training is the expensive phase. If a job times out or crashes after training but before ablations finish, the checkpoint lets you restart from step 3 without retraining:

```bash
sbatch --export=ALL,CKPT_DIR=$HOME/min_unlearn_results slurm/unlearn_benchmarks.slurm
```

The SLURM script detects `$CKPT_DIR/<TAG>/trained_model.pt` and passes `--checkpoint` to `unlearn.py`, which loads the model and jumps straight to the ablation phase.

## Outputs

For each task `t`, results are written to `<output_dir>/task_<t>/`:

| File | Contents |
|---|---|
| `unlearn_task<t>_<timestamp>.json` | All metrics for Baseline/A1/A2/A3, plus metadata |
| `accuracy_comparison.png` | Bar chart: forgotten vs mean-retained accuracy per method |
| `retained_per_task.png` | Line plot: per-task retained accuracy per method |
| `confusion_hist_A3-full_removal.png` | Histogram of where forgotten samples are predicted |

At the `--all_tasks` level:

| File | Contents |
|---|---|
| `all_tasks_summary_<timestamp>.json` | Aggregated results for all tasks |
| `all_tasks_heatmap.png` | Heat-map of forgotten-task accuracy across all (task, method) pairs |
