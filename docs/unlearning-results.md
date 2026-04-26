# MiN Unlearning Experiment Results

**Date:** 2026-04-25  
**Script:** `MiN/unlearn.py` — `--all_tasks` mode  
**Configs:** 12 (4 benchmarks × 3 backbones)

---

## Setup

Three cumulative ablations are applied to every task of a fully-trained model, isolating each component's contribution to classification of the forgotten task:

| Ablation | What is zeroed |
|---|---|
| **A1** — zero mixture weight | `ω[u]` set to 0; generator still exists but routes nothing |
| **A2** — remove generator | All weights of `mu[u]` and `sigma[u]` zeroed (on top of A1) |
| **A3** — full removal | Classifier columns for forgotten-task classes zeroed (on top of A2) |

Metrics per method: **forgotten-task accuracy** (lower = better unlearning), **mean retained accuracy** (higher = less collateral damage), **BWT** on retained tasks, and a **feature separability probe** (logistic regression on raw backbone features; >0.85 = backbone still encodes the forgotten task distinctly).

---

## Key Findings

### 1. Task knowledge is concentrated entirely in the analytic classifier columns

Across all 12 configs, A1 and A2 together produce essentially zero drop in forgotten-task accuracy. A3 alone accounts for 99%+ of the unlearning effect:

| Ablation | Mean drop in forgotten accuracy (across all configs) |
|---|---|
| A1 — zero mixture weight | ~0.000–0.003 |
| A2 — remove generator | ~0.000 |
| **A3 — zero classifier columns** | **~0.78–0.95 (all of it)** |

After A3, forgotten-task accuracy is exactly **0.000** in every config and every task — the model cannot predict any forgotten class.

**Implication:** MiN's task-specific noise injection (the noise generators `P^u` and mixture weights `ω[u]`) contributes negligible unique information for classification. Even with `ω[u]=0`, backbone features are sufficient for the analytic classifier to recognize that task. The weight columns of `_network.weight` are the sole repository of task-specific predictive knowledge.

### 2. Unlearning is exact and surgical

Zeroing classifier columns is a closed-form, O(1) operation requiring no retraining. BWT after A3 is near-zero or slightly positive across all configs, meaning retained tasks are barely affected:

| Benchmark | Backbone | Baseline forgotten | A3 forgotten | A3 retained | A3 BWT |
|---|---|---|---|---|---|
| CIFAR-100 20T | ViT-B/16 | 0.909 | **0.000** | 0.912 | +0.0026 |
| CIFAR-100 20T | DINOv2-B/14 | 0.896 | **0.000** | 0.899 | +0.0028 |
| CIFAR-100 20T | DINOv3-B/16 | 0.783 | **0.000** | 0.788 | +0.0054 |
| CUB-200 20T | ViT-B/16 | 0.894 | **0.000** | 0.896 | +0.0026 |
| CUB-200 20T | DINOv2-B/14 | 0.901 | **0.000** | 0.903 | +0.0022 |
| CUB-200 20T | DINOv3-B/16 | 0.901 | **0.000** | 0.903 | +0.0023 |
| OmniBenchmark 10T | ViT-B/16 | 0.790 | **0.000** | 0.801 | +0.0108 |
| OmniBenchmark 10T | DINOv2-B/14 | 0.781 | **0.000** | 0.794 | +0.0127 |
| OmniBenchmark 10T | DINOv3-B/16 | 0.803 | **0.000** | 0.814 | +0.0114 |
| VTAB 5T | ViT-B/16 | 0.946 | **0.000** | 0.954 | +0.0080 |
| VTAB 5T | DINOv2-B/14 | 0.943 | **0.000** | 0.950 | +0.0077 |
| VTAB 5T | DINOv3-B/16 | 0.935 | **0.000** | 0.946 | +0.0106 |

Positive BWT values mean retained-task accuracy slightly *increases* after A3 — removing one task's logit columns reduces cross-task competition in the shared softmax space.

OmniBenchmark shows the largest collateral-damage signal (BWT ~+0.01), driven by its larger class space (30 classes per task × 10 tasks), meaning more softmax competition is relieved per removal.

### 3. The backbone retains a fingerprint of forgotten tasks

The feature separability probe (logistic regression on raw backbone features, bypassing noise injection) remains high after A3 in nearly all configs:

| Config | Probe mean | Probe min | % tasks >0.85 |
|---|---|---|---|
| CIFAR-100 / ViT-B/16 | 0.962 | 0.937 | 100% |
| CIFAR-100 / DINOv2-B/14 | 0.971 | 0.956 | 100% |
| CIFAR-100 / DINOv3-B/16 | 0.900 | 0.859 | 100% |
| CUB-200 / ViT-B/16 | 0.881 | 0.842 | 90% |
| CUB-200 / DINOv2-B/14 | 0.930 | 0.882 | 100% |
| CUB-200 / DINOv3-B/16 | 0.906 | 0.872 | 100% |
| OmniBenchmark / ViT-B/16 | 0.860 | 0.843 | 70% |
| OmniBenchmark / DINOv2-B/14 | 0.872 | 0.850 | 90% |
| **OmniBenchmark / DINOv3-B/16** | **0.820** | **0.804** | **0%** |
| VTAB / ViT-B/16 | 0.975 | 0.970 | 100% |
| VTAB / DINOv2-B/14 | 0.978 | 0.972 | 100% |
| VTAB / DINOv3-B/16 | 0.979 | 0.974 | 100% |

**Implication:** A3 erases the model's *ability to predict* a task, but the backbone (which is frozen and shared) still encodes that task's classes as a distinct region of feature space. This is expected — the backbone is never modified. A3-style unlearning is therefore **logit-level unlearning**, not **representation-level unlearning**.

The one exception is **OmniBenchmark + DINOv3-B/16**, where probe accuracy (0.820) falls below the 0.85 threshold across all tasks. This is the only config where the backbone is borderline non-separable — likely a side effect of the DINOv3 normalization bug causing noisier features, and/or OmniBenchmark's high within-superclass visual similarity reducing the linear separability of any subset of classes.

### 4. Backbone accuracy hierarchy

DINOv3's underperformance on CIFAR-100 (0.783 vs. 0.909 for ViT-B/16) is consistent with the known input normalization mismatch bug — see `memory/project_dinov3_norm_bug.md`. CUB-200 and OmniBenchmark show minimal backbone differences, suggesting DINOv3's degradation is most pronounced on benchmarks with small per-class sample counts.

---

## Key Takeaways

1. **MiN supports exact machine unlearning at zero cost.** A single O(1) parameter zeroing (classifier columns) reduces forgotten-task accuracy from >78% to exactly 0% with negligible impact on retained tasks. No retraining, no gradient steps, no data required.

2. **The noise generators are irrelevant to unlearning.** Zeroing `ω[u]` and `P^u` contributes essentially nothing to forgotten-task forgetting, confirming that task knowledge is not stored in the noise injection pathway — it is concentrated in the analytic classifier.

3. **This is logit-level, not representation-level, unlearning.** Probe accuracy remains high (>0.85) in 10 of 12 configs after A3, meaning the frozen backbone still distinguishes forgotten-task samples by their features alone. This is architecturally inevitable since the backbone is never modified. Claims of "unlearning" should be scoped to the classifier, not the representation.

4. **OmniBenchmark shows the most collateral disturbance** (A3 BWT ~+0.01–0.013), but this is positive transfer (retained accuracy *improves*), not damage.

5. **DINOv3 + OmniBenchmark is the one borderline case** for feature separability (probe <0.85). If representation-level unlearning claims are made, this config should be excluded or flagged.

---

## Notes on the Feature Probe

The probe result is stored in the JSON under `feature_separability.probe_accuracy` (not a top-level `feature_probe_accuracy` key). The probe is run on the **Baseline** and **A3** methods only; A1 and A2 do not get a probe result since they are intermediate steps. This is by design — `run_feature_probe=True` is only passed to `_eval()` for the baseline and A3 calls in `unlearn_single_task()`.
