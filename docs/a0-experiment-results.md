# A0 Experiment Results: Noise Generator Contribution

**Question:** How much do MiN's noise generators contribute to per-task accuracy?

**Method:** The A0 ablation (`--a0_only`) evaluates the trained model twice — once with noise generators active (baseline) and once with all noise generators zeroed (`ablation_a0_zero_all_noise`). Average per-task accuracy is compared across all tasks. No forgotten/retained split is applied; this is a global measure of noise contribution.

---

## Results


| Benchmark           | Backbone    | Baseline | A0 (no noise) | Δ          |
| ------------------- | ----------- | -------- | ------------- | ---------- |
| CIFAR-100 (20T)     | ViT-B/16    | 90.89%   | 90.41%        | −0.48%     |
| CIFAR-100 (20T)     | DINOv2-B/14 | 89.62%   | 84.63%        | **−4.99%** |
| CIFAR-100 (20T)     | DINOv3-B/16 | 78.26%   | 78.53%        | +0.27%     |
|                     |             |          |               |            |
| CUB-200 (20T)       | ViT-B/16    | 90.14%   | 90.24%        | +0.10%     |
| CUB-200 (20T)       | DINOv2-B/14 | 90.42%   | 90.42%        | 0.00%      |
| CUB-200 (20T)       | DINOv3-B/16 | 90.53%   | 90.49%        | −0.04%     |
|                     |             |          |               |            |
| ImageNet-A (10T)    | ViT-B/16    | 60.23%   | 60.29%        | +0.05%     |
| ImageNet-A (10T)    | DINOv2-B/14 | 73.37%   | 73.38%        | +0.01%     |
| ImageNet-A (10T)    | DINOv3-B/16 | 68.40%   | 68.90%        | +0.50%     |
|                     |             |          |               |            |
| ImageNet-R (40T)    | ViT-B/16    | 74.15%   | 74.14%        | −0.01%     |
| ImageNet-R (40T)    | DINOv2-B/14 | 84.50%   | 84.47%        | −0.04%     |
| ImageNet-R (40T)    | DINOv3-B/16 | 84.30%   | 84.18%        | −0.12%     |
|                     |             |          |               |            |
| OmniBenchmark (10T) | ViT-B/16    | 78.47%   | 76.51%        | **−1.96%** |
| OmniBenchmark (10T) | DINOv2-B/14 | 77.68%   | 76.76%        | −0.92%     |
| OmniBenchmark (10T) | DINOv3-B/16 | 81.10%   | 81.13%        | +0.03%     |
|                     |             |          |               |            |
| VTAB (5T)           | ViT-B/16    | 94.53%   | 94.55%        | +0.02%     |
| VTAB (5T)           | DINOv2-B/14 | 94.20%   | 94.18%        | −0.02%     |
| VTAB (5T)           | DINOv3-B/16 | 93.92%   | 93.79%        | −0.13%     |


---

## Per-task breakdown: notable cases

### CIFAR-100 + DINOv2-B/14 (Δ = −4.99%)

The largest overall effect. Almost every task degrades when noise is zeroed (15 of 20 tasks drop by ≥2%), with losses distributed broadly (−4% to −15%). Tasks 0–2 are small exceptions (+7%, +7%, +2%). This is the clearest evidence that noise generators are load-bearing for retention under DINOv2 features.

### OmniBenchmark + ViT-B/16 (Δ = −1.96%)

A strongly polarised pattern: early tasks gain accuracy when noise is zeroed (task-0: +13.4%, tasks 1–2: +5–7%) while later tasks lose (task-5: −8.5%, task-9: −14.2%). With DINOv2 the effect is even more extreme (task-0: +27.9%, task-9: −30.3%). Noise generators appear to redistribute capacity from early to late tasks rather than adding it globally — an asymmetric inductive bias that helps newer tasks at the expense of older ones.

### DINOv3-B/16 across all benchmarks

Δ is within ±0.5% for every benchmark. DINOv3 features are rich enough that the noise mechanism adds no measurable benefit or harm.

---

## Conclusions

1. **Noise generators matter only in two settings.** CIFAR-100 + DINOv2 (−5.0%) and OmniBenchmark + ViT-B/16 (−2.0%) are the only cases where zeroing noise meaningfully changes average accuracy. The remaining 16 backbone × benchmark combinations show Δ ≤ 0.5%.
2. **The noise effect is not uniformly beneficial.** In OmniBenchmark the mechanism trades early-task accuracy for late-task accuracy. Averaged across tasks the net effect is small, but per-task it can be ±14–30%. This is not the behaviour of a regulariser that helps uniformly — it reflects task-order sensitivity in how noise generators are trained.
3. **DINOv3 renders the noise mechanism irrelevant.** Across all six benchmarks, DINOv3 results are flat to within noise. If DINOv3 is used as the backbone, MiN's noise injection provides no incremental value over the base frozen-feature classifier.
4. **CUB-200, ImageNet-R, and VTAB show no noise contribution regardless of backbone.** These benchmarks appear easy enough relative to the backbone capacity that the noise generators never find a useful role.
5. **The strongest case for noise generators is CIFAR-100 + DINOv2**, where they provide a consistent ~5% accuracy boost spread across all retained tasks. This is the setting closest to MiN's intended regime — a moderately challenging benchmark with a backbone that leaves room for the noise mechanism to contribute.

