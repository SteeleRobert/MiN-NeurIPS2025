# Fixed-MiN on iNat21-mini (C = 10,000) — NeurIPS 29520 rebuttal

**Protocol:** iNat21-**mini**, 10,000 classes, B0-Inc500, 20 stages, 500,000 train
(50/class) / 100,000 val (10/class). Seeds {1993, 42, 7, 2024, 31337}, class order
shuffled per seed. Metrics: `A_B` final average accuracy, `Ā` average incremental
accuracy, `BWT` backward transfer, `task0` first-task accuracy after the last stage.

**Data variant:** matches the `feature/inat21-10k-characterization` Arrow cache
exactly — cache declares train 500,000 / validation 100,000 / 10,000 classes, and the
raw ImageFolder on GPFS (`public_data/inat21/{train_mini,val}`) is identical. Class
indices align (`00000_…_Lumbricus_terrestris` = Arrow index 0). The April MiN runs'
`total_samples = 2,625,000` corresponds to no raw data present on GPFS.

**Configuration:** "fixed" = the min_stab winners — L2-normalize features before the
random projection (`feature_norm: true`) with `gamma: 1`. "orig" = MiN exactly as
published (`gamma: 500`, no normalization). No other differences.

## Results

| Config | Backbone | n | A_B | Ā | BWT | task0 |
|---|---|--:|--:|--:|--:|--:|
| fixed | DINOv3-B/16 | 5 | **64.67 ± 0.07** | 75.23 ± 0.15 | −0.110 | 65.26 |
| fixed | DINOv2-B/14 | 5 | **62.44 ± 0.03** | 73.44 ± 0.15 | −0.115 | 62.33 |
| orig | DINOv3-B/16 | 1 | 37.80 | 53.05 | −0.180 | 45.88 |
| orig | DINOv2-B/14 | 1 | 62.51 | 74.67 | −0.144 | 67.60 |

Per-seed A_B (seeds 7 / 42 / 1993 / 2024 / 31337):

- fixed DINOv3-B: 64.69, 64.57, 64.71, 64.75, 64.65
- fixed DINOv2-B: 62.41, 62.47, 62.42, 62.47, 62.45

## What this shows

**The fix is worth +26.9 pp on DINOv3-B/16 at C = 10,000** (37.80 → 64.67), reproducing
at 10k classes the same repair measured across the six EASE benchmarks. BWT improves in
step (−0.180 → −0.110) and first-task retention nearly doubles (45.88 → 65.26).

**The fix does nothing on DINOv2-B/14** (62.51 → 62.44, a 0.07 pp difference against a
0.03 seed sd). This is the control that makes the claim precise: the intervention is not
a general accuracy boost, it repairs a specific mismatch between MiN's shipped
hyperparameters and a backbone whose feature statistics differ from the one they were
tuned on. Where the statistics already suit `gamma = 500`, normalizing changes nothing.

**Seed variance is negligible** at this scale — sd ≤ 0.07 pp across five seeds for both
fixed cells, against the ±27 pp swings seen on cifar-100 pre-fix. The bimodal collapse
mode is absent here.

**Against the frozen statistical heads, fixed MiN still loses** at C = 10,000:

| Method | A_B | BWT |
|---|--:|--:|
| FeCAM | 66.97 | — |
| MR-QDA (rank-50) | 66.25 | −0.076 |
| MR-QDA (dense) | 66.04 | — |
| **fixed MiN (DINOv3-B)** | **64.67** | **−0.110** |

So the honest reading is that repairing MiN makes it a credible comparator rather than a
broken one, and it is still behind the frozen heads on both accuracy and forgetting at
10k classes. The April numbers (46.2 for DINOv3-B, degrading monotonically to 30.6 for
H/16) reflect the unrepaired configuration and should not be used as comparators.

## Provenance

- Jobs: `25840236` (sp-0014), `25840246` (sp-0007), `25840247` (sp-0013); shared
  work queue with atomic-`mkdir` claims (flock does **not** give cross-node mutual
  exclusion on GPFS).
- Results: `results/{fn-g1,orig}/inat21/MiN_<backbone>_seed<seed>.json`,
  compcont-comparable format with full per-stage `task_accs_history`.
- `manifest.tsv` — every state transition for all cells, including failures.
- `pipeline/` — dataset class, configs, worker and launcher actually used.
- Wall-clock: ~8.5 h per B-tier cell on one H100, against an 8.7 h projection from a
  throughput model calibrated on the cifar-100 and omnibenchmark runs.

### Known issues

- Six `exit=1` failures, all of them the three DINOv2-B/14 cells retried twice: DINOv2
  loads via `torch.hub`, which writes to `/gpfs/home/steelr04/.cache`, read-only inside
  the container. Fixed by setting `TORCH_HOME`/`XDG_CACHE_HOME` to writable GPFS paths;
  all three then completed. No completed cell has an unresolved failed sibling.
- DINOv3-L/16 (5 seeds + 1 orig control) and DINOv3-H/16 (3 seeds) were queued and were
  still running when the allocations expired; they are **not** included above.
- TF32 is disabled in these runs. Enabling it reportedly offers 2.5–3.5× throughput but
  changes matmul precision, so it was deliberately not applied mid-flight — the cells
  would not have been numerically consistent with each other or with the completed sweep.
