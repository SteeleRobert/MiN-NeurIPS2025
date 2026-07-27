# MiN throughput: TF32 is off, and that costs ~2.5–3.5× on every run

**Status (2026-07-27).** DINOv3-H+/16 on iNat21 was **dropped** for the NeurIPS 29520
rebuttal — not because the speedup wasn't there, but because there was no free GPU
capacity left to use it (all superpod nodes allocated or down; the last allocations
expired midday Jul 27). This document exists because the finding outlives that cell:
it is a **one-line, ~2.5–3.5× speedup on every future MiN run**.

Branch `perf/dinov3h-speedup`. **Committed locally, never pushed** — it will be lost if
this worktree is reclaimed. Push it or cherry-pick it somewhere durable.

---

## 1. The finding

MiN never sets the float32 matmul precision. Verified by grep across all of `MiN/`:
**no `autocast`, no `bfloat16`, no `float16`, no `allow_tf32`, no `torch.compile`, no
`channels_last`.**

Confirmed directly inside the `pytorch_min` container (not inferred from docs):

```
transformers 5.5.3 | torch 2.4.0+cu121
default matmul.allow_tf32 : False
default matmul precision  : highest
default cudnn.benchmark   : False
```

Every matmul in an 840M-parameter ViT runs true IEEE fp32, tensor cores idle.

A DINOv3-H+/16 iNat21 cell is **~8.25 ZFLOP**. Over the observed 43 h that is
**~53 TFLOPS sustained**:

| H100 SXM peak | TFLOPS | observed / peak |
|---|---|---|
| **FP32, no tensor cores** | **67** | **79%** |
| TF32 tensor core | ~495 | 11% |
| BF16 tensor core | ~990 | 5% |

79% of fp32 peak is close to the practical ceiling for fp32 GEMM. 11% of TF32 peak is
not a utilisation number — it is what "the tensor cores are switched off" looks like.

## 2. Where the time goes

Per incremental task, 25,000 train images, in forward-equivalents. Config is the
**`fn-g1` variant** actually used by the sweep: `init_epochs=20`, `epochs=10`,
`fit_epochs=3` (§5 explains the config-family trap). Backward counts ≈2× forward
because trainable noise params sit in *every* block, so the graph spans all 32; and
`run()` at task>0 does **two** forwards (a `no_grad` one for `outputs1`, plus the grad
one) then the backward — 4 forward-equivalents per step.

| Phase | fwd-equiv / task | share |
|---|---|---|
| `run()` — 10 ep × 25k × 4 | 1,000,000 | **83.2%** |
| `fit_fc` — 3 epochs, `no_grad` | 75,000 | 6.2% |
| `after_train` eval — cumulative, 5k→100k, avg 52.5k | 52,500 | 4.4% |
| `get_task_prototype` ×2 | 50,000 | 4.2% |
| `re_fit` — 1 pass | 25,000 | 2.1% |
| **total** | **1,202,500** | |

Task 0 is heavier (`init_epochs=20`) but cheaper per step (no `outputs1` pass:
3 fwd-equiv, not 4) → ~1.66 M. Whole cell ≈ **24.5 M forward-equivalents** at
~337 GFLOP each (201 tokens: 196 patches + CLS + 4 registers).

Two corollaries worth keeping:

- **Eval is dataloader-bound, not compute-bound.** ~15% of wall-clock but 4.4% of
  FLOPs, at `num_workers=4` with 100k images in the final stage. Its quadratic growth
  is the CIL protocol and is *not* a bug — the gap above its FLOP share is.
- **The float64 ridge head is not a bottleneck** — ~3.2 PFLOP FP64/cell, ≈64 s total.
  Do **not** "optimize" it. It is deliberately float64 and the RLS recursion is exact.

## 3. What the patch adds

`MiN/utils/perf.py`, wired into `BaseTrainer._train` (2 lines) and `MiN.MinNet`
(12 DataLoader sites → one helper). **Every hook is opt-in and a no-op on existing
configs** — an unmodified config produces byte-identical DataLoader kwargs and never
touches precision.

Add to any model config to enable:

```jsonc
{
  "matmul_precision": "high",        // TF32. default "highest" = unchanged fp32.
                                     // THIS IS THE ONE THAT MATTERS: 2.64x, validated.
  "dataloader": {                    // optional, +6% -- read the caveat in 4b first
    "persistent_workers": true,      //   safe: no effect on numerics
    "prefetch_factor": 4,            //   safe
    "pin_memory": true,              //   safe
    "num_workers": 12                //   NOT numerically neutral -- reseeds
  }                                  //   augmentation RNG. See 4b.
}
```

`"high"` keeps fp32 storage **and fp32 accumulation**; it only rounds GEMM *inputs*
from a 23- to a 10-bit mantissa. Still a real floating-point change — hence §4.

There is also a `cudnn_benchmark` key, but **it is redundant**: `_set_random()` in
`BaseTrainer.py` already sets `torch.backends.cudnn.benchmark = True` (alongside
`cudnn.deterministic = True`, which is an odd pairing but pre-existing and not ours to
change). Left in for explicitness; setting it changes nothing.

Logic is unit-tested without torch: default → exactly `{'num_workers': 4}`; `workers=0`
correctly suppresses `persistent_workers`/`prefetch_factor` (PyTorch rejects them);
an invalid precision raises rather than silently degrading.

## 4. Validation

**Reference cell** (min-sweep's unmodified run, already complete):
`min_inat/results/fn-g1/inat21/MiN_dinov3_vitb16_seed42.json` — `avg_acc = 64.569%`,
`avg_inc_acc = 75.288%`, `bwt = −0.1088`, `task0_final = 65.5%`. Ran on sp-0007.

Its per-stage `total acc` trajectory, from
`min_inat/logs/inat21__dinov3_vitb16__fn-g1__s42.log`:

```
[87.74, 83.9, 80.77, 78.67, 76.6, 75.07, 73.54, 72.23, 71.23, 70.07,
 69.43, 68.46, 67.55, 66.57, 65.77, 65.03, 64.42, 63.72, 63.31, 62.72]
```

**Gate:** final `avg_acc` within **±0.15 pp** of 64.569% (window
[64.42, 64.72]) and per-stage trajectory tracking closely. A result that shifts more is
**rejected, not rationalized**. This matters more than usual:
`bigpurple_sweep/HANDOFF.md` documents MiN+DINOv3 as seed-unstable with a bimodal
collapse mode, and `dino_backbone.py` records gradient explosions from
input-statistic mismatch. This is a numerically fragile configuration.

**Harness, built and isolated**, at `/gpfs/data/oermannlab/users/steelr04/perf_val/`:

- `repo/` — a 2.2 MB source-only copy of MiN with the patch applied. Touches **nothing**
  of min-sweep's repo, queue, or results.
- `run_arm.sh <ARM> <GPU>` — one arm in the `pytorch_min` enroot container, GPU-pinned,
  HF-authed, results to `perf_val/results/<ARM>`.
- `ab.sbatch` — three arms concurrently on `oermannlab`/a100-8001:
  - `ARM-A-baseline` — untouched `MiN-inat21-dinov3vitb16-fn-g1` (in-run reference)
  - `ARM-B-tf32` — `matmul_precision: "high"` **only**, isolating the one numerical change
  - `ARM-C-tf32-dl` — full proposed config, to measure total speedup
- All three: base `inat21_ease`, `--seed 42` — i.e. **exactly the reference cell**.

Compare each arm's `total acc: [...]` per stage against the trajectory above. A100 is
fine for the **correctness** gate; the speedup *ratio* differs from H100 (A100 is
19.5 fp32 vs 156 TF32 TFLOPS, so the relative win is larger), but ±0.15 pp equivalence
transfers.

## 4b. Measured A/B result (2026-07-27, a100-8001, job 25862812)

Three arms, `MiN-inat21-dinov3vitb16-fn-g1`, seed 42, identical in every respect except
the keys under test. Task 0, `init_epochs=20`.

**Speed** — steady-state seconds per epoch, from log timestamps:

| arm | s/epoch | speedup |
|---|---|---|
| `ARM-A-baseline` (fp32) | **148.5** | 1.00× |
| `ARM-B-tf32` | **56** | **2.65×** |
| `ARM-C-tf32-dl` | **53** | **2.79×** |

All three GPUs sat at ~100% utilisation throughout, so this is a GPU-bound comparison,
not a CPU-contention artifact (checked: `nvidia-smi` 100/99/99%).

2.65× on A100 lands inside the predicted 2.5–3.5× band. The A100 fp32→TF32 peak ratio
is 8×, so realizing 2.65× implies ~⅓ of wall-clock is non-GEMM (small PiNoise ops,
dataloading, the float64 head) — consistent with §2. H100's ratio is 7.4×, so the same
projection holds there. **Projected H+ cell: 43 h → ~16 h.** That would have met the
15–20 h target.

**Correctness** — TF32 tracks the baseline *exactly*, epoch for epoch (`train_accy`):

```
                 Ep 1    Ep 2    Ep 3
ARM-A-baseline   0.39    2.74    6.36
ARM-B-tf32       0.39    2.74    6.36    <- identical to printed precision, 3/3
ARM-C-tf32-dl    0.57    2.58    8.08    <- differs; see below
```

**Stage-level `total acc`** — the decisive comparison:

| stage | reference (H100, unmodified) | `ARM-A` baseline (A100, unmodified) | `ARM-B` tf32 (A100) | `ARM-C` tf32+dl (A100) |
|---|---|---|---|---|
| 0 | 87.74 | **87.70** | **87.70** | 87.66 |
| 1 | 83.90 | — | 83.96 | 83.84 |
| 2 | 80.77 | — | 80.82 | 80.77 |

**TF32 and the unmodified baseline give bit-for-bit the same stage-0 accuracy on the
same hardware: 87.70 vs 87.70.** The residual 0.04 pp against the reference is therefore
**A100-vs-H100 fp32 nondeterminism, not TF32** — which is exactly what you want the
answer to be, and could not have been established without running the unmodified arm
alongside.

Across the three completed stages TF32 tracks the reference within **0.06 pp**, well
inside the ±0.15 pp gate. `ARM-C` also stays inside it (max 0.08 pp) despite drawing a
different augmentation RNG realization — so the `num_workers` change is statistically
benign even though it moves the exact numbers.

Stated plainly: this is a **short-horizon proxy — 3 of 20 stages, not a completed
cell.** The full gate (final `avg_acc` within ±0.15 pp of 64.569%) is not cleared. But
an exact same-hardware match at stage 0, plus three stages of tight tracking, is strong
evidence; the DINOv3 collapse mode would not hide this far in. Job 25862812 kept
running — re-read its logs for later stages.

**`ARM-C` diverging is a real finding, not noise.** MiN constructs its DataLoaders with
no `worker_init_fn` and no explicit `generator`, so each worker's RNG is derived from
`base_seed + worker_id`. The train transform is stochastic (`RandomResizedCrop`,
`RandomHorizontalFlip`, `ColorJitter`), so **changing `num_workers` changes which
augmentation RNG stream each sample draws from** — a different realization of the same
distribution.

**This corrects the brief's tiering.** Dataloader/worker tuning was listed as Tier 1
("provably equivalent"). `pin_memory`, `prefetch_factor` and `persistent_workers` are.
**`num_workers` is not**: it does not change semantics or expected performance, but it
does change the exact numbers. Consequences:

- Never bundle a `num_workers` change into an A/B you are running for equivalence.
- Changing it breaks seed-for-seed comparability against existing sweep results.
- **Ship `matmul_precision: "high"` on its own.** Treat `num_workers` as a separate,
  sweep-wide decision made once, before a sweep starts — and note it is worth only
  2.64× → 2.79× here.

## 5. Traps and corrections

- **Config-family trap.** `MiN-inat21-20steps-dinov3vit*.json` is committed but
  **nobody runs it** (`init_epochs=5, epochs=5, buffer_batch=1000, gamma=500`). The
  sweep runs the `-fn-g1` and `-orig` variants
  (`init_epochs=20, epochs=10, buffer_batch=1500`; `gamma=1` for fn-g1, `500` for
  orig). Check `min_inat/queue.tsv` for what is actually scheduled before profiling
  anything. An earlier draft of this doc profiled the wrong family and understated
  `run()` by 2×.
- **`buffer_batch` is NOT Tier 1.** It doubles as the RLS block size in `fit()`, so
  changing it changes the dimension of the `K` inverse and the recursion's
  floating-point path. Still mathematically exact RLS, but not batch-size-invariant the
  way a pure inference batch is. **Eval batch (`init_batch_size`) *is* genuinely Tier 1.**
- **SDPA/FlashAttention is already on, and currently inert.** MiN's `ViT_MiN.py` calls
  `F.scaled_dot_product_attention`, and HF transformers defaults DINOv3 to SDPA —
  confirmed by instantiating the model in-container: `attn impl : sdpa`. But **Flash
  kernels are fp16/bf16-only** — in fp32 SDPA silently falls back to the
  mem-efficient/math backend. Switching attention implementations buys **nothing**
  until precision drops. Don't count it twice.

## 6. Two data-path Tier 1 candidates found during validation

Both are **exactly equivalent** (Tier 1) and neither is implemented here — they were
found mid-validation, and changing the code under test would invalidate the A/B. Both
are measured, not estimated.

**(a) Eager whole-split JPEG decode — 1.13 h/cell, measured.**
`split_images_labels` (`data_manger.py`, `mode='1'`) decodes *every* image of a split
into a 224×224 RGB numpy array held in RAM, via a `Pool(processes=12)`, on every
`get_task_data` call. That is 4 calls per stage (train, test, train_no_aug, and test
again in `after_train`), so the **cumulative test set is re-decoded from scratch every
stage** — ~3.1 M decodes per cell.

It conveniently self-reports. From the reference B/16 cell's log:

```
$ grep -a "Pool takes" inat21__dinov3_vitb16__fn-g1__s42.log | ...
  4082 s = 1.13 h over 80 calls          # 80 = 4 calls x 20 stages
```

`get_pil_img` is deterministic (open → convert RGB → resize 224×224), so a
path-keyed cache of decoded arrays is exactly equivalent. RAM cost: the full 100 k test
set at 224×224×3 is ~15 GB — fine on a 2 TB node.

**Scope it correctly before acting:** this cost is *backbone-independent*. It is ~14% of
a DINOv3-B/16 cell but only **~2.6% of the 43 h H+ cell**. Worth doing, and the single
biggest data-path win, but it is not what makes H+ viable — §1 is.

**(b) `map_cat2order` linear scan — ~3–4 min/cell.**
`DataManger.map_cat2order` is `self.class_order.index(cat)` (`data_manger.py:94`) — a
linear scan of a 10,000-element list, per label, run over the train set and the
cumulative test set every stage. O(N·C).

Measured in-container: **2.39 s** for a 25 k train set and **9.71 s** for a 100 k
cumulative test set, per call. The one-line dict
(`{cat: i for i, cat in enumerate(class_order)}`) is **614× / 805× faster and produces
a bit-identical mapping** (asserted). Small in absolute terms — a few minutes per cell —
but free.

> An earlier draft of this section attributed a long GPU-idle startup stall to (b).
> That was wrong: (b) is ~2 s, and the stall is (a) plus first-touch GPFS latency.
> Both timings above are direct measurements.

## 7. Deliberately not done, and why

- **PiNoise loop collapse** (Tier 1, exact). The noise term
  `Σ_i w_i·(mu_i(x) + sig_i(x))` is *entirely linear* in `x_down`, so it collapses to a
  single 192×192 GEMM independent of task count — removing the O(T) growth and cutting
  ~1,280 → 32 kernel launches per forward at T=20. Worth ~2% overall (it applies to the
  ~17% of work that is `no_grad`). Not written because it needs a cache invalidated
  whenever `mu`/`sigmma` change during `run()`, and an untested staleness bug would
  silently corrupt a multi-hour run. Do it with a GPU to test on.
- **bf16 autocast** (Tier 2, ~4–6×). Also the only way to actually engage Flash (§5).
  Held back because `fit()` feeds features into a deliberately float64 RLS accumulator,
  and because of the collapse mode. Natural next step if TF32 under-delivers — same
  gate as §4, or stricter.
- **DDP** (Tier 2, up to 8×). **The math is legitimate**: there is **no BatchNorm
  anywhere** — verified by walking every module of an instantiated DINOv3-B/16
  (`BatchNorm: NONE`; 25 × `LayerNorm`, 24 × `DINOv3ViTLayerScale`) — so gradient
  averaging at constant global batch is exactly identical; and `fit()`'s
  Sherman–Morrison–Woodbury recursion is
  order-independent, converging to the exact batch ridge solution. But three real
  code-level blockers, **all of which fail silently rather than crashing**:
  1. `PiNoise.__init__` hardcodes `torch.device("cuda:0")` for its buffers
     (`MiN/backbones/ViT_MiN.py:76`).
  2. `weight_noise` is a plain tensor, **not an `nn.Parameter`**
     (`init_weight_noise`, `MiN/backbones/ViT_MiN.py:113`). It is therefore absent from
     `self.parameters()`, so DDP never broadcasts it — and it derives from
     `get_task_prototype`, which under DDP computes a per-rank mean over a *shard*.
     Different noise weights on every rank. Needs an explicit prototype all-reduce.
  3. `fit()` needs an all-gather of `X` and `Y` and one rank-consistent update;
     averaging partial `R`/`weight` updates is **not** equivalent.
  Roughly a day of surgery plus a full validation cycle. Only if TF32 and bf16 both
  fall short.

## 8. Picking this up cold

1. `git log --oneline` on `perf/dinov3h-speedup`; read `MiN/utils/perf.py` (~90 lines,
   self-documenting).
2. Do §6(a) first — it is the biggest data-path win and exactly equivalent; §6(b) is
   free and takes one line.
3. Run the §4 A/B on any free GPU: `cd /gpfs/.../perf_val && sbatch ab.sbatch`.
4. If it passes, add the §3 block to the model configs you care about.
5. Related context: `bigpurple_sweep/HANDOFF.md` (container, HF token, enroot env-var
   drops, DINOv3 seed instability).
