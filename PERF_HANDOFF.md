# DINOv3-H+/16 iNat21 speedup — handoff

Branch `perf/dinov3h-speedup` (local commit `05a709f`). **Nothing here has run on a
GPU.** The author had no route to BigPurple (VPN down; `bigpurple.nyumc.org` does not
resolve, `nyu-gpu` proxies through it) and no local CUDA. Treat every number below as
analytical and every code change as unvalidated.

## The finding

A DINOv3-H+/16 iNat21 cell is **~8.1 ZFLOP**. Over the observed 43 h that is
**~52 TFLOPS sustained**.

| H100 SXM peak | TFLOPS | ratio to observed |
|---|---|---|
| FP32, no tensor cores | 67 | **0.78×** |
| TF32 tensor core | ~495 | 0.10× |
| BF16 tensor core | ~990 | 0.05× |

The run is at ~78% of *FP32 non-tensor-core* peak. It is not inefficient — it is
running an 840M-parameter ViT with the tensor cores switched off. Verified by grep:
no `autocast`, `bfloat16`, `float16`, `allow_tf32`, `torch.compile`, or
`channels_last` anywhere in `MiN/`. PyTorch has defaulted
`torch.backends.cuda.matmul.allow_tf32 = False` since 1.12; the container is 2.4.0.

## Where the time goes

Per incremental task, 25,000 train images, in forward-equivalents
(`epochs=10`, `fit_epochs=3`; backward ≈ 2× forward because trainable noise params
sit in *every* block, so the graph spans all 32):

| Phase | fwd-equiv | share |
|---|---|---|
| `run()` — 10 ep × (1 no_grad fwd + 1 grad fwd + bwd) | 1,000,000 | **83.2%** |
| `fit_fc` — 3 epochs, no_grad | 75,000 | 6.2% |
| `after_train` eval — cumulative, 5k→100k, avg 52.5k | 52,500 | 4.4% |
| `get_task_prototype` ×2 | 50,000 | 4.2% |
| `re_fit` — 1 pass | 25,000 | 2.1% |
| **total** | **~1.20 M** | |

×20 tasks ≈ **24 M forward-equivalents/cell** at ~337 GFLOP each (201 tokens).

Two corollaries:
- **Eval is dataloader-bound, not compute-bound** — 15% of wall-clock but 4.4% of
  FLOPs, at `num_workers=4` with 100k images in the last stage.
- **The float64 ridge head is not a bottleneck** — ~3.2 PFLOP FP64/cell ≈ 64 s total.
  Do not "optimize" it; it is deliberately float64 and the RLS recursion is exact.

## What is in this commit

`MiN/utils/perf.py`, wired into `BaseTrainer._train` and `MiN.MinNet`. Both hooks are
**opt-in and no-ops on every existing config**:

```jsonc
{
  "matmul_precision": "high",        // TF32. default "highest" = unchanged fp32
  "cudnn_benchmark": true,           // safe: fixed shapes after batch 1
  "dataloader": {
    "num_workers": 12,               // was 4
    "persistent_workers": true,
    "prefetch_factor": 4,
    "pin_memory": true
  }
}
```

`"high"` keeps fp32 storage *and fp32 accumulation*; it only rounds GEMM inputs from a
23- to a 10-bit mantissa. That is a real floating-point change, hence the gate below.

Logic is unit-tested without torch (default → `{'num_workers': 4}` exactly; `workers=0`
correctly suppresses `persistent_workers`/`prefetch_factor`, which PyTorch rejects;
invalid precision raises).

## Validation gate — REQUIRED before any H cell uses this

1. Run **DINOv3-B/16 seed 42 on iNat21** with `matmul_precision: "high"`.
2. Compare against min-sweep's unmodified run of that exact cell.
3. Accept only if final `avg_acc` is within **±0.15 pp** and the per-task history
   tracks closely.
4. If no time for the full cell: 3-stage truncated run, unmodified vs optimized, same
   seed — and **say explicitly that is what was done**.

This matters more than usual here: the sweep HANDOFF documents MiN + DINOv3 as
seed-unstable with a bimodal collapse mode, and `dino_backbone.py` records gradient
explosions from input-statistic mismatch. This is a numerically fragile configuration.
A TF32 result that shifts more than the tolerance is **rejected, not rationalized**.

## Projection

If TF32 delivers the expected 2.5–3.5× on the ~90% of work that is GEMM, and the
dataloader fix recovers most of the eval gap, **43 h → ~15–18 h**. That meets the
target. It has not been measured.

## Deliberately NOT done

- **PiNoise loop collapse** (Tier 1, exact). The noise term
  `Σ_i w_i·(mu_i(x) + sig_i(x))` is entirely linear in `x_down`, so it equals a single
  192×192 GEMM independent of task count — removing the O(T) growth and cutting
  1,280 → 32 kernel launches per forward at T=20. Worth only ~2% overall (it applies to
  the 17% of work that is no_grad). Not written, because it needs a cache invalidated
  whenever `mu`/`sigmma` change during `run()`, and an untested staleness bug would
  silently corrupt a 15-hour run. Do it *after* TF32 lands, with a GPU to test on.
- **bf16 autocast** (Tier 2, ~4–6×). Also the only way to actually engage the Flash
  kernel — SDPA is already the default path for both MiN's ViT and HF DINOv3, but
  Flash is fp16/bf16-only, so in fp32 it silently falls back to mem-efficient/math and
  contributes nothing. Held back because `fit()` feeds features into a deliberately
  float64 RLS accumulator, and the collapse mode above. Fallback if TF32 under-delivers.
- **DDP** (Tier 2, up to 8×). The math is legitimate: **no BatchNorm anywhere**
  (LayerNorm + LayerScale only), so gradient averaging at constant global batch is
  exactly identical; and `fit()`'s Sherman–Morrison–Woodbury recursion is
  order-independent, converging to the exact batch ridge solution. But three real
  code-level blockers, all of which would fail *silently*:
  1. `PiNoise.__init__` hardcodes `torch.device("cuda:0")` (`ViT_MiN.py:76`).
  2. `weight_noise` is a plain tensor, **not** an `nn.Parameter`
     (`init_weight_noise`, `ViT_MiN.py:113`) — so DDP never broadcasts it, and it is
     derived from `get_task_prototype`, which under DDP would be a per-rank mean over a
     shard. Different noise weights per rank. Needs an explicit prototype all-reduce.
  3. `fit()` needs an all-gather of X and Y and one rank-consistent update; averaging
     partial `R`/`weight` updates is *not* equivalent.
  Roughly a day of surgery plus a full validation cycle. Only if TF32 and bf16 both
  fall short.

## Correction to the original brief

Raising `buffer_batch` is **not** Tier 1. It doubles as the RLS block size in `fit()`,
so changing it changes the `K` inverse dimension and the recursion's floating-point
path. Still mathematically exact RLS, but not batch-size-invariant the way a pure
inference batch is. Eval batch (`init_batch_size`) *is* genuinely Tier 1.
