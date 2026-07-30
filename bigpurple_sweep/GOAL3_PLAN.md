# Goal-3 plan (draft, 2026-07-30) — seed-stable HPs inside the original knob space

Constraint: only knobs the paper's method exposes — init_lr, init_epochs, lr
(=min_lr), epochs, batch/init_batch/buffer_batch, buffer_size, hidden_dim,
gamma. No feature normalization, no disabling the noise mechanism.

## Mechanism-guided structure (from Goal-2)

Collapse decomposes into two modes with different levers:

- **Mode A (task-0 blow-up; all B collapses, most H):** lever = `init_lr`
  (secondarily `init_epochs` — exposure time — and `init_batch_size` —
  gradient noise). Prediction P4: init_lr 3e-4/1e-4 eliminates it.
- **Mode B (incremental noise-training failure; H-only so far):** lever =
  `lr` (noise-lr). Rebut evidence: at lr 1e-4 noise ≈ no-op (stable); at
  1e-3 destabilizing. Danger: lr low enough to be stable may also make the
  noise mechanism inert — the honest question is whether there is a middle
  band where noise both helps and is seed-stable. Report either way.

The axes are approximately separable (different training phases), so search
is L-shaped, not a grid:

1. Fix lr at the per-cell winner; ladder init_lr ∈ {1e-3(pub), 3e-4, 1e-4}.
2. Fix init_lr at the stage-1 winner; ladder lr ∈ {winner, 3e-4, 1e-4, 3e-5}
   (only for cells still unstable — expected: H cells / Mode B).
3. If cells remain unstable, widen: init_epochs {5, 10}, init_batch × 2,
   epochs {5, 10}, gamma as tiebreak (γ was ruled out as a stability lever
   on B, but is untested as one at low lr).

## Screening protocol

- **3-stage probes** (validated on all 120 published runs: mean acc @ stage 2
  < ~0.7 ⇒ collapse trajectory; only borderline runs ambiguous). ~5-7×
  cheaper than full runs. 5 seeds per candidate.
- Candidates per target cell stage-1: 2 new init_lr points × 5 seeds = 10
  probes/cell. Target cells = those Goal-1 finds unstable (expected: the B
  cells with lr 2e-4, all H cells, maybe more).
- **Confirmation:** full-length 5-seed runs for the chosen config per cell.
- Success per cell: 5-seed sd ≤ 2pp AND mean within ~2pp of the per-cell-best
  repaired number (rebut §4). Negative result acceptable with evidence:
  "no setting in the original space is seed-stable on cell X".

## Inputs still pending

- Goal-1 paperhp results (which cells are unstable under per-cell winners;
  per-cell mean±sd + collapse counts). ETA ~1.5 days.
- Diag probe results (25974300): validation tier (hook bit-exactness),
  gradient-norm carrier for B-vs-L, init_lr ladder (P4 direct test),
  lr ladder (drift scaling, P2). These calibrate stage-1 choices; if P4
  holds cleanly, stage-1 collapses to confirming init_lr=1e-4 per cell.

## Budget sketch (all on the sp-0014 chain, after 25974300)

- Stage-1 screening: ~6 cells × 10 probes ≈ 60 probes ≈ 25-35 GPU·h.
- Stage-2 confirmation: ~6 cells × 5 seeds full ≈ 60-90 GPU·h.
- Fits one 48h job with the same queue/claims machinery (probe rows and
  full-run rows can share a queue with a runner column, or two queues +
  two worker pools).
