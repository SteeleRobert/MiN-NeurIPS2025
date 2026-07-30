# Lab notebook — original-HP seed-stability study (MiN × DINOv3)

Worker session for the three-goal study: (1) seed the paper's per-cell DINOv3 HP
winners across 5 seeds, (2) mechanism for the B-vs-L collapse asymmetry,
(3) seed-stable HPs inside the original knob space.

## 2026-07-30 — Goal-1 queue built and submitted

**HP recovery.** All 18 per-cell winners (6 benchmarks × DINOv3-B/L/H) recovered
from `best_hyperparameters.csv` (qz-compcont-learning @ REBUT/min) cross-checked
against each winner's result JSON. One cell had no recorded config anywhere:
**cifar-100 × v3-B** (run `af00f92dfdf2`, sweep `dinov3_min_focused`; result JSON
has no config dict, sweep manifest overwritten). Recovered by brute-forcing qz's
deterministic run_id (`sha256(sorted cfg)[:12]`) over the sweep grid — matches
only under the *pre-6bd7f98* grid (batch 512/512, buffer_batch 2000):
**lr=5e-05, γ=100, hidden_dim=96**. Procedure validated by exactly reproducing
two recorded run_ids (cub-200 `6eb614af33dd`, imagenet-r `c119cc4410e1`).
Full assumption list: `bigpurple_sweep/dinov3_paperhp/ASSUMPTIONS.md`.

Notable HP facts picked up along the way:
- The winners are heterogeneous per backbone: B cells mostly 256/1000 γ=75,
  L cells uniformly 128/500 γ=75 lr=5e-05, H cells 64/256 γ∈{25,50,75} lr≤5e-05.
  Two B cells have non-default hidden_dim (omni 48, vtab 384); cifar-B is 96.
- cifar × v3-H's winner is a distinct "boil" config: lr=1e-06, γ=10,
  init_lr=1e-4 — the original tuning already had to go to extreme low-lr
  territory to make H work on cifar (consistent with the low-lr rescue theme).
- imagenet-r × v3-H's "winner" is itself half-collapsed (avg_acc 53.7 at
  seed 1993) — the original sweep never found a working config for that cell.

**Sweep submitted.** Job **25969625** (48h, 8×H100), chained
`25969200 (min_dinov2 paperhp, running) → 25969090 (min_inat_lh) → 25969625`,
pinned sp-0014 per allocation rules. 90 runs (18 cells × seeds
{1993,42,7,2024,31337}), H-cells queued first. Machinery is the min_dinov2
pattern verbatim: never-mutated queue.tsv + atomic-mkdir claims + 3-attempt
worker judging success by `FINAL:` in logs; results to
`min_dinov3_paperhp/results/paperhp/` in seed-tagged compcont JSONs.
Expected ≈300 GPU·h ≈ 38h wall; if it doesn't drain, extend the chain with the
same queue/claims dirs (claims persist → no reruns).

**Expected start:** after ~11.5h of min_dinov2 remaining + up to 48h of
min_inat_lh — so roughly 1–2.5 days from now. Idle time goes to Goal-2 prep
(instrumentation design + CPU-side analysis of existing per-task histories).
