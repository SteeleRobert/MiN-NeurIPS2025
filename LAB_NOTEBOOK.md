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

**Repo access note:** upstream zrcjessica/MiN-NeurIPS2025 is READ-only for
SteeleRobert; July commits 646e207/537aed9 were laptop-only. Forked to
SteeleRobert/MiN-NeurIPS2025 (`mine` remote), branch pushed there. GitHub SSH
from this network needs port 443 (`ssh.github.com`).

## 2026-07-30 — Goal-2: collapse anatomy from existing data (CPU-side)

All from the 120-run published-config sweep JSONs (local copies in
`bigpurple_sweep/results/`). Collapse := final avg_acc < 0.40 (27/120 runs:
v3-B 6, v3-H 21, v3-L 0, orig-B 0 — counts at this threshold; the rebut doc's
7/30 for v3-B used a different borderline rule).

**1. Collapse is early and total, not gradual.** Stage×task accuracy matrices
show the signature: in a collapsing run each new task is learned to ~95-100%
and everything earlier is wiped to ~0 at the NEXT stage (per-stage mean acc
≈ 1/t — pure recency). Non-collapsing runs on the same cell forget gradually
(T0: 100→98→94→…→56 over 20 stages). Bimodality confirmed at trajectory level;
there is no intermediate mode (borderline runs 0.39-0.43 avg are rare).

**2. The bifurcation happens at stages 1-3, always.** Onset (first stage with
mean acc < 50%) is 1-3 in all 27 collapsed runs — never later. Two variants:
(a) common: T0 retention at s1 already depressed (0.5-0.85 vs ≥0.95 in OK
runs); (b) deep-H variant (imagenet-r/v3-H): stage 1 fails the NEW task
(T1@s1=0, T0 kept), then pure-recency wipe-out from s2.

**3. A 3-stage probe predicts final collapse almost perfectly.** Mean acc at
stage 2 separates collapsed (max 0.641) from OK (min 0.631) with overlap only
among runs that are themselves borderline (avg 0.39-0.43). Rule of thumb:
mean@s2 < 0.7 ⇒ collapse trajectory. T0@s1 alone: r=0.70 with final avg_acc,
catches 17/27 at <0.8 with 1/93 false alarm, but misses late-onset cases.
⇒ Goal-3 can screen HP configs with 3-stage runs (~5-7× cheaper than full).

**4. Architecture reading (MiN.py, inc_net.py, ViT_MiN.py) — candidate
mechanisms.** The "noise" is deterministic: per block, per task, a zero-init
(mu_t, sigma_t) MLP pair on a fixed random down-projection; contributions are
softmax-reweighted by task-prototype cosine similarity (`init_weight_noise`)
at EVERY task — so all past noise contributions are rescaled ~1/(t+1) each
stage, and the shared feature stream drifts for all classes. The analytic head
(float64 RLS, R init I/γ, never reset; each task's data enters ~4× via
fit_epochs=3 + re_fit) only ever refits on the CURRENT task with one-hot
targets that are 0 on old classes — so any subspace overlap between drifted
new-task features and old-task features actively suppresses old-class logits.
Task 0 additionally trains all LayerNorms + per-block MLP branches
(init_unfreeze), i.e. the feature geometry itself is task-0-seed dependent.
Candidate mechanisms to discriminate (instrumented probe):
  H1 task-0 LayerNorm/MLP training distorts feature scale/geometry differently
     for B vs L (the frozen-backbone norm measurement ≈15 for both is NOT the
     operative quantity — post-task-0 norms are);
  H2 per-stage noise-branch drift moves old-class features off their fitted
     analytic weights (lr-sensitive — matches fn-g1-lr1e4 rescues and the
     low-lr Table-A2 cells);
  H3 RLS overwrite: new-task features overlapping old-task buffer subspace
     force W's old columns toward 0 (γ- and geometry-sensitive).

**5. Instrumentation built: `MiN/min_stab_probe.py`** — K-stage driver
mirroring the patched BaseTrainer sequence exactly, with wrapper-based
read-only snapshots after run/fit_fc/re_fit/update_noise: feature norms+drift
(L2/cos) on fixed probes, buffer act norms, per-block noise perturbation
ratios, W per-class column norms, R trace/diag/eig-quantiles, old-vs-new
logit stats, weight_noise mixture vectors. Probe loaders are shuffle=False /
num_workers=0 and measurements never touch global RNG. Validation protocol:
(a) --no-measure vs measured run must agree to float noise; (b) stage-0/1/2
accuracies must match the published full runs for same (config, seed).
Plan: 2-3-stage probes on cifar-100 × {v3-B, v3-L} × 5 seeds (published
config lr=0.001, γ=100) — 10 runs, ~4-6 GPU·h total — to discriminate H1-H3.

**Blocked on:** BigPurple SSH (VPN down since ~mid-session; monitor armed).
Pending when back: noise-loss trajectories from the 120 existing logs
(parse_noise_loss.sh), min_diag_conditioning.py retrieval, diag job
submission chained after 25969625.
