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

**6. Variant-stability mining (rebut_min_cells.csv, local).** On cifar/v3-B:
the γ-only ladder without feature norm is unstable at EVERY γ (g100…g500000:
sd 12-39, collapses at each), while L2 feature norm stabilizes at every γ
tried (0.1, 0.3, 1, 10, 100, 500: 0 collapses) — γ:XᵀX ratio alone is NOT the
stability factor. On cifar/v3-L: published is stable-ish (76.5±1.9), fn-g1
DEstabilizes (54.3±9.3), and fn-g1-lr1e4 rescues to 93.04±0.07 — numerically
identical to noiseoff-fn-g1 (93.00±0.07); same pattern on omni (85.13≈85.01).
So at noise-lr 1e-4 the noise mechanism contributes ~nothing, and at the
published 1e-3 it is the destabilizer once features are normalized. Even on
v3-B the residual variance under fn tracks noise-lr (fn-g1 sd 4.61 vs
fn-g1-lr3e4 sd 0.12).

**Working mechanism hypothesis (pre-instrumentation).** Collapse is driven by
noise-adapter feature drift per stage, whose magnitude is set by the noise
learning rate × gradient scale relative to feature magnitude in the buffer
projection; the analytic head cannot follow drift because it only refits on
current-task data (and actively suppresses old-class logits on overlapping
subspace). γ mis-scale matters mainly through the logits1 term in the noise
loss (gradient scale), not RLS conditioning. B-vs-L asymmetry: candidate
carriers are (a) per-block gradient scale (12 vs 24 adapters), (b) feature
norm heterogeneity/anisotropy (mean norms match, tails may not — B per-dim
absmax 1.91 vs L 1.69), (c) task-0 LayerNorm training distortion. Probe runs
will discriminate.

**Registered predictions (falsifiable, before Goal-1 results):**
  P1. In the Goal-1 paperhp sweep, cells with min_lr ≤ 5e-05 will show few or
      no collapses across seeds; the B cells with lr = 2e-04 (cub-B, ina-B)
      are the most likely to be seed-unstable.
  P2. The probe will show per-stage old-probe feature drift in collapsing
      (B, seed) pairs ≫ non-collapsing pairs at the same HPs, visible already
      at stage 1, and drift magnitude will scale ~linearly with min_lr.
  P3. A "baseline + low min_lr" config (no feature norm, γ published) will be
      substantially more seed-stable than published on v3-B cifar/omni — the
      core Goal-3 candidate direction.

## 2026-07-30 (later) — Goal-2 BREAKTHROUGH: the bifurcation is at task 0

SSH restored. Chain REORDERED so the small diag job runs before the 48h
paperhp sweep: 25969200 (min_dinov2, running) → 25969090 (min_inat_lh) →
**25971966 (min_stab_diag, 12h)** → **25971981 (min_dinov3_paperhp, 48h)**.
(Old paperhp job 25969625 cancelled by me; diag dependency repointed to
25969090 via scontrol so it cannot jump ahead of min_inat_lh.)

**Task-0 training curves from the 120 published-run logs** (worker stdout
retains every tqdm epoch line via \r; `tr '\r' '\n'` recovers them):

- cifar/v3-B, ALL 5 seeds: normal convergence for 2-3 epochs (→83-92% train
  acc) then a **loss blow-up at epoch 3-4**. The one surviving seed (2024)
  dips 89→69 and recovers to ~87%; the 4 collapsing seeds crash to ~20-30%
  (≈ chance = 20%) and never recover (final 28-58%).
- cifar/v3-L, ALL 5 seeds: smooth, no dips, final 90-95%. Zero blow-ups.
- cifar/v3-H: blow-ups on 5/5 (epochs 2-7).
- Stage-0 EVAL is 100% in every run regardless — the 16384-dim analytic head
  interpolates 5 classes perfectly even on wrecked features, so the standard
  metrics are blind to the failure until task 1 arrives.

**Quantitative rule across all 120 runs** (task-0 per-epoch trajectories):
blow-up (>25pp drop from running max) AND no recovery (final >15pp below
peak) catches **21/27 collapses with 1/93 false alarms** (the false alarm,
omni-H s2024 @ 42.7 final, is itself borderline). The 6 uncaught collapses
are all v3-H on imagenet-a/vtab — a second, H-specific mode.

**Two collapse modes:**
- **Mode A (21/27; includes 6/6 of v3-B's collapses):** first-session
  (task-0) optimization blow-up while training LayerNorms + adapter MLPs +
  normal_fc with SGD-momentum at init_lr=1e-3. Features wrecked → analytic
  head goes recency-only from stage 1-2.
- **Mode B (6/27; v3-H only):** task-0 survives, but incremental noise
  training fails to learn new tasks (e.g. imagenet-r/v3-H: T1@s1=0), then
  pure-recency wipe-out. Likely the same instability surfacing in the
  incremental phase (lr=1e-3) instead.

**Answer shape for the B-vs-L puzzle:** v3-B sits at the edge of optimization
stability at init_lr=1e-3 (every cifar seed spikes at epoch 3-4; severity and
recovery are seed-dependent → bimodal outcome); v3-L is comfortably inside
the stable region (zero spikes anywhere). Static feature statistics can't
separate them because the operative quantity is optimization dynamics
(gradient scale / curvature of the LN+adapter system), not feature norms.
The instrumented probes (now with per-step per-group gradient norms and
LayerNorm-drift tracking) will identify the carrier.

**Goal-3 implication (registered as P4):** `init_lr` — explicitly inside the
allowed knob space ("init settings") — is the primary stabilizer for Mode A;
lowering it to 3e-4/1e-4 should eliminate B's blow-ups. The cifar/v3-H "boil"
winner already used init_lr=1e-4 (the original sweep found this without
knowing why). `lr` is the corresponding lever for Mode B. Diag queue extended
with an init_lr ladder (ilr3e4/ilr1e4 × 4 seeds, 3-stage probes): 44 rows.

**Evidence inventory for the milestone report:** task-0 curves (all 120),
blow-up rule stats, mode taxonomy, chain state. Probe-based gradient
evidence lands when 25971966 runs (~2.5 days).

## 2026-07-30 (night) — chain change by Robert's order (via manager)

min_inat_lh de-scoped and cancelled by Robert (9 stale claims in its queue if
resurrected). Manager cancelled my min_stab_diag 25971966 so the original-HP
sweeps complete first. New chain: **25971981 min_dinov3_paperhp (RUNNING —
Goal-1 ~2 days early)** → 25973957 min_vit_paperhp (manager's, original-ViT
per-cell paper HPs) → **25974300 min_stab_diag (resubmitted by me,
afterany:25973957)**. Standing rule: no chain reorders without checking with
the manager first. P4 remains registered and untested until diag runs.

## 2026-07-30 (evening) — chain progressed; DINOv2 paperhp baseline complete

min_dinov2 (25969200) drained clean: 30/30 done, 0 failures, ~3h wall.
min_inat_lh (25969090) now RUNNING on sp-0014. A new job appeared behind
mine: 25973957 `min_vit_paperhp` (another session's original-ViT paperhp
sweep), chained afterany:25971981 — no conflict.

**DINOv2 paperhp results (context for Goal-1's expected outcome):** per-cell
winners are seed-STABLE everywhere — cifar 90.75±0.84, cub 90.91±0.35,
ina 75.41±0.79, inr 87.79±0.38, omni 79.52±1.34, vtab 95.34±0.55; zero
collapses — despite noise-lr up to 1e-3. So the edge-of-stability pathology
is specific to DINOv3-B/H checkpoints, not to MiN's protocol per se, and not
to DINOv2 (nor DINOv3-L). Sharpens the Goal-1 question: do the DINOv3
per-cell winners (mostly lr 5e-05 but init_lr still 1e-3, except the cifar-H
boil cell at 1e-4) inherit Mode-A blow-up risk through init_lr?

**omni-B addendum (worker logs, init_epochs=20 — definitive):** every omni-B
seed except 31337 hits a dip/blow-up somewhere in epochs 8-14; the two
collapsed seeds (1993: 90.3%→6.5 at ep14; 7: 91.3%→31→3.7 at ep11) never
recover, the two dippers that recovered (2024, 42) survive. Blow-ups strike
at ANY epoch, late included — so init_epochs (omni stock = 20) also modulates
exposure. Confirms Mode A generalizes beyond cifar; survival == recovery.

## 2026-07-31 — second node joined the paperhp sweep

Robert/manager launched `min_d3php_sp10` (25991134, sp-0010, 2d limit) running
workers against the shared min_dinov3_paperhp queue — the atomic-mkdir claims
machinery absorbed it natively (no duplicate claims). 10/90 done, 0 failures,
16 GPUs active. ETA ~15-20h. (sp-0010 was allocated by Robert's side, not by
this worker — my jobs remain sp-0014-only per the resource rules.)

## 2026-07-31 — Goal-1 early H-tier results (27/90 done, 0 failures)

| cell | mean±sd | ncol | s1993 vs paper | verdict |
|---|---|---|---|---|
| cifar-H | 88.21±6.61 | 0 | 76.7 / 84.4 | ROUGH (stable!) |
| cub-L | 91.75±0.18 | 0 | 91.65 / 91.66 | REPRODUCED |
| cub-H | 73.03±36.92 | 1 | 89.9 / 89.0 | repro but lucky-seed |
| ina-H | 32.08±14.88 | 4 | 16.2 / 64.3 | IRREPRODUCIBLE |
| inr-H | 15.86±18.89 (n=3) | 3 | 5.1 / 53.7 | IRREPRODUCIBLE |
| vtab-H | 51.57±23.21 | 3 | 70.7 / 95.3 | IRREPRODUCIBLE |

Headline: **cifar-H — the only winner with init_lr=1e-4 — is the only stable
H cell** (P4 corroborated from independent data). ina/vtab-H paper values are
unreachable even by the best of 5 seeds (51.0 / 81.6 vs 64.3 / 95.3); caveat:
qz-harness vs MiN-pipeline test-split offsets apply to ina/inr (and partly
vtab) level comparisons, not to sd/collapse counts. Flagged to manager per
reporting rule 5.

## 2026-07-31 — de-scope directive (Robert, via manager)

Both cancellations intentional: Robert de-scoped everything except the
original-HP reruns ("I just want the original HPs to be rerun").
min_vit_paperhp cancelled while pending (hence absent from sacct);
min_stab_diag deferred INDEFINITELY pending Robert's explicit go — do not
resubmit. After the two dinov3 paperhp node jobs drain, cluster footprint
goes to zero. Goal-2 instrumented evidence (P4 direct test, gradient carrier,
hook validation) and Goal-3 screening compute are paused with it. Continuing:
Goal-1 aggregation + CPU-free analysis only. The Goal-1 report must include a
"what the diag would settle" paragraph (~10 GPU-h ask) for Robert's go/no-go.

## 2026-07-31 — GOAL-1 COMPLETE: 90/90, 0 failures. Per-cell verdict.

| cell | mean±sd | collapses | s1993 vs paper | verdict |
|---|---|---|---|---|
| cifar-B | 87.82±1.93 | 0/5 | 85.1 / 84.8 | REPRODUCED |
| cub-B | 90.79±0.22 | 0/5 | 91.0 / 91.1 | REPRODUCED |
| ina-B | 72.65±0.50 | 0/5 | 72.1 / 76.1 | ROUGH (stable; likely split offset) |
| inr-B | 89.48±0.34 | 0/5 | 89.8 / 88.8 | REPRODUCED |
| omni-B | 81.25±0.69 | 0/5 | 80.7 / 82.1 | REPRODUCED |
| vtab-B | 94.81±0.71 | 0/5 | 95.1 / 95.1 | REPRODUCED |
| cifar-L | 84.26±2.01 | 0/5 | 81.5 / 82.3 | REPRODUCED |
| cub-L | 91.80±0.20 | 0/5 | 91.7 / 91.7 | REPRODUCED |
| ina-L | 82.72±1.67 | 0/5 | 84.0 / 82.9 | REPRODUCED |
| inr-L | 91.26±1.03 | 0/5 | 92.5 / 89.9 | REPRODUCED |
| omni-L | 79.46±1.12 | 0/5 | 79.9 / 80.4 | REPRODUCED |
| vtab-L | 94.55±1.12 | 0/5 | 94.9 / 95.4 | REPRODUCED |
| cifar-H | 88.21±6.61 | 0/5 | 76.7 / 84.4 | ROUGH (stable — init_lr=1e-4 cell) |
| cub-H | 73.03±36.92 | 1/5 | 89.9 / 89.0 | repro at 1993, LUCKY-SEED cell |
| ina-H | 32.08±14.88 | 4/5 | 16.2 / 64.3 | IRREPRODUCIBLE |
| inr-H | 21.21±22.72 | 4/5 | 5.1 / 53.7 | IRREPRODUCIBLE |
| omni-H | 17.30±4.52 | 5/5 | 18.0 / 53.9 | IRREPRODUCIBLE (max 23.0 < paper) |
| vtab-H | 51.57±23.21 | 3/5 | 70.7 / 95.3 | IRREPRODUCIBLE |

**Headlines.**
1. Collapse frequency by backbone: **B 0/30, L 0/30, H 13/30.** The paper's
   per-cell winners fully stabilize B and L; H remains fundamentally
   seed-unstable except cifar-H.
2. The two stable-H facts point the same way: cifar-H is the only H winner
   with init_lr=1e-4 (0/5 collapse); every other H winner kept init_lr=1e-3
   with small batches (64/96) and collapses on 1-5 of 5 seeds.
3. **Mechanism wrinkle discovered by P1 failing:** B winners kept
   init_lr=1e-3 — same as the collapsing stock configs — yet 0/30 collapse.
   The only task-0-relevant knobs they changed: init_batch_size
   (64/128 → 256/512) and, in 2 cells, hidden_dim (192 → 96/48). By the Mode-A
   account (task-0 blow-up), the batch increase (2-8× fewer, less noisy steps)
   is the likely stabilizer — but this is exactly what the paused diag's
   ladder would disentangle (init_lr vs init_batch_size vs hidden_dim).
4. Prediction scoring (registered 2026-07-30): P1 HALF-WRONG both ways —
   lr=2e-4 B cells are stable (batch effect dominates), and low-lr H cells
   still collapse (init_lr=1e-3 + small batch dominates). The refined claim —
   task-0 knobs (init_lr, init_batch) decide Mode A; incremental lr decides
   Mode B — explains both misses and all 18 outcomes. P2/P3/P4 untested
   (diag paused).
5. Goal-3 status implied by Goal-1: for B and L the original tuning
   methodology ALREADY reaches seed-stability (sd ≤ 2pp on 11/12 cells;
   means within ~2pp of the repaired per-cell best on 8/12, ABOVE it on 3).
   Residual gaps vs repaired-best on cifar-L (-8.7pp), omni-L (-5.7), inr-L
   (-3.4) are where noise-lr-lowered variants beat the original space so far.
   For H, the open question is whether cifar-H-style init settings rescue the
   other 5 cells (P4's direct test — paused with the diag).

## 2026-07-31 — H-cell exact-replay (Robert-ordered coda), interim

Byte-fidelity established: replay YAML expands through qz's loader to dicts
EQUAL to the recorded configs; harness echoes all 4 recorded run_ids at load.
qz min path unchanged since winner commit 8c541bc; stem/vith16 config
fallback immaterial; same container, WDS shards, seed, GPU type. gpu_ids is
set but never consumed (no DataParallel) — multi-GPU visibility in May could
not have changed the math. Zero argument deviations.

- **vtab-H: 95.00 vs recorded 95.27 (−0.27pp) → recorded value CONFIRMED
  real.** A genuine lucky draw of a config that is seed-unstable (5-seed:
  51.6±23.2, 3/5 collapse).
- **ina-H: replays 10.50, 25.96, 13.56 vs recorded 64.32 — 3/3 fail, none
  close.** All three show depressed/blown-up task-0 training (final train acc
  11-47%; the first replay converged to ~62% then blew up at epoch 10/10 —
  the Mode-A fingerprint). At fixed (config, seed), the outcome is decided by
  CUDA nondeterminism at the task-0 knife edge; the recorded ≥64 outcome is
  at best a rare draw (0/3), at worst provenance-compromised. Fixed-seed
  spread (10.5-26.0) is comparable to cross-seed spread (16.2-51.0): class
  order is NOT the dominant randomness for this cell.
- inr-H (40-task qz protocol), omni-H: still running.

## 2026-08-01 — H-cell exact-replay COMPLETE. Verdict: run-level lottery.

All 4 originals + 4 repeats done before holder 25999938 hit its natural 30h
TIMEOUT (queue now empty; zero footprint). Raw JSONs in
bigpurple_sweep/h_replay/results/.

| cell | recorded | fixed-seed replays | verdict |
|---|--:|---|---|
| vtab-H | 95.27 | 95.00 | **CONFIRMED real** (−0.27pp) |
| ina-H | 64.32 | 10.50, 25.96, 13.56 | NOT REPLAYABLE — 3/3 fail, none close |
| inr-H | 53.71 | **63.31, 2.50, 3.54** | NOT REPLAYABLE — fixed-seed range spans 2.5→63.3; recorded value sits inside it |
| omni-H | 53.88 | 21.97 (n=1; holder timeout precluded repeats) | MATERIALLY OFF, same pattern |

**Read.** For ina/inr/omni-H the run-level outcome at byte-identical
(config, seed, harness, data, container, GPU model) is decided by CUDA
nondeterminism at the task-0/incremental knife edge — the fixed-seed spread
(inr: 2.5–63.3) is as large as the cross-seed spread. The recorded values are
best explained as favorable draws from this lottery, NOT as provenance
errors: inr's recorded 53.7 is inside our observed fixed-seed range, and the
mechanism (Mode-A blow-up; observed at epoch 10/10 in the first ina replay)
predicts exactly this sensitivity. vtab-H's number is a real, replayable run
— but of a config that is still a 5-seed lottery (51.6±23.2).
The H-row story is therefore STRONGER than "lucky seeds": for 3 of the 4
disputed cells, the published numbers are not reproducible even at fixed
seed; "the experiment" as recorded does not define its own outcome.

**Deviations from original args: NONE** (dict-equality + run_id hash + the
harness echoing recorded run_ids at load). Environmental deltas, both
immaterial: CUDA_VISIBLE_DEVICES pinning (single visible GPU vs 8 in May —
gpu_ids is computed but never consumed; no DataParallel) and
OMP_NUM_THREADS 16 vs 32 (CPU-side only).
