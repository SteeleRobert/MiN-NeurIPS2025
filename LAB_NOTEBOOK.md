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
