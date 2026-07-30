#!/usr/bin/env python3
"""Aggregate the Goal-1 DINOv3 paperhp sweep: per-cell mean +- sd, collapse
frequency, and reproduction verdict vs the paper's per-cell seed-1993 values.

Usage: aggregate_paperhp.py <results_dir>   # .../min_dinov3_paperhp/results/paperhp
"""
import json, glob, os, sys, statistics as st

# paper's per-cell avg_acc at seed 1993 (best_hyperparameters.csv)
PAPER = {
 ('cifar_100','dinov3_vitb16'): 84.81, ('cifar_100','dinov3_vitl16'): 82.31, ('cifar_100','dinov3_vith16'): 84.35,
 ('cub_200','dinov3_vitb16'): 91.14, ('cub_200','dinov3_vitl16'): 91.66, ('cub_200','dinov3_vith16'): 88.99,
 ('imagenet_a','dinov3_vitb16'): 76.10, ('imagenet_a','dinov3_vitl16'): 82.91, ('imagenet_a','dinov3_vith16'): 64.32,
 ('imagenet_r','dinov3_vitb16'): 88.83, ('imagenet_r','dinov3_vitl16'): 89.92, ('imagenet_r','dinov3_vith16'): 53.71,
 ('omnibenchmark','dinov3_vitb16'): 82.12, ('omnibenchmark','dinov3_vitl16'): 80.42, ('omnibenchmark','dinov3_vith16'): 53.88,
 ('vtab','dinov3_vitb16'): 95.06, ('vtab','dinov3_vitl16'): 95.42, ('vtab','dinov3_vith16'): 95.27,
}
COLLAPSE_TH = 40.0  # avg_acc %

def main(root):
    cells = {}
    for p in sorted(glob.glob(os.path.join(root, '*', 'MiN_*_seed*.json'))):
        j = json.load(open(p))
        key = (j['benchmark'], j['backbone'])
        cells.setdefault(key, {})[j['seed']] = j
    print(f"{'benchmark':<14} {'backbone':<14} {'n':>2} {'mean':>6} {'sd':>6} {'min':>6} {'max':>6} "
          f"{'ncol':>4} {'s1993':>6} {'paper':>6} {'d1993':>6}  verdict")
    rows = []
    for (bm, bb), seeds in sorted(cells.items()):
        accs = {s: 100*j['metrics']['avg_acc'] for s, j in seeds.items()}
        v = list(accs.values())
        m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0.0
        ncol = sum(1 for a in v if a < COLLAPSE_TH)
        p1993 = PAPER.get((bm, bb))
        a1993 = accs.get(1993)
        d = (a1993 - p1993) if (p1993 is not None and a1993 is not None) else None
        if d is None:
            verdict = 'PENDING'
        elif abs(d) <= 3:
            verdict = 'REPRODUCED'
        elif abs(d) <= 8:
            verdict = 'ROUGH'
        else:
            verdict = 'IRREPRODUCIBLE'
        rows.append(dict(benchmark=bm, backbone=bb, n=len(v), mean=m, sd=sd,
                         vmin=min(v), vmax=max(v), n_collapse=ncol,
                         acc_s1993=a1993, paper_s1993=p1993, delta_s1993=d,
                         verdict=verdict, per_seed=accs))
        print(f"{bm:<14} {bb:<14} {len(v):>2} {m:6.2f} {sd:6.2f} {min(v):6.2f} {max(v):6.2f} "
              f"{ncol:>4} {a1993 if a1993 is not None else float('nan'):6.2f} "
              f"{p1993:6.2f} {d if d is not None else float('nan'):+6.2f}  {verdict}")
    out = os.path.join(root, 'paperhp_summary.json')
    json.dump(rows, open(out, 'w'), indent=1)
    print(f"\nwrote {out}")

if __name__ == '__main__':
    main(sys.argv[1])
