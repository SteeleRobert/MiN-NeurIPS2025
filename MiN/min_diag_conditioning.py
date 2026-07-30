"""Diagnose the DINOv3 collapse: feature-scale + RLS conditioning, per backbone.

Run from the repo root inside the pytorch_min container:
    python3 MiN/min_diag_conditioning.py \
        --base_configs MiN/configs/base_configs/cifar_ease.json \
        --model_configs MiN/configs/model_configs/MiN-cifar-10steps-dinov3vitb16.json \
        --seed 7 --gammas 1,100,500,5000,50000 [--feat_norm]

Extracts task-0/1 train features with the frozen backbone, pushes them through the
real RandomBuffer, and simulates the exact RLS update from MiNbaseNet.fit(),
logging the condition number of the matrix inverted at each step.
Prints greppable "DIAG:" lines.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_process.data_manger import DataManger
from backbones.pretrained_backbone import get_pretrained_backbone
from utils.inc_net import RandomBuffer


def load_json(p):
    with open(p) as f:
        return json.load(f)


@torch.no_grad()
def extract(backbone, loader, device, max_samples):
    feats = []
    n = 0
    for _, img, _ in loader:
        f = backbone(img.to(device))
        feats.append(f.double().cpu())
        n += f.shape[0]
        if n >= max_samples:
            break
    return torch.cat(feats)[:max_samples]


def norm_stats(tag, X):
    n = X.norm(dim=-1)
    print(f"DIAG: {tag} norms mean={n.mean():.2f} median={n.median():.2f} "
          f"p99={n.quantile(0.99):.2f} max={n.max():.2f} dim={X.shape[-1]} n={X.shape[0]}",
          flush=True)
    # per-dim outliers: how heavy are the largest coordinates vs typical
    absmax = X.abs().amax(dim=0)
    print(f"DIAG: {tag} per-dim absmax: median={absmax.median():.2f} "
          f"top5={[round(v, 1) for v in absmax.topk(5).values.tolist()]}", flush=True)


@torch.no_grad()
def rls_sim(H, gamma, device, chunk=500):
    """Replicate MiNbaseNet.fit()'s R updates, tracking conditioning."""
    d = H.shape[1]
    R = torch.eye(d, dtype=torch.double, device=device) / gamma
    conds = []
    for i in range(0, H.shape[0], chunk):
        X = H[i:i + chunk].to(device)
        M = torch.eye(X.shape[0], dtype=torch.double, device=device) + X @ R @ X.T
        conds.append(torch.linalg.cond(M).item())
        K = torch.inverse(M)
        R -= R @ X.T @ K @ X @ R
    # R should stay symmetric PSD; asymmetry signals accumulated numerical error
    asym = (R - R.T).abs().max().item()
    diag_min = R.diagonal().min().item()
    print(f"DIAG: gamma={gamma} cond_first={conds[0]:.3e} cond_max={max(conds):.3e} "
          f"cond_last={conds[-1]:.3e} R_asym={asym:.3e} R_diag_min={diag_min:.3e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_configs', required=True)
    ap.add_argument('--model_configs', required=True)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--gammas', default='1,100,500,5000,50000')
    ap.add_argument('--feat_norm', action='store_true')
    ap.add_argument('--max_samples', type=int, default=2500)
    cli = ap.parse_args()

    args = {**load_json(cli.base_configs), **load_json(cli.model_configs)}
    args['seed'] = cli.seed
    device = 'cuda'

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    dm = DataManger(args['dataset'], device, args)
    train_list, _, _ = dm.get_task_list(0)
    print(f"DIAG: backbone={args['backbone_type']} seed={cli.seed} "
          f"task0_classes={train_list}", flush=True)
    ds = dm.get_task_data(source='train_no_aug', class_list=train_list)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    backbone = get_pretrained_backbone(args).to(device).eval()
    # The PiNoise modules are only initialized by the trainer; for scale
    # diagnostics we want the clean pre-noise features, so bypass them.
    if hasattr(backbone, 'noise_maker'):
        backbone.noise_maker = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in backbone.noise_maker])
    X = extract(backbone, loader, device, cli.max_samples)
    norm_stats('raw_features', X)

    buffer = RandomBuffer(in_features=X.shape[1], buffer_size=args['buffer_size'],
                          device=device, feat_norm=cli.feat_norm)
    H = []
    for i in range(0, X.shape[0], 256):
        H.append(buffer(X[i:i + 256].to(device)).cpu())
    H = torch.cat(H)
    norm_stats('buffer_acts' + ('_featnorm' if cli.feat_norm else ''), H)

    for gamma in [float(g) for g in cli.gammas.split(',')]:
        rls_sim(H, gamma, device, chunk=args.get('buffer_batch', 500))

    print("DIAG: done", flush=True)


if __name__ == '__main__':
    main()
