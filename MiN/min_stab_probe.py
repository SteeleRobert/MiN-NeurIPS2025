#!/usr/bin/env python3
"""Instrumented K-stage probe driver for the MiN seed-stability study (Goal 2).

Replicates BaseTrainer._train()'s exact call sequence (seeding, DataManger,
model construction, init_train, increment_train...) but stops after --stages
stages and records read-only measurements at phase boundaries:

  - backbone feature norms + drift (L2, cosine) on fixed probe batches
  - buffer activation norms
  - per-block noise-branch perturbation ratios ||out-in||/||in||
  - analytic-head W per-class column norms; R trace/diag stats (optionally
    full eigenspectrum with --eig)
  - old-vs-new class logit stats on probe batches
  - per-task test accuracy after each phase (probe-sized)
  - weight_noise mixture vectors after each update_noise()

Instrumentation is wrapper-based: model methods (fit_fc, re_fit, run,
update_noise) are wrapped so the ORIGINAL method runs unchanged and pure-read
measurement callbacks fire before/after. Measurement code never touches global
RNG (probe loaders use shuffle=False + num_workers=0; no randomized linalg on
the default generator), so the training trajectory is identical to an
uninstrumented run up to CUDA atomics noise. Validate with --no-measure:
the final metrics must match a measured run to float noise, and stage-0/1
accuracies must match the published full runs for the same (config, seed).

Usage:
  python MiN/min_stab_probe.py --base_configs ... --model_configs ... \
      --seed 42 --stages 2 --probe_out /path/out.jsonl [--eig] [--no-measure]
"""
import argparse
import copy
import json
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch
from torch.utils.data import DataLoader


def load_json(path):
    with open(path) as f:
        return json.load(f)


class ProbeRecorder(object):
    def __init__(self, out_path, model, datamanger, args, measure=True, eig=False):
        self.out = open(out_path, 'a')
        self.model = model          # MinNet wrapper
        self.net = model._network   # MiNbaseNet
        self.dm = datamanger
        self.args = args
        self.measure = measure
        self.eig = eig
        self.device = args['device']
        self.prev_feats = {}        # probe_name -> last feature snapshot (cpu)
        self.probes = {}            # probe_name -> (loader, class_list_orders)
        self.phase_ctr = 0

    def log(self, rec):
        rec['t'] = time.time()
        self.out.write(json.dumps(rec) + '\n')
        self.out.flush()

    # ---- probe construction (called between stages, no global RNG use) ----
    def add_probe(self, name, task_id, n_max=256, source="train_no_aug"):
        train_list, _, _ = self.dm.get_task_list(task_id)
        ds = self.dm.get_task_data(source=source, class_list=train_list)
        ds.labels = self.model.cat2order(ds.labels, self.dm)
        if len(ds) > n_max:
            # deterministic subset: every k-th item, no RNG
            idx = list(range(0, len(ds), max(1, len(ds) // n_max)))[:n_max]
            ds.images = [ds.images[i] for i in idx]
            ds.labels = np.asarray([ds.labels[i] for i in idx])
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
        orders = sorted(set(int(l) for l in ds.labels))
        self.probes[name] = (loader, orders)

    def add_test_probe(self, name, task_id):
        # classes of THIS task only == its train_list
        cls, _, _ = self.dm.get_task_list(task_id)
        ds = self.dm.get_task_data(source="test", class_list=cls)
        ds.labels = self.model.cat2order(ds.labels, self.dm)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
        orders = sorted(set(int(l) for l in ds.labels))
        self.probes[name] = (loader, orders)

    # ---- measurements ----
    @torch.no_grad()
    def snapshot(self, phase):
        if not self.measure:
            return
        self.phase_ctr += 1
        net = self.net
        net.eval()
        rec = {'phase': phase, 'phase_i': self.phase_ctr, 'cur_task': self.model.cur_task}

        # W / R stats
        W = net.weight  # (buffer, C)
        if W.numel():
            cn = torch.linalg.vector_norm(W, dim=0)
            rec['W_class_norms'] = [round(float(x), 5) for x in cn]
        R = net.R
        d = torch.diagonal(R)
        rec['R_trace'] = float(d.sum())
        rec['R_diag_mean'] = float(d.mean())
        rec['R_diag_min'] = float(d.min())
        rec['R_diag_max'] = float(d.max())
        rec['R_fro'] = float(torch.linalg.matrix_norm(R))
        if self.eig:
            ev = torch.linalg.eigvalsh(R)
            q = torch.tensor([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], dtype=ev.dtype, device=ev.device)
            rec['R_eig_quantiles'] = [float(x) for x in torch.quantile(ev, q)]

        # weight_noise mixtures (first/last block)
        try:
            nm = net.backbone.noise_maker
            wn0 = nm[0].weight_noise
            wnl = nm[-1].weight_noise
            rec['weight_noise_block0'] = None if wn0 is None else [float(x) for x in wn0]
            rec['weight_noise_blockL'] = None if wnl is None else [float(x) for x in wnl]
        except Exception:
            pass

        # probe passes with per-block perturbation hooks
        pert = {}
        handles = []
        try:
            nm = net.backbone.noise_maker
            def mk_hook(j):
                def hook(mod, inp, outp):
                    x = inp[0]
                    dnorm = torch.linalg.vector_norm(outp - x, dim=-1)
                    xnorm = torch.linalg.vector_norm(x, dim=-1).clamp_min(1e-9)
                    pert.setdefault(j, []).append(float((dnorm / xnorm).mean()))
                return hook
            for j in range(len(nm)):
                handles.append(nm[j].register_forward_hook(mk_hook(j)))
        except Exception:
            pass

        for name, (loader, orders) in self.probes.items():
            feats, acts, logit_list, targ_list = [], [], [], []
            for _, inputs, targets in loader:
                inputs = inputs.to(self.device)
                f = net.backbone(inputs)
                a = net.buffer(f)
                l = net.forward_fc(a)
                feats.append(f.detach().float().cpu())
                acts.append(a.detach().float().cpu())
                logit_list.append(l.detach().float().cpu())
                targ_list.append(targets)
            f = torch.cat(feats); a = torch.cat(acts); lo = torch.cat(logit_list)
            tg = torch.cat(targ_list).long()
            p = {'feat_norm': float(f.norm(dim=1).mean()),
                 'act_norm': float(a.norm(dim=1).mean())}
            if name in self.prev_feats:
                pf = self.prev_feats[name]
                p['feat_drift_l2'] = float((f - pf).norm(dim=1).mean())
                p['feat_drift_rel'] = float(((f - pf).norm(dim=1) / pf.norm(dim=1).clamp_min(1e-9)).mean())
                p['feat_cos'] = float(torch.nn.functional.cosine_similarity(f, pf, dim=1).mean())
            self.prev_feats[name] = f
            if lo.shape[1] > 0:
                pred = lo.argmax(dim=1)
                p['acc'] = float((pred == tg).float().mean())
                own = torch.tensor([o for o in orders if o < lo.shape[1]], dtype=torch.long)
                if len(own):
                    p['own_logit_max'] = float(lo[:, own].max(dim=1).values.mean())
                    tgc = tg.clamp_max(lo.shape[1] - 1)
                    p['true_logit'] = float(lo.gather(1, tgc.unsqueeze(1)).mean())
                other = torch.tensor([c for c in range(lo.shape[1]) if c not in set(orders)], dtype=torch.long)
                if len(other):
                    p['other_logit_max'] = float(lo[:, other].max(dim=1).values.mean())
            rec['probe_%s' % name] = p

        for h in handles:
            h.remove()
        if pert:
            rec['noise_pert_by_block'] = {str(j): round(sum(v) / len(v), 6) for j, v in sorted(pert.items())}
        self.log(rec)


def install_wrappers(model, rec):
    """Wrap MinNet phase methods; originals run unchanged, snapshots fire around them."""
    for name in ('fit_fc', 're_fit', 'run'):
        orig = getattr(model, name)
        def wrapped(*a, __orig=orig, __name=name, **kw):
            out = __orig(*a, **kw)
            rec.snapshot('after_%s_task%d' % (__name, model.cur_task))
            return out
        setattr(model, name, wrapped)
    net = model._network
    orig_un = net.update_noise
    def wrapped_un(*a, **kw):
        out = orig_un(*a, **kw)
        rec.snapshot('after_update_noise_task%d' % model.cur_task)
        return out
    net.update_noise = wrapped_un


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_configs', required=True)
    ap.add_argument('--model_configs', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--stages', type=int, default=2)
    ap.add_argument('--probe_out', required=True)
    ap.add_argument('--eig', action='store_true')
    ap.add_argument('--no-measure', dest='measure', action='store_false')
    opts = ap.parse_args()

    args = load_json(opts.base_configs)
    args.update(load_json(opts.model_configs))
    args['seed'] = opts.seed

    # mirror the patched BaseTrainer._train() exactly
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    devs = [torch.device('cpu') if d == -1 else torch.device('cuda:%s' % d) for d in args['device']]
    args['device'] = devs[0]
    args['gpu_ids'] = None
    args.setdefault('save_all_checkpoint', False)

    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s [probe] %(message)s')

    from data_process.data_manger import DataManger
    from utils.factory import get_model

    datamanger = DataManger(args['dataset'], args['device'], args)
    model = get_model(args, logging)

    rec = ProbeRecorder(opts.probe_out, model, datamanger, args,
                        measure=opts.measure, eig=opts.eig)
    rec.log({'phase': 'meta', 'args': {k: str(v) for k, v in args.items()},
             'seed': opts.seed, 'stages': opts.stages,
             'class_order': list(map(int, datamanger.class_order)),
             'measure': opts.measure})
    if opts.measure:
        rec.add_probe('t0_train', 0)
        rec.add_test_probe('t0_test', 0)
    install_wrappers(model, rec)

    history = []
    model.init_train(data_manger=datamanger)
    ev = model.after_train(data_manger=datamanger)
    history.append(dict(ev['all_task_accy']))
    rec.log({'phase': 'stage_end', 'stage': 0, 'task_accs': history[-1]})

    for i in range(1, opts.stages):
        if opts.measure:
            rec.add_probe('t%d_train' % i, i)
            rec.add_test_probe('t%d_test' % i, i)
        model.increment_train(data_manger=datamanger)
        ev = model.after_train(data_manger=datamanger)
        history.append(dict(ev['all_task_accy']))
        rec.log({'phase': 'stage_end', 'stage': i, 'task_accs': history[-1]})

    rec.log({'phase': 'final', 'history': history})
    print('PROBE_FINAL: ' + json.dumps(history))


if __name__ == '__main__':
    main()
