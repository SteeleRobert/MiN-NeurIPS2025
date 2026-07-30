#!/usr/bin/env python3
"""Generate the DINOv3 'paperhp' sweep: the paper's per-cell HP winners
(best_hyperparameters.csv in qz-compcont-learning, REBUT/min branch) baked into
MiN model configs, x 5 seeds.

Mirrors the dinov2 paperhp convention exactly: start from the stock per-benchmark
stem config (the paper's original), override backbone_type + the HPs the qz sweep
recorded for that (benchmark, backbone) winner, leave every other field stock.

Empty CSV fields = stock value, per qz's _load_model_defaults semantics
(kanerva_sdm/classifiers/min_learner.py) -- the qz harness loaded the MiN stem
config as defaults and applied only the min_* overrides.

The cifar-100/dinov3_vitb16 winner (run af00f92dfdf2, sweep dinov3_min_focused)
has no config in its result JSON or the (overwritten) sweep manifest. Its HPs
were recovered by brute-forcing make_run_id() (sha256 of the sorted config dict,
kanerva_sdm/cli/sweep.py) over the sweep grid; the pre-6bd7f98 grid
(min_batch_size=512, min_init_batch_size=512, min_buffer_batch=2000) matches:
lr=5e-05, gamma=100, hidden_dim=96. The procedure was validated by exactly
reproducing the run_ids of two cells whose configs ARE recorded
(cub-200 6eb614af33dd, imagenet-r c119cc4410e1).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_OUT = os.path.join(HERE, 'configs')
os.makedirs(CFG_OUT, exist_ok=True)

SEEDS = [1993, 42, 7, 2024, 31337]

# Stock stem configs (identical to the -dinov3vitb16 variants on every field we
# fall back to; verified 2026-07-30 against main@646e207).
STOCK = {
    'cifar-100':     dict(stem='MiN-cifar-10steps', base='cifar_ease',
                          init_epochs=10, batch_size=64,  buffer_batch=500,  gamma=100),
    'imagenet-r':    dict(stem='MiN-inr-10steps', base='imagenetr_ease',
                          init_epochs=10, batch_size=128, buffer_batch=1000, gamma=500),
    'imagenet-a':    dict(stem='MiN-imageneta', base='imageneta_ease',
                          init_epochs=10, batch_size=128, buffer_batch=1000, gamma=500),
    'cub-200':       dict(stem='MiN-cub-10steps', base='cub_ease',
                          init_epochs=10, batch_size=128, buffer_batch=1000, gamma=100),
    'omnibenchmark': dict(stem='MiN-omni-10steps', base='omnibenchmark_ease',
                          init_epochs=20, batch_size=128, buffer_batch=1500, gamma=500),
    'vtab':          dict(stem='MiN-vtab-5steps', base='vtab_ease',
                          init_epochs=10, batch_size=128, buffer_batch=1000, gamma=100),
}

BB_SUFFIX = {'dinov3_vitb16': 'dinov3vitb16',
             'dinov3_vitl16': 'dinov3vitl16',
             'dinov3_vith16': 'dinov3vith16'}

# The 18 per-cell winners. Keys are MiN config field names. Source: qz
# best_hyperparameters.csv (min rows, dinov3 backbones) cross-checked against
# each winner's result JSON config dict; cifar/v3-B hash-recovered (see docstring).
CELLS = {
 ('cifar-100','dinov3_vitb16'):  dict(batch_size=512, init_batch_size=512, buffer_batch=2000,
                                      lr=5e-05, gamma=100, hidden_dim=96,
                                      sweep='dinov3_min_focused', run_id='af00f92dfdf2',
                                      note='hash-recovered (no config in JSON/manifest)'),
 ('cifar-100','dinov3_vitl16'):  dict(batch_size=128, init_batch_size=128, buffer_batch=500,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='all_classes_min_dinov3l', run_id='fddf45396a44'),
 ('cifar-100','dinov3_vith16'):  dict(batch_size=96, init_batch_size=96, buffer_batch=1000,
                                      lr=1e-06, gamma=10, hidden_dim=192,
                                      init_lr=0.0001, epochs=10, buffer_size=16384,
                                      sweep='min_dinov3vith16_cifar100_boil', run_id='289baabd9b50'),
 ('cub-200','dinov3_vitb16'):    dict(batch_size=256, init_batch_size=256, buffer_batch=1000,
                                      lr=0.0002, gamma=75, hidden_dim=192,
                                      sweep='dinov3_min_focused', run_id='6eb614af33dd'),
 ('cub-200','dinov3_vitl16'):    dict(batch_size=128, init_batch_size=128, buffer_batch=500,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='all_classes_min_dinov3l', run_id='5b8357f64734'),
 ('cub-200','dinov3_vith16'):    dict(batch_size=64, init_batch_size=64, buffer_batch=256,
                                      lr=1e-05, gamma=25, hidden_dim=192,
                                      sweep='all_classes_min_dinov3h_sweep', run_id='ca3b6a24447b'),
 ('imagenet-a','dinov3_vitb16'): dict(batch_size=256, init_batch_size=256, buffer_batch=1000,
                                      lr=0.0002, gamma=75, hidden_dim=192,
                                      sweep='dinov3_min_remaining', run_id='81636b1fe740'),
 ('imagenet-a','dinov3_vitl16'): dict(batch_size=128, init_batch_size=128, buffer_batch=500,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='all_classes_min_dinov3l', run_id='a985397d7e51'),
 ('imagenet-a','dinov3_vith16'): dict(batch_size=64, init_batch_size=64, buffer_batch=256,
                                      lr=1e-05, gamma=50, hidden_dim=192,
                                      sweep='all_classes_min_dinov3h_sweep', run_id='126c1991af0e'),
 ('imagenet-r','dinov3_vitb16'): dict(batch_size=256, init_batch_size=256, buffer_batch=1000,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='dinov3_min_focused', run_id='c119cc4410e1'),
 ('imagenet-r','dinov3_vitl16'): dict(batch_size=128, init_batch_size=128, buffer_batch=500,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='all_classes_min_dinov3l', run_id='87a04be3a554'),
 ('imagenet-r','dinov3_vith16'): dict(batch_size=64, init_batch_size=64, buffer_batch=256,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='all_classes_min_dinov3h', run_id='f912d4b7b518'),
 ('omnibenchmark','dinov3_vitb16'): dict(batch_size=256, init_batch_size=256, buffer_batch=1000,
                                      lr=5e-05, gamma=50, hidden_dim=48,
                                      sweep='omni_min_sweep', run_id='3467206b03bf'),
 ('omnibenchmark','dinov3_vitl16'): dict(batch_size=128, init_batch_size=128, buffer_batch=500,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='all_classes_min_dinov3l', run_id='5ba8ece594d7'),
 ('omnibenchmark','dinov3_vith16'): dict(batch_size=64, init_batch_size=64, buffer_batch=256,
                                      lr=5e-05, gamma=50, hidden_dim=192,
                                      sweep='all_classes_min_dinov3h_sweep', run_id='3677ab51f933'),
 ('vtab','dinov3_vitb16'):       dict(batch_size=256, init_batch_size=256, buffer_batch=1000,
                                      lr=0.0001, gamma=75, hidden_dim=384,
                                      sweep='dinov3_min_remaining', run_id='f1befbc81767'),
 ('vtab','dinov3_vitl16'):       dict(batch_size=128, init_batch_size=128, buffer_batch=500,
                                      lr=5e-05, gamma=75, hidden_dim=192,
                                      sweep='all_classes_min_dinov3l', run_id='40ff06a85884'),
 ('vtab','dinov3_vith16'):       dict(batch_size=64, init_batch_size=64, buffer_batch=256,
                                      lr=5e-05, gamma=25, hidden_dim=192,
                                      sweep='all_classes_min_dinov3h_sweep', run_id='8a0c2392da30'),
}

META_KEYS = ('sweep', 'run_id', 'note')

def stock_config(bm, backbone):
    s = STOCK[bm]
    return {
        "model": "min", "input_size": 224, "device": ["0"],
        "backbone_type": backbone, "pretrained": True,
        "memery_size": 0, "memery_per_class": 0,
        "optimizer_type": "sgd", "scheduler_type": "step",
        "init_epochs": s['init_epochs'], "init_lr": 0.001,
        "init_batch_size": s['batch_size'], "init_weight_decay": 0.0005,
        "epochs": 10, "lr": 0.001, "batch_size": s['batch_size'],
        "buffer_batch": s['buffer_batch'], "fit_epochs": 3,
        "weight_decay": 0.0005, "num_workers": 4, "hidden_dim": 192,
        "buffer_size": 16384, "gamma": s['gamma'],
    }

queue = []
prov = {}
for (bm, bb), hp in CELLS.items():
    cfg = stock_config(bm, bb)
    overrides = {k: v for k, v in hp.items() if k not in META_KEYS}
    cfg.update(overrides)
    name = f"{STOCK[bm]['stem']}-{BB_SUFFIX[bb]}-paperhp"
    with open(os.path.join(CFG_OUT, name + '.json'), 'w') as f:
        json.dump(cfg, f, indent=1)
        f.write('\n')
    prov[name] = {'benchmark': bm, 'backbone': bb,
                  'qz_sweep': hp['sweep'], 'qz_run_id': hp['run_id'],
                  'overrides': overrides, 'note': hp.get('note', '')}
    for seed in SEEDS:
        rid = f"{bm}__{bb}__paperhp__s{seed}"
        queue.append((rid, STOCK[bm]['base'], name, bm, bb, seed, 'paperhp',
                      # sort key: H first (slowest), then L, then B
                      {'dinov3_vith16': 0, 'dinov3_vitl16': 1, 'dinov3_vitb16': 2}[bb]))

queue.sort(key=lambda r: (r[7], r[3], r[5]))
with open(os.path.join(HERE, 'queue.tsv'), 'w') as f:
    for r in queue:
        f.write('\t'.join(str(x) for x in r[:7]) + '\n')

with open(os.path.join(HERE, 'provenance.json'), 'w') as f:
    json.dump(prov, f, indent=1)
    f.write('\n')

print(f"wrote {len(prov)} configs, {len(queue)} queue rows")
