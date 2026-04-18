#!/usr/bin/env python3
"""
cache_embeddings.py — Cache post-noise MiN embeddings after each task T.

After each task T, all test images seen so far are passed through the frozen
backbone + all T noise modules (PiNoise).  The resulting CLS-token vectors are
the "effective" feature space the classifier sees at that stage of training.

Usage
-----
    cd /home/jessicaz/MiN-NeurIPS2025
    python MiN/cache_embeddings.py \
        --base_configs  MiN/configs/base_configs/imagenetr_ease.json \
        --model_configs MiN/configs/model_configs/MiN-inr-10steps.json \
        --output_dir    embeddings/

Output layout
-------------
    {output_dir}/{dataset}/{backbone_type}/task_{T:02d}.npz

Each .npz contains:
    features     float32 [N, D]   CLS-token after all T noise modules
    labels       int64   [N]      contiguous class indices (0 … K-1)
    class_order  int64   [C]      shuffled class order used by DataManger
                                  (label i corresponds to original class class_order[i])
    task_id      int              task index T (0-based)
"""

import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Make MiN's internal modules importable ────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from data_process.data_manger import DataManger
from utils.factory import get_model
from trainer.BaseTrainer import _set_device, print_args


# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _extract_embeddings(
    network,           # MiNbaseNet instance  (model._network)
    data_manger: DataManger,
    task_id: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features [N,D], labels [N]) for cumulative test set at task T.

    Features are the CLS-token after the backbone and all T+1 noise modules.
    Labels are remapped to contiguous order (0 … K-1) matching training targets.
    """
    _, test_class_list, _ = data_manger.get_task_list(task_id)
    test_set = data_manger.get_task_data(source="test", class_list=test_class_list)

    # Remap raw dataset class IDs → contiguous order indices (same as MinNet.cat2order)
    test_set.labels = [data_manger.map_cat2order(lbl) for lbl in test_set.labels]

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    network.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device)
            # extract_feature → backbone(x) → CLS-token [B, D]
            # backbone.forward() applies all currently-registered noise modules
            feats = network.extract_feature(inputs)
            all_features.append(feats.cpu().float().numpy())
            all_labels.append(np.asarray(targets))

    features = np.concatenate(all_features, axis=0)   # [N, D]
    labels   = np.concatenate(all_labels,   axis=0)   # [N]
    return features, labels


def _save_embeddings(
    output_dir: str,
    dataset: str,
    backbone_type: str,
    task_id: int,
    features: np.ndarray,
    labels: np.ndarray,
    class_order: list,
) -> str:
    task_dir = os.path.join(output_dir, dataset, backbone_type)
    os.makedirs(task_dir, exist_ok=True)
    out_path = os.path.join(task_dir, f"task_{task_id:02d}.npz")
    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        class_order=np.asarray(class_order, dtype=np.int64),
        task_id=np.int64(task_id),
    )
    return out_path


def _already_cached(output_dir: str, dataset: str, backbone_type: str, task_id: int) -> bool:
    path = os.path.join(output_dir, dataset, backbone_type, f"task_{task_id:02d}.npz")
    return os.path.isfile(path)


# ─────────────────────────────────────────────────────────────────────────────

def run_caching(args: dict, output_dir: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(
        output_dir, "logs", args["dataset"], args.get("backbone_type", "unknown"), now
    )
    os.makedirs(log_dir, exist_ok=True)
    _setup_logging(os.path.join(log_dir, "cache_embeddings.log"))

    _set_device(args)
    device: torch.device = args["device"]
    print_args(args)

    logging.info("Output directory: %s", output_dir)
    logging.info("Embedding cache will be written to: %s/%s/%s/",
                 output_dir, args["dataset"], args.get("backbone_type", "unknown"))

    data_manger = DataManger(args["dataset"], device, args)
    model       = get_model(args, logging)

    batch_size   = args.get("init_batch_size", 128)
    num_workers  = args.get("num_workers", 4)
    dataset      = args["dataset"]
    backbone     = args.get("backbone_type", "unknown")
    class_order  = list(data_manger.class_order)

    def _cache_task(task_id: int) -> None:
        if _already_cached(output_dir, dataset, backbone, task_id):
            logging.info("Task %02d already cached — skipping embedding extraction.", task_id)
            return
        logging.info("Extracting embeddings for task %02d …", task_id)
        features, labels = _extract_embeddings(
            model._network, data_manger, task_id, batch_size, num_workers, device
        )
        path = _save_embeddings(output_dir, dataset, backbone, task_id, features, labels, class_order)
        logging.info(
            "Task %02d: saved %d samples, shape %s → %s",
            task_id, len(labels), features.shape, path,
        )

    # ── Task 0 ────────────────────────────────────────────────────────────────
    model.init_train(data_manger=data_manger)
    model.after_train(data_manger=data_manger)
    _cache_task(0)

    # ── Tasks 1 … N ───────────────────────────────────────────────────────────
    for i in range(data_manger.task_size):
        model.increment_train(data_manger=data_manger)
        model.after_train(data_manger=data_manger)
        _cache_task(i + 1)

    logging.info("Embedding caching complete.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache MiN post-noise embeddings (backbone + all T noise modules) "
                    "for the cumulative test set after each task T."
    )
    parser.add_argument("--base_configs",  required=True,
                        help="Path to base config JSON (dataset, init_class, increment, …)")
    parser.add_argument("--model_configs", required=True,
                        help="Path to model config JSON (backbone_type, epochs, …)")
    parser.add_argument("--output_dir",    required=True,
                        help="Root directory for cached .npz files")
    parser.add_argument("--data_root",     default=None,
                        help="Override data_root from base config")
    parser.add_argument("--results_dir",   default=None,
                        help="Override results output directory (unused by caching, kept for parity)")
    args_ns = parser.parse_args()

    with open(args_ns.base_configs)  as f: base_cfg  = json.load(f)
    with open(args_ns.model_configs) as f: model_cfg = json.load(f)
    args = {**base_cfg, **model_cfg}   # model config wins on collision

    if args_ns.data_root   is not None: args["data_root"]   = args_ns.data_root
    if args_ns.results_dir is not None: args["results_dir"] = args_ns.results_dir

    run_caching(args, output_dir=args_ns.output_dir)


if __name__ == "__main__":
    main()
