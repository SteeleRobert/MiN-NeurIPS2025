#!/usr/bin/env python3
"""
MiN Unlearning Experiments
===========================
Tests where task-specific knowledge lives in MiN and how effectively it
can be erased via three progressive ablation levels:

  A1 — Zero mixture weight     : suppress ω[u], renormalise others
  A2 — Remove task generator   : zero all P^u parameters (mu, sigmma)
  A3 — Remove generator + head : A2 + zero analytic classifier columns

Usage (from MiN-NeurIPS2025/ root):
    python MiN/unlearn.py \\
        --base_configs  MiN/configs/base_configs/cifar_ease.json \\
        --model_configs MiN/configs/model_configs/MiN-cifar-10steps.json \\
        --task_to_forget 0

Optional flags:
    --checkpoint PATH   Load a saved unlearn checkpoint instead of training.
    --save_checkpoint   Save the trained model to --output_dir after training.
    --output_dir DIR    Directory for plots/JSON (default: MiN/logs/unlearn/).
    --device DEVICE     Override device, e.g. "0" or "1" (default: from config).
    --seed SEED         Random seed override.
    --no_feature_probe  Skip the (slow) linear separability probe.
    --data_root PATH    Override data_root from base config.
"""

import sys
import os

# Add MiN/ to sys.path so relative imports from models/, utils/, etc. work.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Path to qz-compcont-learning's kanerva_sdm package (WdsDataset lives there).
_QZ_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "qz-compcont-learning", "src"))

# Map MiN dataset name → wds sub-directory name (under data_root/wds/).
_WDS_SUBDIR: dict = {
    "cifar224":      "cifar_100",
    "imagenetr":     "imagenet_r",
    "imageneta":     "imagenet_a",
    "cub":           "cub_200",
    "omnibenchmark": "omnibenchmark",
    "vtab":          "vtab",
    "objectnet":     "objectnet",
}

import argparse
import copy
import datetime
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms as T
from data_process.data_manger import DataManger
from models.MiN import MinNet
from utils.factory import get_model
from trainer.BaseTrainer import _set_device, _set_random

# MiN test-time transform (matches trainer/data pipeline).
_MIN_TEST_TRSF = T.Compose([
    T.Resize(256, interpolation=3),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_task_order_indices(task_id: int, init_class: int, increment: int) -> List[int]:
    """Return the order-space class indices that belong to task_id."""
    if task_id == 0:
        return list(range(init_class))
    start = init_class + (task_id - 1) * increment
    return list(range(start, start + increment))


def _zero_and_renorm(wn: torch.Tensor, idx: int) -> torch.Tensor:
    """Clone wn, zero entry idx, renormalise remaining entries to sum to 1."""
    wn = wn.detach().clone().float()
    wn[idx] = 0.0
    s = wn.sum()
    return wn / s if s > 1e-8 else wn


# ─────────────────────────────────────────────────────────────────────────────
# WebDataset helpers
# ─────────────────────────────────────────────────────────────────────────────

class _WdsTaskDataset(Dataset):
    """Per-task subset of a WdsDataset returning (pos, image, order_label) 3-tuples.

    Labels are remapped from wds class IDs to MiN order-space indices via
    `wds_to_order`, which accounts for the alphabetical-vs-os.listdir class
    ordering difference between the two pipelines.
    """

    def __init__(self, wds_test, indices: List[int], wds_to_order: dict, transform):
        self.wds_test = wds_test
        self.indices = indices
        self.transform = transform
        self.labels = np.array(
            [wds_to_order[wds_test.targets[i]] for i in indices], dtype=np.int64
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos: int):
        image = self.wds_test.get_pil_image(self.indices[pos])
        return pos, self.transform(image), int(self.labels[pos])


class _WdsTrainTaskDataset(Dataset):
    """Per-task WDS training dataset with category-space labels.

    Labels stay in category-space so MiN's cat2order() can remap them after
    get_task_data() returns, matching DataManger.get_task_data(source='train').
    """

    def __init__(self, wds_train, indices: List[int], labels: np.ndarray, trsf):
        self.wds_train = wds_train
        self.indices = indices
        self.labels = labels  # cat-space; reassigned to order-space by MiN
        self.trsf = trsf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos: int):
        image = self.wds_train.get_pil_image(self.indices[pos])
        return pos, self.trsf(image), int(self.labels[pos])


def _find_wds_dir(args: dict) -> Optional[str]:
    """Return the wds directory for this benchmark if it exists, else None."""
    data_root = args.get("data_root", "")
    subdir = _WDS_SUBDIR.get(args.get("dataset", ""))
    if not subdir:
        return None
    wds_dir = os.path.join(data_root, "wds", subdir)
    return wds_dir if os.path.isfile(os.path.join(wds_dir, "test_metadata.json")) else None


def _build_wds_per_task_test_loaders(
    wds_dir: str,
    datamanger: DataManger,
    args: dict,
    num_tasks: int,
) -> Dict[int, DataLoader]:
    """Build per-task test loaders from WebDataset shards.

    Bridges the class-naming difference: wds uses alphabetically sorted class
    names while MiN's split_img_label uses os.listdir order.  The translation
    goes through the class name string, which is stable across both pipelines.
    """
    if _QZ_SRC not in sys.path:
        sys.path.insert(0, _QZ_SRC)
    from kanerva_sdm.data.wds_loader import WdsDataset  # type: ignore

    logging.info("Using WebDataset test data from %s", wds_dir)
    wds_test = WdsDataset(wds_dir, "test", transform=None)

    # Build per-wds-class sample index.
    by_wds_class: Dict[int, List[int]] = {}
    for i, lbl in enumerate(wds_test.targets):
        by_wds_class.setdefault(lbl, []).append(i)

    # Build wds_class_id → order-space mapping via class name.
    # datamanger.category_index[min_class_id] == class name for all dataset types.
    wds_to_order: Dict[int, int] = {}
    for wds_cid, cls_name in enumerate(wds_test.classes):
        try:
            min_cid = list(datamanger.category_index).index(cls_name)
            wds_to_order[wds_cid] = datamanger.map_cat2order(min_cid)
        except (ValueError, KeyError):
            pass  # class absent from this datamanger's class_order

    loaders: Dict[int, DataLoader] = {}
    for t in range(num_tasks):
        train_list, _, _ = datamanger.get_task_list(t)
        # Convert MiN class IDs to wds sample indices via class name lookup.
        indices: List[int] = []
        for min_cid in train_list:
            cls_name = datamanger.map_cat2cat_name(min_cid)
            wds_cid = wds_test.class_to_idx.get(cls_name)
            if wds_cid is not None:
                indices.extend(by_wds_class.get(wds_cid, []))
        ds = _WdsTaskDataset(wds_test, indices, wds_to_order, _MIN_TEST_TRSF)
        loaders[t] = DataLoader(
            ds,
            batch_size=args.get("init_batch_size", 64),
            shuffle=False,
            num_workers=args.get("num_workers", 4),
        )
    return loaders


class _WdsBackedDataManger:
    """DataManger wrapper that serves train and test data from WDS shards.

    All metadata methods (get_task_list, map_*, task_size, category_index, …)
    delegate to the underlying DataManger.  get_task_data for train /
    train_no_aug / test is intercepted to read from WDS instead of disk files.
    """

    def __init__(self, dm: DataManger, wds_train, wds_test):
        self._dm = dm
        self._wds_train = wds_train
        self._wds_test = wds_test

        cat_names = list(dm.category_index)

        def _build_cat_index(wds_dataset) -> Dict[int, List[int]]:
            wds_cls_to_cat: Dict[int, int] = {}
            for wds_cid, cls_name in enumerate(wds_dataset.classes):
                try:
                    wds_cls_to_cat[wds_cid] = cat_names.index(cls_name)
                except ValueError:
                    pass
            cat_to_indices: Dict[int, List[int]] = {}
            for i, wds_lbl in enumerate(wds_dataset.targets):
                cat_id = wds_cls_to_cat.get(wds_lbl)
                if cat_id is not None:
                    cat_to_indices.setdefault(cat_id, []).append(i)
            return cat_to_indices

        self._cat_to_wds_indices = _build_cat_index(wds_train)
        self._cat_to_wds_test_indices = _build_cat_index(wds_test)

    def __getattr__(self, name):
        return getattr(self._dm, name)

    def get_task_data(self, source: str, class_list: list):
        if source in ("train", "train_no_aug", "test"):
            if source == "train":
                wds_ds = self._wds_train
                index_map = self._cat_to_wds_indices
                trsf = self._dm.train_trsf
            elif source == "train_no_aug":
                wds_ds = self._wds_train
                index_map = self._cat_to_wds_indices
                trsf = self._dm.test_trsf
            else:  # test
                wds_ds = self._wds_test
                index_map = self._cat_to_wds_test_indices
                trsf = self._dm.test_trsf
            indices: List[int] = []
            labels: List[int] = []
            for cat_id in class_list:
                for idx in index_map.get(cat_id, []):
                    indices.append(idx)
                    labels.append(cat_id)
            if not indices:
                raise ValueError(
                    f"WDS {source}: no samples found for class_list={class_list!r}"
                )
            return _WdsTrainTaskDataset(
                wds_ds, indices,
                np.array(labels, dtype=np.int64), trsf,
            )
        return self._dm.get_task_data(source, class_list)


def _wrap_datamanger_with_wds_train(dm: DataManger, wds_dir: str) -> _WdsBackedDataManger:
    """Return a DataManger wrapper that serves train and test data from WDS shards."""
    if _QZ_SRC not in sys.path:
        sys.path.insert(0, _QZ_SRC)
    from kanerva_sdm.data.wds_loader import WdsDataset  # type: ignore

    wds_train = WdsDataset(wds_dir, "train", transform=None)
    wds_test = WdsDataset(wds_dir, "test", transform=None)
    logging.info(
        "WDS train loader: %d samples, %d classes from %s",
        len(wds_train), len(wds_train.classes), wds_dir,
    )
    logging.info(
        "WDS test loader: %d samples, %d classes from %s",
        len(wds_test), len(wds_test.classes), wds_dir,
    )
    return _WdsBackedDataManger(dm, wds_train, wds_test)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation A0 — zero ALL noise generators
# ─────────────────────────────────────────────────────────────────────────────

def ablation_a0_zero_all_noise(model: MinNet) -> MinNet:
    """
    A0: Zero weight_noise for every task across all PiNoise layers.

    The noise sum in each block becomes identically zero, so the forward pass
    reduces to x1 + hyper_features (shared MLP residual + transformer block
    output) for every layer. This is a task-agnostic ablation used to measure
    how much discriminative signal the noise generators contribute overall,
    independent of task-specific routing.

    If accuracy on the forgotten task is unchanged relative to A1/A2, the noise
    generators carry negligible information and the backbone features alone drive
    classification. If accuracy degrades substantially across all tasks, the
    generators are load-bearing.
    """
    m = copy.deepcopy(model)
    net = m._network
    for j in range(net.backbone.layer_num):
        pi = net.backbone.noise_maker[j]
        pi.weight_noise = torch.zeros_like(pi.weight_noise.detach())
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Ablation A1 — zero mixture weight
# ─────────────────────────────────────────────────────────────────────────────

def ablation_a1_zero_weight(model: MinNet, task_to_forget: int) -> MinNet:
    """
    A1: Set ω[task_to_forget] = 0 and renormalise the remaining weights.

    The noise generator P^u still exists but contributes zero to the mixed
    feature signal. This is the weakest intervention: feature-level routing
    is suppressed but all network parameters are intact.
    """
    m = copy.deepcopy(model)
    net = m._network
    for j in range(net.backbone.layer_num):
        pi = net.backbone.noise_maker[j]
        pi.weight_noise = _zero_and_renorm(pi.weight_noise, task_to_forget)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Ablation A2 — remove task generator
# ─────────────────────────────────────────────────────────────────────────────

def ablation_a2_remove_generator(model: MinNet, task_to_forget: int) -> MinNet:
    """
    A2: Zero all parameters of mu[u] and sigmma[u] for the forgotten task.

    Ensures ε_u ≡ 0 regardless of weight_noise — there is no feature-space
    signal from P^u in any forward pass. weight_noise[u] is also zeroed for
    consistency, though it is irrelevant once the generator is dead.

    Design note: we zero the linear weights rather than removing the module
    from the ModuleList so the model structure stays intact for deepcopy and
    state_dict compatibility. Both weight and bias are zeroed so the map is
    identically 0 for any input.
    """
    m = copy.deepcopy(model)
    net = m._network
    for j in range(net.backbone.layer_num):
        pi = net.backbone.noise_maker[j]
        with torch.no_grad():
            for p in pi.mu[task_to_forget].parameters():
                p.zero_()
            for p in pi.sigmma[task_to_forget].parameters():
                p.zero_()
        pi.weight_noise = _zero_and_renorm(pi.weight_noise, task_to_forget)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Ablation A3 — remove generator + analytic classifier columns
# ─────────────────────────────────────────────────────────────────────────────

def ablation_a3_full_removal(
    model: MinNet,
    task_to_forget: int,
    classes_in_task: List[int],
) -> MinNet:
    """
    A3: Remove P^u (via A2) and zero analytic classifier columns for the
    forgotten task's classes.

    `classes_in_task` must be ORDER-space indices (0-indexed positions in
    class_order) matching the column layout of _network.weight
    [buffer_size × total_classes].

    After this operation:
    - Noise generator contributes nothing (same as A2).
    - Logits for all forgotten-task classes are identically 0, so predictions
      can never land on those classes. This is the most complete intervention.
    """
    m = ablation_a2_remove_generator(model, task_to_forget)
    with torch.no_grad():
        m._network.weight[:, classes_in_task] = 0.0
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def save_unlearn_checkpoint(model: MinNet, path: str) -> None:
    """
    Save a MiN model including weight_noise and w_down tensors that are NOT
    captured by state_dict() (they are plain attributes, not parameters/buffers).
    """
    net = model._network
    wn_state: Dict[int, torch.Tensor] = {}
    wd_state: Dict[int, torch.Tensor] = {}
    for j in range(net.backbone.layer_num):
        pi = net.backbone.noise_maker[j]
        wn_state[j] = pi.weight_noise.detach().cpu()
        wd_state[j] = pi.w_down.detach().cpu()
    torch.save(
        {
            "state_dict": net.state_dict(),
            "weight_noise": wn_state,
            "w_down": wd_state,
            "task_prototypes": [p.cpu() for p in net.task_prototypes],
        },
        path,
    )
    logging.info("Saved unlearn checkpoint → %s", path)


def load_unlearn_checkpoint(model: MinNet, path: str) -> MinNet:
    """Restore a MiN model from a checkpoint written by save_unlearn_checkpoint()."""
    ckpt = torch.load(path, map_location="cpu")
    net = model._network
    net.load_state_dict(ckpt["state_dict"])
    dev = net.device
    for j in range(net.backbone.layer_num):
        pi = net.backbone.noise_maker[j]
        pi.weight_noise = ckpt["weight_noise"][j].to(dev)
        pi.w_down = ckpt["w_down"][j].to(dev)
    net.task_prototypes = [p.to(dev) for p in ckpt["task_prototypes"]]
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_per_task_test_loaders(
    datamanger: DataManger,
    args: dict,
    num_tasks: int,
    wds_dir: Optional[str] = None,
) -> Dict[int, DataLoader]:
    """Build one DataLoader per task containing only that task's own classes.

    Labels are remapped to order-space (0 … total_classes-1) to match
    what the model's analytic classifier produces.

    When `wds_dir` is provided (or auto-detected via args["data_root"] +
    the dataset name), the loaders are built from WebDataset shards instead
    of the MiN DataManger, avoiding the slow parallel image-loading path.
    """
    resolved_wds = wds_dir or _find_wds_dir(args)
    if resolved_wds is not None:
        return _build_wds_per_task_test_loaders(resolved_wds, datamanger, args, num_tasks)

    loaders: Dict[int, DataLoader] = {}
    for t in range(num_tasks):
        train_list, _, _ = datamanger.get_task_list(t)
        dataset = datamanger.get_task_data(source="test", class_list=train_list)
        dataset.labels = MinNet.cat2order(dataset.labels, datamanger)
        loaders[t] = DataLoader(
            dataset,
            batch_size=args.get("init_batch_size", 64),
            shuffle=False,
            num_workers=args.get("num_workers", 4),
        )
    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation framework
# ─────────────────────────────────────────────────────────────────────────────

class UnlearningEvaluator:
    """Evaluate unlearning effectiveness across multiple metrics."""

    def __init__(
        self,
        model: MinNet,
        test_loaders: Dict[int, DataLoader],
        task_to_forget: int,
        retained_tasks: List[int],
        device: torch.device,
    ):
        self.model = model
        self.test_loaders = test_loaders
        self.task_to_forget = task_to_forget
        self.retained_tasks = retained_tasks
        self.device = device

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference; return (predictions, true_labels) in order-space."""
        self.model._network.eval()
        preds, labels = [], []
        for _, inputs, targets in loader:
            out = self.model._network(inputs.to(self.device))
            preds.extend(out["logits"].argmax(1).cpu().tolist())
            labels.extend(targets.tolist())
        return np.asarray(preds), np.asarray(labels)

    # ── primary metrics ───────────────────────────────────────────────────────

    def evaluate_forgotten_accuracy(self) -> float:
        """Accuracy on forgotten task. Should approach chance level after strong unlearning."""
        p, l = self._predict(self.test_loaders[self.task_to_forget])
        return float((p == l).mean())

    def evaluate_retained_accuracy(self) -> Dict[int, float]:
        """Per-task accuracy on retained tasks. Should stay high after unlearning."""
        results: Dict[int, float] = {}
        for t in self.retained_tasks:
            p, l = self._predict(self.test_loaders[t])
            results[t] = float((p == l).mean())
        return results

    def compute_backward_transfer(
        self, baseline_accuracies: Dict[int, float]
    ) -> float:
        """
        BWT = mean(acc_after[t] - acc_before[t]) for retained tasks.

        Negative → collateral damage (retained tasks degraded by unlearning).
        Near-zero → surgical unlearning with no side effects.
        """
        retained = self.evaluate_retained_accuracy()
        deltas = [
            retained[t] - baseline_accuracies[t]
            for t in self.retained_tasks
            if t in baseline_accuracies
        ]
        return float(np.mean(deltas)) if deltas else 0.0

    def analyze_confusion_patterns(self) -> Dict:
        """
        For forgotten-task samples, record where predictions land.

        Returns:
            accuracy         : accuracy on the forgotten task after ablation
            pred_distribution: {predicted_class: count} histogram
            predictions      : raw prediction array (list)
            labels           : raw label array (list)

        High accuracy → unlearning failed (model still knows the task).
        Distribution spread across many retained classes → knowledge redistributed.
        Distribution collapsing to a few classes → migration to a surrogate.
        """
        preds, labels = self._predict(self.test_loaders[self.task_to_forget])
        unique, counts = np.unique(preds, return_counts=True)
        return {
            "accuracy": float((preds == labels).mean()),
            "pred_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
            "predictions": preds.tolist(),
            "labels": labels.tolist(),
        }

    def measure_feature_separability(
        self, max_samples_per_task: int = 500
    ) -> Dict:
        """
        Train a logistic probe on raw backbone features (no noise injection)
        to check whether forgotten-class representations remain linearly
        separable from retained classes.

        High probe accuracy: backbone still encodes forgotten classes distinctly —
        the unlearning erased routing/classifier but not the underlying encoding.
        Low probe accuracy: backbone itself has lost distinctive features for the
        forgotten task (rare with frozen backbones).
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import normalize as sk_norm
        except ImportError:
            return {"error": "scikit-learn not installed; skipping probe"}

        self.model._network.eval()

        def _extract(loader: DataLoader, limit: int):
            feats, labs = [], []
            collected = 0
            with torch.no_grad():
                for _, inputs, targets in loader:
                    f = self.model._network.backbone(inputs.to(self.device))
                    feats.append(f.cpu().numpy())
                    labs.append(targets.numpy())
                    collected += len(targets)
                    if collected >= limit:
                        break
            return np.concatenate(feats), np.concatenate(labs)

        Xf, yf = _extract(self.test_loaders[self.task_to_forget], max_samples_per_task)

        retained_sample = self.retained_tasks[:4]  # cap at 4 tasks for speed
        Xr_parts = []
        for t in retained_sample:
            Xr, _ = _extract(self.test_loaders[t], max_samples_per_task)
            Xr_parts.append(Xr)
        if not Xr_parts:
            return {"error": "no retained data available"}

        Xr_all = np.concatenate(Xr_parts)
        X_all = np.concatenate([Xf, Xr_all])
        y_binary = np.concatenate([np.ones(len(yf)), np.zeros(len(Xr_all))])

        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        clf.fit(sk_norm(X_all), y_binary)
        probe_acc = clf.score(sk_norm(X_all), y_binary)

        return {
            "probe_accuracy": float(probe_acc),
            "n_forgotten": int(len(yf)),
            "n_retained": int(len(Xr_all)),
            "features_separable": bool(probe_acc > 0.85),
        }

    # ── bundled evaluation ────────────────────────────────────────────────────

    def full_eval(
        self,
        baseline_accuracies: Optional[Dict[int, float]] = None,
        run_feature_probe: bool = False,
    ) -> Dict:
        forgotten_acc = self.evaluate_forgotten_accuracy()
        retained_accs = self.evaluate_retained_accuracy()
        mean_retained = float(np.mean(list(retained_accs.values()))) if retained_accs else 0.0
        bwt = (
            self.compute_backward_transfer(baseline_accuracies)
            if baseline_accuracies is not None
            else None
        )
        confusion = self.analyze_confusion_patterns()

        result = {
            "forgotten_accuracy": forgotten_acc,
            "retained_accuracy": retained_accs,
            "mean_retained_accuracy": mean_retained,
            "bwt": bwt,
            "confusion": confusion,
        }
        if run_feature_probe:
            result["feature_separability"] = self.measure_feature_separability()
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Dict], task_to_forget: int) -> None:
    """Print a concise comparison table to stdout."""
    header = f"{'Method':<28} {'Forgotten acc':>14} {'Mean retained':>14} {'BWT':>10}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(f"  Task to forget: {task_to_forget}")
    print(sep)
    print(header)
    print(sep)
    for name, r in results.items():
        f_acc = r.get("forgotten_accuracy", float("nan"))
        m_ret = r.get("mean_retained_accuracy", float("nan"))
        bwt = r.get("bwt")
        bwt_str = f"{bwt:+.4f}" if bwt is not None else "    —"
        print(f"  {name:<26} {f_acc:>13.1%} {m_ret:>13.1%} {bwt_str:>10}")
    print(sep)


def save_plots(
    results: Dict[str, Dict],
    task_to_forget: int,
    output_dir: str,
) -> None:
    """Generate and save summary plots to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    method_names = list(results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    # Figure 1: forgotten vs mean-retained accuracy per method.
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(method_names))
    w = 0.35
    f_accs = [results[n]["forgotten_accuracy"] * 100 for n in method_names]
    r_accs = [results[n]["mean_retained_accuracy"] * 100 for n in method_names]
    ax.bar(x - w / 2, f_accs, w, label="Forgotten task acc", color=colors[0])
    ax.bar(x + w / 2, r_accs, w, label="Mean retained acc", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Unlearning accuracy — task {task_to_forget}")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    path1 = os.path.join(output_dir, "accuracy_comparison.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    logging.info("Saved plot → %s", path1)

    # Figure 2: per-task retained accuracy per method.
    retained_tasks_set = sorted(
        {t for n in method_names for t in results[n].get("retained_accuracy", {})}
    )
    if retained_tasks_set:
        fig, ax = plt.subplots(figsize=(max(8, len(retained_tasks_set) * 0.6), 4))
        for i, name in enumerate(method_names):
            ret_d = results[name].get("retained_accuracy", {})
            ys = [ret_d.get(t, float("nan")) * 100 for t in retained_tasks_set]
            ax.plot(
                retained_tasks_set, ys, marker="o",
                label=name, color=colors[i % len(colors)],
            )
        ax.set_xlabel("Retained task index")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Per-task retained accuracy after unlearning")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 105)
        fig.tight_layout()
        path2 = os.path.join(output_dir, "retained_per_task.png")
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        logging.info("Saved plot → %s", path2)

    # Figure 3: predicted-class histogram for forgotten task (strongest ablation).
    for target_method in ["A3 — full removal", "A2 — remove generator", "A0 — zero all noise"]:
        if target_method in results:
            cf = results[target_method].get("confusion", {})
            dist = cf.get("pred_distribution", {})
            if dist:
                classes = sorted(dist.keys())
                counts = [dist[c] for c in classes]
                fig, ax = plt.subplots(figsize=(min(16, max(8, len(classes) * 0.4)), 4))
                ax.bar(classes, counts, color=colors[2])
                ax.set_xlabel("Predicted class (order-space index)")
                ax.set_ylabel("Sample count")
                ax.set_title(
                    f"Where forgotten samples land — {target_method}\n"
                    f"accuracy on forgotten task: {cf['accuracy']:.1%}"
                )
                fig.tight_layout()
                safe_name = target_method.replace(" ", "_").replace("—", "-")
                path3 = os.path.join(output_dir, f"confusion_hist_{safe_name}.png")
                fig.savefig(path3, dpi=150)
                plt.close(fig)
                logging.info("Saved plot → %s", path3)
            break


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_and_return(
    args: dict, wds_dir: Optional[str] = None
) -> Tuple[MinNet, DataManger]:
    """Run the full MiN training pipeline and return the trained model in memory."""
    _set_random(args.get("seed", 1993))
    datamanger = DataManger(args["dataset"], args["device"], args)
    if wds_dir is not None:
        logging.info("Using WDS train data from %s", wds_dir)
        datamanger = _wrap_datamanger_with_wds_train(datamanger, wds_dir)
    model: MinNet = get_model(args, logging.getLogger(__name__))
    model.init_train(data_manger=datamanger)
    for _ in range(datamanger.task_size):
        model.increment_train(data_manger=datamanger)
    return model, datamanger


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_unlearning_experiments(
    args: dict,
    model: MinNet,
    datamanger: DataManger,
    task_to_forget: int,
    output_dir: str,
    run_feature_probe: bool = True,
    wds_dir: Optional[str] = None,
) -> Dict:
    """
    Run baseline + A1/A2/A3 ablations and return all results as a dict.

    Prints a summary table, an interpretation, and saves plots + JSON.
    """
    device = args["device"]
    init_class = args["init_class"]
    increment = args["increment"]
    total_tasks = datamanger.task_size + 1  # task 0 + increment tasks

    if task_to_forget >= total_tasks:
        raise ValueError(
            f"task_to_forget={task_to_forget} but model only trained {total_tasks} tasks"
        )

    retained_tasks = [t for t in range(total_tasks) if t != task_to_forget]
    forgotten_classes = get_task_order_indices(task_to_forget, init_class, increment)

    logging.info(
        "Unlearning experiment: task_to_forget=%d  order_classes=%s  total_tasks=%d",
        task_to_forget, forgotten_classes, total_tasks,
    )

    test_loaders = build_per_task_test_loaders(datamanger, args, total_tasks, wds_dir=wds_dir)

    def _eval(m: MinNet, baseline_ret=None, probe=False) -> Dict:
        ev = UnlearningEvaluator(m, test_loaders, task_to_forget, retained_tasks, device)
        return ev.full_eval(baseline_accuracies=baseline_ret, run_feature_probe=probe)

    # Baseline.
    logging.info("Evaluating baseline …")
    baseline = _eval(model, probe=run_feature_probe)
    baseline_retained = baseline["retained_accuracy"]

    # A0.
    logging.info("A0: zeroing ALL noise generators …")
    eval_a0 = _eval(ablation_a0_zero_all_noise(model), baseline_retained)

    # A1.
    logging.info("A1: zeroing mixture weight …")
    eval_a1 = _eval(ablation_a1_zero_weight(model, task_to_forget), baseline_retained)

    # A2.
    logging.info("A2: removing noise generator …")
    eval_a2 = _eval(ablation_a2_remove_generator(model, task_to_forget), baseline_retained)

    # A3.
    logging.info("A3: removing generator + classifier columns …")
    eval_a3 = _eval(
        ablation_a3_full_removal(model, task_to_forget, forgotten_classes),
        baseline_retained,
        probe=run_feature_probe,
    )

    all_results = {
        "Baseline": baseline,
        "A0 — zero all noise": eval_a0,
        "A1 — zero weight": eval_a1,
        "A2 — remove generator": eval_a2,
        "A3 — full removal": eval_a3,
    }

    print_results_table(all_results, task_to_forget)
    _print_interpretation(all_results, task_to_forget)

    # Save JSON.
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"unlearn_task{task_to_forget}_{ts}.json")
    payload = _make_serialisable(all_results)
    payload["meta"] = {
        "task_to_forget": task_to_forget,
        "forgotten_classes": forgotten_classes,
        "total_tasks": total_tasks,
        "dataset": args.get("dataset"),
        "backbone": args.get("backbone_type"),
        "timestamp": ts,
    }
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logging.info("Results JSON → %s", json_path)

    save_plots(all_results, task_to_forget, output_dir)

    return all_results


def _print_interpretation(results: Dict[str, Dict], task_to_forget: int) -> None:
    """Print a human-readable interpretation of the unlearning results."""
    b_f  = results["Baseline"]["forgotten_accuracy"]
    a0_f = results["A0 — zero all noise"]["forgotten_accuracy"]
    a1_f = results["A1 — zero weight"]["forgotten_accuracy"]
    a2_f = results["A2 — remove generator"]["forgotten_accuracy"]
    a3_f = results["A3 — full removal"]["forgotten_accuracy"]

    a0_bwt = results["A0 — zero all noise"].get("bwt") or 0.0
    a1_bwt = results["A1 — zero weight"].get("bwt") or 0.0
    a2_bwt = results["A2 — remove generator"].get("bwt") or 0.0
    a3_bwt = results["A3 — full removal"].get("bwt") or 0.0

    n_drop  = b_f  - a0_f   # effect of zeroing all noise (global noise contribution)
    w_drop  = b_f  - a1_f   # effect of zeroing task mixture weight alone
    g_drop  = a1_f - a2_f   # additional effect of zeroing generator params
    c_drop  = a2_f - a3_f   # additional effect of zeroing classifier columns

    print("\n── Interpretation ───────────────────────────────────────────────────────")
    print(f"  Forgotten task accuracy:  baseline={b_f:.1%}  A0={a0_f:.1%}  A1={a1_f:.1%}  A2={a2_f:.1%}  A3={a3_f:.1%}")
    print()
    print(f"  Drop A0 vs baseline : {n_drop:+.1%}  (all noise generators — global signal)")
    print(f"  Drop A1 vs baseline : {w_drop:+.1%}  (mixture-weight routing)")
    print(f"  Drop A2 vs A1       : {g_drop:+.1%}  (noise-generator parameters)")
    print(f"  Drop A3 vs A2       : {c_drop:+.1%}  (analytic classifier columns)")

    a0_mean_ret = results["A0 — zero all noise"].get("mean_retained_accuracy", float("nan"))
    b_mean_ret  = results["Baseline"].get("mean_retained_accuracy", float("nan"))
    print()
    if abs(n_drop) < 0.02 and abs(a0_mean_ret - b_mean_ret) < 0.02:
        print("  ► Noise generators carry negligible discriminative signal; backbone")
        print("    features (x1 + hyper_features) alone drive classification.")
    elif n_drop > 0.05:
        print(f"  ► Noise generators contribute meaningfully to the forgotten task")
        print(f"    ({n_drop:+.1%} drop), but task-specific routing (A1/A2) is still insufficient.")
    if abs(a0_mean_ret - b_mean_ret) > 0.05:
        print(f"  ► Zeroing all noise degrades retained tasks by {a0_mean_ret - b_mean_ret:+.1%} mean acc —")
        print(f"    noise is load-bearing for general classification, not just the forgotten task.")

    if w_drop > 0.10:
        print("  ► Mixture weights carry significant routing — A1 partially effective.")
    else:
        print("  ► Mixture weights alone are insufficient; task knowledge is in generator params.")

    if g_drop > 0.10:
        print("  ► Noise generators P^u store task-specific features — zeroing them is necessary.")
    else:
        print("  ► P^u parameters store little additional info beyond routing.")

    if c_drop > 0.05:
        print("  ► Classifier columns add residual signal — A3 needed for full suppression.")
    else:
        print("  ► Once the generator is removed, the classifier adds minimal task signal.")

    print(f"\n  Collateral damage (BWT on retained tasks):")
    print(f"    A0: {a0_bwt:+.4f}  |  A1: {a1_bwt:+.4f}  |  A2: {a2_bwt:+.4f}  |  A3: {a3_bwt:+.4f}")
    worst = min(a0_bwt, a1_bwt, a2_bwt, a3_bwt)
    if worst < -0.05:
        print("  ► Significant collateral damage — retained tasks degraded by unlearning.")
    elif worst < -0.01:
        print("  ► Minor collateral damage; unlearning is mostly surgical.")
    else:
        print("  ► Negligible collateral damage — retained tasks unaffected.")

    # Feature probe (A3 if available, else A2).
    for key in ["A3 — full removal", "A2 — remove generator"]:
        probe = results.get(key, {}).get("feature_separability")
        if probe and "error" not in probe:
            acc = probe["probe_accuracy"]
            sep = probe["features_separable"]
            print(f"\n  Feature separability probe ({key}): probe_acc={acc:.3f}")
            if sep:
                print("  ► Backbone features still separable — representation intact after ablation.")
                print("    Unlearning erased routing/classification but not the underlying encoding.")
            else:
                print("  ► Backbone features not linearly separable — representation significantly altered.")
            break

    print("─" * 72)


def _make_serialisable(obj):
    """Recursively convert numpy/tensor types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def print_all_tasks_summary(per_task_results: Dict[int, Dict]) -> None:
    """Print a cross-task aggregate table after running --all_tasks."""
    methods = ["Baseline", "A0 — zero all noise", "A1 — zero weight", "A2 — remove generator", "A3 — full removal"]
    col = 14
    header = f"{'Task':>5}  " + "  ".join(f"{m[:col]:>{col}}" for m in methods)
    sep = "─" * len(header)
    print(f"\n{'═' * len(header)}")
    print("  All-tasks unlearning summary — forgotten task accuracy")
    print(sep)
    print(header)
    print(sep)
    for t, res in sorted(per_task_results.items()):
        row = f"  {t:>3}  "
        for m in methods:
            acc = res.get(m, {}).get("forgotten_accuracy", float("nan"))
            row += f"  {acc:>{col}.1%}"
        print(row)
    print(sep)

    print(f"\n{'─' * len(header)}")
    print("  Mean retained accuracy (should stay high)")
    print(sep)
    print(header)
    print(sep)
    for t, res in sorted(per_task_results.items()):
        row = f"  {t:>3}  "
        for m in methods:
            acc = res.get(m, {}).get("mean_retained_accuracy", float("nan"))
            row += f"  {acc:>{col}.1%}"
        print(row)
    print(sep)

    print(f"\n{'─' * len(header)}")
    print("  BWT on retained tasks (negative = collateral damage)")
    print(sep)
    print(header)
    print(sep)
    for t, res in sorted(per_task_results.items()):
        row = f"  {t:>3}  "
        for m in methods:
            bwt = res.get(m, {}).get("bwt")
            row += f"  {(bwt or 0.0):+{col}.4f}"
        print(row)
    print("═" * len(header))


def save_all_tasks_plot(per_task_results: Dict[int, Dict], output_dir: str) -> None:
    """Heat-map of forgotten-task accuracy across all (task, method) combinations."""
    methods = ["Baseline", "A0 — zero all noise", "A1 — zero weight", "A2 — remove generator", "A3 — full removal"]
    tasks = sorted(per_task_results.keys())
    data = np.array([
        [per_task_results[t].get(m, {}).get("forgotten_accuracy", float("nan")) * 100
         for m in methods]
        for t in tasks
    ])
    fig, ax = plt.subplots(figsize=(len(methods) * 2.2, max(4, len(tasks) * 0.5)))
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=100, cmap="RdYlGn_r")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.split("—")[-1].strip() for m in methods], rotation=20, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"Task {t}" for t in tasks])
    for i, t in enumerate(tasks):
        for j in range(len(methods)):
            ax.text(j, i, f"{data[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="Forgotten acc (%)")
    ax.set_title("Forgotten-task accuracy after each unlearning intervention")
    fig.tight_layout()
    path = os.path.join(output_dir, "all_tasks_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("Saved plot → %s", path)


def run_a0_experiment(
    args: dict,
    model: MinNet,
    datamanger: DataManger,
    output_dir: str,
    wds_dir: Optional[str] = None,
) -> Dict:
    """Evaluate baseline vs A0 (all noise zeroed) across every task.

    Since A0 is task-agnostic the ablated model is built once, then per-task
    accuracy is reported for both baseline and A0 side-by-side.  This is the
    right framing for the question "how much do noise generators contribute
    globally?" — no forgotten/retained split is needed.
    """
    device = args["device"]
    total_tasks = datamanger.task_size + 1
    test_loaders = build_per_task_test_loaders(datamanger, args, total_tasks, wds_dir=wds_dir)

    def _per_task_acc(m: MinNet) -> Dict[int, float]:
        m._network.eval()
        accs: Dict[int, float] = {}
        for t, loader in test_loaders.items():
            ev = UnlearningEvaluator(m, test_loaders, t, [], device)
            p, l = ev._predict(loader)
            accs[t] = float((p == l).mean())
        return accs

    logging.info("A0 experiment: evaluating baseline …")
    baseline_accs = _per_task_acc(model)

    logging.info("A0 experiment: zeroing all noise generators …")
    model_a0 = ablation_a0_zero_all_noise(model)
    a0_accs = _per_task_acc(model_a0)

    # Print table.
    col = 12
    header = f"  {'Task':>5}  {'Baseline':>{col}}  {'A0 — no noise':>{col}}  {'Delta':>{col}}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("  A0: per-task accuracy — baseline vs all noise zeroed")
    print(sep)
    print(header)
    print(sep)
    for t in sorted(baseline_accs):
        b = baseline_accs[t]
        a = a0_accs[t]
        print(f"  {t:>5}  {b:>{col}.1%}  {a:>{col}.1%}  {a - b:>+{col}.1%}")
    mean_b = float(np.mean(list(baseline_accs.values())))
    mean_a = float(np.mean(list(a0_accs.values())))
    print(sep)
    print(f"  {'mean':>5}  {mean_b:>{col}.1%}  {mean_a:>{col}.1%}  {mean_a - mean_b:>+{col}.1%}")
    print(sep)

    if abs(mean_a - mean_b) < 0.02:
        print("\n  ► Noise generators carry negligible signal — backbone features dominate.")
    elif mean_a < mean_b - 0.05:
        print(f"\n  ► Noise contributes meaningfully: mean accuracy drops {mean_b - mean_a:.1%} without it.")
    print()

    result = {"baseline": baseline_accs, "a0": a0_accs}

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"a0_experiment_{ts}.json")
    with open(json_path, "w") as fh:
        json.dump(_make_serialisable(result), fh, indent=2)
    logging.info("A0 results JSON → %s", json_path)

    # Plot.
    tasks = sorted(baseline_accs)
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 0.5), 4))
    ax.plot(tasks, [baseline_accs[t] * 100 for t in tasks], marker="o", label="Baseline")
    ax.plot(tasks, [a0_accs[t] * 100 for t in tasks], marker="s", label="A0 — no noise")
    ax.set_xlabel("Task index")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-task accuracy: baseline vs all noise zeroed")
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "a0_per_task.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info("A0 plot → %s", plot_path)

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiN unlearning experiments")
    p.add_argument("--base_configs",    required=True, help="Base config JSON path")
    p.add_argument("--model_configs",   required=True, help="Model config JSON path")
    p.add_argument("--task_to_forget",  type=int, default=None,
                   help="Task index to unlearn (0-indexed). Omit with --all_tasks.")
    p.add_argument("--all_tasks",       action="store_true",
                   help="Run ablations for every task (train once, ablate all)")
    p.add_argument("--a0_only",         action="store_true",
                   help="Run only the A0 (zero all noise) experiment across all tasks")
    p.add_argument("--checkpoint",      default=None,
                   help="Path to unlearn checkpoint (skips training)")
    p.add_argument("--save_checkpoint", action="store_true",
                   help="Save trained model as unlearn checkpoint after training")
    p.add_argument("--output_dir",      default=None,
                   help="Output directory for plots and JSON (default: MiN/logs/unlearn/)")
    p.add_argument("--device",          default=None,
                   help="Override GPU device index, e.g. 0 or 1")
    p.add_argument("--seed",            type=int, default=None,
                   help="Random seed override")
    p.add_argument("--no_feature_probe", action="store_true",
                   help="Skip the linear separability probe (saves time)")
    p.add_argument("--data_root",       default=None,
                   help="Override data_root from base config")
    p.add_argument("--wds_dir",         default=None,
                   help="Explicit path to WebDataset shards for test loaders. "
                        "If omitted, auto-detected from data_root/wds/<dataset>/.")
    return p.parse_args()


def main() -> None:
    cli = parse_args()

    if not cli.all_tasks and not cli.a0_only and cli.task_to_forget is None:
        raise SystemExit("error: provide --task_to_forget N, --all_tasks, or --a0_only")

    with open(cli.base_configs) as fh:
        args = json.load(fh)
    with open(cli.model_configs) as fh:
        args.update(json.load(fh))

    if cli.data_root:
        args["data_root"] = cli.data_root
    if cli.device is not None:
        args["device"] = [cli.device]
    if cli.seed is not None:
        args["seed"] = cli.seed

    _set_device(args)  # converts args["device"] list → torch.device

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [unlearn] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    out_dir = cli.output_dir or os.path.join(_HERE, "logs", "unlearn")

    # ── Resolve wds directory (used for both train and test loaders) ──────────
    wds_dir = cli.wds_dir or _find_wds_dir(args)
    if wds_dir:
        logging.info("WebDataset shards → %s", wds_dir)
    else:
        logging.info("WebDataset not found; using DataManger for data loading")

    # ── Train or load ──────────────────────────────────────────────────────────
    if cli.checkpoint:
        logging.info("Loading model from checkpoint: %s", cli.checkpoint)
        datamanger = DataManger(args["dataset"], args["device"], args)
        model: MinNet = get_model(args, logging.getLogger(__name__))
        total_tasks = datamanger.task_size + 1
        for t in range(total_tasks):
            nb = args["init_class"] if t == 0 else args["increment"]
            model._network.update_fc(nb)
            model._network.update_noise()
            proto = torch.zeros(model._network.feature_dim, device=args["device"])
            model._network.extend_task_prototype(proto)
            model.cur_task = t
        load_unlearn_checkpoint(model, cli.checkpoint)
    else:
        logging.info("Training model from scratch …")
        model, datamanger = train_and_return(args, wds_dir=wds_dir)

    if cli.save_checkpoint:
        os.makedirs(out_dir, exist_ok=True)
        save_unlearn_checkpoint(model, os.path.join(out_dir, "trained_model.pt"))

    # ── Run experiments ────────────────────────────────────────────────────────
    if cli.a0_only:
        run_a0_experiment(
            args=args,
            model=model,
            datamanger=datamanger,
            output_dir=out_dir,
            wds_dir=wds_dir,
        )
        return

    if cli.all_tasks:
        total_tasks = datamanger.task_size + 1
        logging.info("Running ablations for all %d tasks …", total_tasks)
        per_task_results: Dict[int, Dict] = {}
        for t in range(total_tasks):
            logging.info("─── task_to_forget = %d ───", t)
            task_dir = os.path.join(out_dir, f"task_{t}")
            per_task_results[t] = run_unlearning_experiments(
                args=args,
                model=model,
                datamanger=datamanger,
                task_to_forget=t,
                output_dir=task_dir,
                run_feature_probe=not cli.no_feature_probe,
                wds_dir=wds_dir,
            )
        print_all_tasks_summary(per_task_results)
        save_all_tasks_plot(per_task_results, out_dir)
        # Save aggregated JSON.
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        agg_path = os.path.join(out_dir, f"all_tasks_summary_{ts}.json")
        with open(agg_path, "w") as fh:
            json.dump(_make_serialisable({str(t): r for t, r in per_task_results.items()}), fh, indent=2)
        logging.info("Aggregated results → %s", agg_path)
    else:
        run_unlearning_experiments(
            args=args,
            model=model,
            datamanger=datamanger,
            task_to_forget=cli.task_to_forget,
            output_dir=out_dir,
            run_feature_probe=not cli.no_feature_probe,
            wds_dir=wds_dir,
        )


if __name__ == "__main__":
    main()
