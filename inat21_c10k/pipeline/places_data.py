"""Places365-Standard for MiN — the domain-shift benchmark (NeurIPS 29520 rebuttal).

Built from the official filelists rather than by walking the directory tree, for
two reasons:

  * ``data_256/`` is not a flat ImageFolder — categories are nested two or three
    levels deep (``a/airfield``, but also ``a/apartment_building/outdoor``), so
    ``split_img_label``'s single ``os.listdir`` would mis-index them.
  * ``places365_train_standard.txt`` carries the canonical class index, which is
    the same ordering as ``categories_places365.txt`` and therefore the same
    ordering the qz-side Arrow shards use (their ClassLabel names come from that
    file). Matching it is what makes the head-to-head comparison valid.

Train images live under ``data_256/`` at the paths given in the filelist; the
test split is the official validation set (36,500 images, 100/class) which is
FLAT in ``val_256/`` with labels supplied by ``places365_val.txt``.

``samples_per_class`` subsamples the training set with the run seed, matching the
qz harness (its ``build_task_splits`` subsamples training indices only, under the
run seed; test indices are always taken whole).
"""

import os

import numpy as np
from torchvision import transforms

from data_process.data import iData


class iPlaces365(iData):
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.category_index = None

        self.train_trsf = [
            transforms.RandomResizedCrop(256, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        self.test_trsf = [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
        self.class_order = np.arange(365).tolist()

    def data_initialization(self):
        root = self.args.get('places365_root',
                             '/gpfs/data/oermannlab/public_data/places365')
        spc = self.args.get('samples_per_class', None)
        seed = self.args.get('seed', 0)

        categories = []
        with open(os.path.join(root, 'categories_places365.txt')) as f:
            for line in f:
                line = line.strip()
                if line:
                    # "/a/airfield 0" -> "a/airfield"
                    categories.append(line.split(' ')[0].lstrip('/'))
        assert len(categories) == 365, f'expected 365 categories, got {len(categories)}'
        self.category_index = categories

        # ---- train: filelist -> per-class lists of (abs_path, class_idx) ----
        train_by_class = [[] for _ in range(365)]
        img_root = os.path.join(root, 'data_256')
        with open(os.path.join(root, 'places365_train_standard.txt')) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel, idx = line.rsplit(' ', 1)
                train_by_class[int(idx)].append(
                    (os.path.join(img_root, rel.lstrip('/')), int(idx)))

        if spc:
            rng = np.random.RandomState(seed)
            for c in range(365):
                pool = train_by_class[c]
                if len(pool) > spc:
                    keep = rng.choice(len(pool), size=spc, replace=False)
                    train_by_class[c] = [pool[i] for i in sorted(keep)]
        self.train_data = train_by_class

        # ---- test: official validation split, flat dir + label file ----
        test_by_class = [[] for _ in range(365)]
        val_root = os.path.join(root, 'val_256')
        with open(os.path.join(root, 'places365_val.txt')) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname, idx = line.rsplit(' ', 1)
                test_by_class[int(idx)].append(
                    (os.path.join(val_root, fname), int(idx)))
        self.test_data = test_by_class

        n_train = sum(len(c) for c in self.train_data)
        n_test = sum(len(c) for c in self.test_data)
        print(f'[places365] classes=365 train={n_train} test={n_test} '
              f'samples_per_class={spc} subsample_seed={seed}', flush=True)
