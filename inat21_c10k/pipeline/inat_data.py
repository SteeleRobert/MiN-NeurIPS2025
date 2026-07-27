"""iNat21-mini for MiN — the C=10,000 rung of the scale characterization.

Variant: iNat21-**mini**, which is what is on GPFS as raw images
(``train_mini/`` = 10,000 classes x 50 images = 500,000; ``val/`` = 100,000 at
10/class) and what the ``feature/inat21-10k-characterization`` Arrow cache
declares (train 500,000 / validation 100,000, 10,000 classes). Matching it is
what lets these numbers join that table.

Class indices come from sorting the directory names, whose numeric prefix
(``00000_Animalia_...``) makes the sort order identical to the Arrow cache's
ClassLabel order — verified: index 0 is Lumbricus terrestris on both sides.
``split_img_label`` in data.py uses an unsorted ``os.listdir``, which would give
a machine-dependent class order, so this class does its own sorted walk.

NOTE: MiN adapts the backbone through PiNoise at every task boundary, so the
frozen-feature cache used by the statistical heads is deliberately NOT wired in
here — reusing it would silently change the method.
"""

import os

import numpy as np
from torchvision import transforms

from data_process.data import iData


class iNat21(iData):
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
        self.class_order = np.arange(10000).tolist()

    def _index_split(self, root, classes):
        by_class = []
        for idx, cat in enumerate(classes):
            d = os.path.join(root, cat)
            files = sorted(os.listdir(d))
            by_class.append([(os.path.join(d, f), idx) for f in files])
        return by_class

    def data_initialization(self):
        root = self.args.get('inat21_root',
                             '/gpfs/data/oermannlab/public_data/inat21')
        train_root = os.path.join(root, 'train_mini')
        val_root = os.path.join(root, 'val')

        classes = sorted(os.listdir(train_root))
        assert len(classes) == 10000, f'expected 10000 classes, got {len(classes)}'
        self.category_index = classes

        self.train_data = self._index_split(train_root, classes)
        self.test_data = self._index_split(val_root, classes)

        spc = self.args.get('samples_per_class', None)
        if spc:
            rng = np.random.RandomState(self.args.get('seed', 0))
            for c in range(len(self.train_data)):
                pool = self.train_data[c]
                if len(pool) > spc:
                    keep = rng.choice(len(pool), size=spc, replace=False)
                    self.train_data[c] = [pool[i] for i in sorted(keep)]

        n_train = sum(len(c) for c in self.train_data)
        n_test = sum(len(c) for c in self.test_data)
        print(f'[inat21-mini] classes={len(classes)} train={n_train} test={n_test} '
              f'samples_per_class={spc}', flush=True)
