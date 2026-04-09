"""Create train/test symlink splits for flat ImageNet-R and ImageNet-A.

MiN's data loader expects separate train/ and test/ directories, but the
qz-compcont-learning data directory stores ImageNet-R and ImageNet-A as flat
folders (200 class dirs, all images mixed). This script creates a symlink-based
train/test split (80/20 per class, deterministic by seed) under:

  <data_root>/imagenet-r/train/<class>/
  <data_root>/imagenet-r/test/<class>/
  <data_root>/imagenet-a/train/<class>/
  <data_root>/imagenet-a/test/<class>/

Run once before run_benchmarks.sh.
"""

import argparse
import os
import random


def make_split(flat_dir, split_dir, train_ratio=0.8, seed=1993):
    """Symlink images from a flat class-dir dataset into train/ and test/ trees.

    Args:
        flat_dir: Source directory with layout flat_dir/<class>/<image>.
        split_dir: Destination directory; train/ and test/ are created here.
        train_ratio: Fraction of images per class assigned to train.
        seed: Random seed for reproducibility.
    """
    if not os.path.isdir(flat_dir):
        print(f"  SKIP: {flat_dir} not found")
        return

    train_dir = os.path.join(split_dir, 'train')
    test_dir = os.path.join(split_dir, 'test')

    if os.path.exists(train_dir) and os.path.exists(test_dir):
        n_train_classes = len(os.listdir(train_dir))
        n_source_classes = sum(
            1 for d in os.listdir(flat_dir)
            if os.path.isdir(os.path.join(flat_dir, d))
        )
        if n_train_classes >= n_source_classes:
            print(f"  Already exists: {split_dir} ({n_train_classes} train classes)")
            return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    rng = random.Random(seed)
    class_dirs = sorted(
        d for d in os.listdir(flat_dir)
        if os.path.isdir(os.path.join(flat_dir, d))
        and d not in ('train', 'test')
    )

    total_train, total_test = 0, 0
    for cls in class_dirs:
        cls_src = os.path.join(flat_dir, cls)
        images = sorted(
            f for f in os.listdir(cls_src)
            if not f.startswith('.')
        )
        rng.shuffle(images)
        n_train = max(1, round(len(images) * train_ratio))
        train_imgs, test_imgs = images[:n_train], images[n_train:]

        for subset, imgs in [('train', train_imgs), ('test', test_imgs)]:
            dst_cls = os.path.join(split_dir, subset, cls)
            os.makedirs(dst_cls, exist_ok=True)
            for img in imgs:
                src = os.path.abspath(os.path.join(cls_src, img))
                dst = os.path.join(dst_cls, img)
                if not os.path.exists(dst):
                    os.symlink(src, dst)

        total_train += len(train_imgs)
        total_test += len(test_imgs)

    print(
        f"  Created {split_dir}: "
        f"{len(class_dirs)} classes, "
        f"{total_train} train / {total_test} test images"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        default=os.path.expanduser('~/qz-compcont-learning/data'),
        help='Root of the qz-compcont-learning data directory (default: ~/qz-compcont-learning/data)',
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.8,
        help='Fraction of images per class used for training (default: 0.8)',
    )
    parser.add_argument(
        '--seed', type=int, default=1993,
        help='Random seed for the split (default: 1993, matches MiN configs)',
    )
    args = parser.parse_args()

    datasets = ['imagenet-r', 'imagenet-a']

    for name in datasets:
        flat_dir = os.path.join(args.data_root, name)
        print(f"Processing {name}...")
        make_split(flat_dir, flat_dir, train_ratio=args.train_ratio, seed=args.seed)

    print("Done. Data ready for run_benchmarks.sh.")


if __name__ == '__main__':
    main()
