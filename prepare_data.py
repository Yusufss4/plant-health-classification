"""
Data preparation script for Plant Health Classification.

This script downloads and prepares the PlantVillage dataset from TensorFlow Datasets,
splitting it into train/val/test sets with a simple healthy vs diseased classification.

Usage:
    python prepare_data.py [--seed SEED]

For 3-class training (healthy / diseased / background), also run:
    python prepare_background_data.py
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

# Leaf classes created by this script; background/ is added by prepare_background_data.py.
LEAF_CATEGORIES = ('healthy', 'diseased')
OPTIONAL_CATEGORIES = ('background',)
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png')


def _count_images(category_path: Path) -> int:
    n = 0
    for pattern in IMAGE_EXTENSIONS:
        n += len(list(category_path.glob(pattern)))
    return n


def download_and_prepare_data(output_dir='data', seed=None):
    """
    Download PlantVillage dataset and prepare it for binary classification.

    Args:
        output_dir (str): Directory to save prepared data
        seed (int, optional): RNG seed for reproducible train/val/test splits

    The PlantVillage dataset contains 38 classes of plant diseases across 14 crop species.
    Uses all plants with binary classification:
    - healthy: All healthy plant classes
    - diseased: All diseased plant classes
    """
    if seed is not None:
        np.random.seed(seed)

    print("=" * 80)
    print("Downloading PlantVillage Dataset from TensorFlow Datasets")
    print("=" * 80)
    if seed is not None:
        print(f"Using random seed {seed} for train/val/test split")

    # Download dataset
    print("\nDownloading dataset...")
    dataset, info = tfds.load(
        'plant_village',
        split='train',
        with_info=True,
        as_supervised=True
    )

    print(f"\nDataset info:")
    print(f"  Total samples: {info.splits['train'].num_examples}")
    print(f"  Number of classes: {info.features['label'].num_classes}")
    print(f"  Class names: {info.features['label'].names[:5]}... (showing first 5)")

    # Create output directories
    output_path = Path(output_dir)
    splits = ['train', 'val', 'test']

    for split in splits:
        for category in LEAF_CATEGORIES:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)

    print(f"\nPreparing binary classification (healthy vs diseased)...")

    # Get class names
    class_names = info.features['label'].names

    # Use all plants - any class with 'healthy' is healthy, rest are diseased
    healthy_classes = [i for i, name in enumerate(class_names) if 'healthy' in name.lower()]
    diseased_classes = [i for i in range(len(class_names)) if i not in healthy_classes]
    print(f"\nUsing all plant types:")
    print(f"  Healthy classes: {len(healthy_classes)}")
    print(f"  Diseased classes: {len(diseased_classes)}")

    # Process and save images
    counters = {split: {c: 0 for c in LEAF_CATEGORIES} for split in splits}

    print("\nProcessing images...")
    for idx, (image, label) in enumerate(tfds.as_numpy(dataset)):
        if idx % 1000 == 0:
            print(f"  Processed {idx} images...")

        label_idx = int(label)

        # Determine category (healthy or diseased)
        if label_idx in healthy_classes:
            category = 'healthy'
        elif label_idx in diseased_classes:
            category = 'diseased'
        else:
            continue  # Skip if not in our filtered classes

        # Determine split (70% train, 15% val, 15% test)
        rand = np.random.random()
        if rand < 0.70:
            split = 'train'
        elif rand < 0.85:
            split = 'val'
        else:
            split = 'test'

        # Save image
        img = Image.fromarray(image)
        img_filename = f"{class_names[label_idx]}_{counters[split][category]:05d}.jpg"
        img_path = output_path / split / category / img_filename
        img.save(img_path)

        counters[split][category] += 1

    # Print statistics
    print("\n" + "=" * 80)
    print("Dataset preparation complete!")
    print("=" * 80)

    total_samples = 0
    for split in splits:
        split_total = sum(counters[split].values())
        total_samples += split_total
        print(f"\n{split.capitalize()} set:")
        print(f"  Healthy: {counters[split]['healthy']:,} images")
        print(f"  Diseased: {counters[split]['diseased']:,} images")
        print(f"  Total: {split_total:,} images")

    print(f"\nTotal samples: {total_samples:,}")
    print(f"\nData saved to: {output_path.absolute()}/")

    return counters


def verify_data_structure(data_dir='data', require_background=False):
    """
    Verify that the data directory structure is correct.

    Args:
        data_dir: Root data directory (default ``data``).
        require_background: If True, each split must have a non-empty
            ``background/`` folder (needed for 3-class training).

    Returns:
        True if all required checks pass.
    """
    print("\n" + "=" * 80)
    print("Verifying data structure...")
    print("=" * 80)

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        return False

    required_structure = {
        'train': list(LEAF_CATEGORIES),
        'val': list(LEAF_CATEGORIES),
        'test': list(LEAF_CATEGORIES),
    }

    all_good = True
    for split, categories in required_structure.items():
        split_path = data_path / split
        if not split_path.exists():
            print(f"❌ Missing {split} directory")
            all_good = False
            continue

        for category in categories:
            category_path = split_path / category
            if not category_path.exists():
                print(f"❌ Missing {split}/{category} directory")
                all_good = False
            else:
                num_images = _count_images(category_path)
                print(f"✓ {split}/{category}: {num_images:,} images")
                if num_images == 0:
                    print(f"  ⚠️  {split}/{category} is empty")
                    all_good = False

    # Optional background class (3-class model)
    background_ok = True
    for split in ('train', 'val', 'test'):
        bg_path = data_path / split / 'background'
        if not bg_path.is_dir():
            print(f"⚠️  Missing {split}/background/ (run prepare_background_data.py)")
            background_ok = False
        else:
            n = _count_images(bg_path)
            print(f"{'✓' if n > 0 else '⚠️ '} {split}/background: {n:,} images")
            if n == 0:
                background_ok = False

    if require_background and not background_ok:
        print("\n❌ Background class required but missing or empty in one or more splits")
        all_good = False
    elif not background_ok:
        print(
            "\n⚠️  Background class not ready — 3-class training needs "
            "prepare_background_data.py"
        )

    if all_good:
        if background_ok:
            print("\n✅ Data structure is valid (leaf + background classes).")
        else:
            print("\n✅ Leaf data structure is valid (background pending).")
    else:
        print("\n❌ Data structure is incomplete")

    return all_good


def main():
    """Main function to prepare data."""
    parser = argparse.ArgumentParser(
        description='Download PlantVillage and prepare healthy/diseased splits'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible 70/15/15 train/val/test split',
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing data/ layout and exit',
    )
    parser.add_argument(
        '--require-background',
        action='store_true',
        help='Fail verification unless background/ exists in all splits',
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Plant Health Classification Data Setup                     ║
║                                                                              ║
║  This script downloads the PlantVillage dataset from TensorFlow Datasets    ║
║  and prepares it for healthy vs diseased leaf images.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    data_dir = 'data'

    if args.verify_only:
        verify_data_structure(data_dir, require_background=args.require_background)
        return

    # Check if data already exists
    if os.path.exists(data_dir) and any(Path(data_dir).iterdir()):
        print(f"\n⚠️  Data directory '{data_dir}' already exists.")
        response = input("Do you want to delete and re-download? (y/N): ")
        if response.lower() == 'y':
            print("Removing existing data...")
            shutil.rmtree(data_dir)
        else:
            print("Keeping existing data. Verifying structure...")
            verify_data_structure(data_dir, require_background=args.require_background)
            return

    try:
        download_and_prepare_data(output_dir=data_dir, seed=args.seed)
        verify_data_structure(data_dir, require_background=args.require_background)

        print("\n" + "=" * 80)
        print("Next steps:")
        print("=" * 80)
        print("\n1. Add background class (required for 3-class training):")
        print("   python prepare_background_data.py")
        print("\n2. Train model (3 classes):")
        print("   python train.py")
        print("\n3. Export ONNX for edge deployment:")
        print("   python export_mobilenet_onnx.py")
        print("\n4. Evaluate:")
        print("   python evaluate.py")

    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        print("\nPlease ensure you have tensorflow and tensorflow_datasets installed:")
        print("  pip install tensorflow tensorflow-datasets")
        raise


if __name__ == '__main__':
    main()
