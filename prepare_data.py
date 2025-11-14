"""
Data preparation script for Plant Health Classification.

This script downloads and prepares the PlantVillage dataset from TensorFlow Datasets,
splitting it into train/val/test sets with a simple healthy vs diseased classification.

Usage:
    python prepare_data.py
"""

import os
import shutil
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np


def download_and_prepare_data(output_dir='data', healthy_only=True):
    """
    Download PlantVillage dataset and prepare it for binary classification.
    
    Args:
        output_dir (str): Directory to save prepared data
        healthy_only (bool): If True, only use healthy vs diseased tomato leaves
    
    The PlantVillage dataset contains 38 classes of plant diseases across 14 crop species.
    For simplicity, we'll focus on tomato leaves with binary classification:
    - healthy: Tomato_healthy
    - diseased: All other tomato disease classes
    """
    
    print("="*80)
    print("Downloading PlantVillage Dataset from TensorFlow Datasets")
    print("="*80)
    
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
    categories = ['healthy', 'diseased']
    
    for split in splits:
        for category in categories:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)
    
    print(f"\nPreparing binary classification (healthy vs diseased)...")
    
    # Get class names
    class_names = info.features['label'].names
    
    # Filter for tomato classes or all classes
    if healthy_only:
        # Focus on tomato leaves for clearer healthy/diseased distinction
        healthy_classes = [i for i, name in enumerate(class_names) if 'Tomato___healthy' in name]
        diseased_classes = [i for i, name in enumerate(class_names) if 'Tomato___' in name and 'healthy' not in name]
        print(f"\nFocusing on tomato leaves:")
        print(f"  Healthy classes: {[class_names[i] for i in healthy_classes]}")
        print(f"  Diseased classes: {len(diseased_classes)} disease types")
    else:
        # Use all plants - any class with 'healthy' is healthy, rest are diseased
        healthy_classes = [i for i, name in enumerate(class_names) if 'healthy' in name.lower()]
        diseased_classes = [i for i in range(len(class_names)) if i not in healthy_classes]
        print(f"\nUsing all plant types:")
        print(f"  Healthy classes: {len(healthy_classes)}")
        print(f"  Diseased classes: {len(diseased_classes)}")
    
    # Process and save images
    counters = {split: {'healthy': 0, 'diseased': 0} for split in splits}
    
    print("\nProcessing images...")
    for idx, (image, label) in enumerate(tfds.as_numpy(dataset)):
        if idx % 1000 == 0:
            print(f"  Processed {idx} images...")
        
        label_idx = int(label)
        
        # Filter classes if needed
        if healthy_only and label_idx not in (healthy_classes + diseased_classes):
            continue
        
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
    print("\n" + "="*80)
    print("Dataset preparation complete!")
    print("="*80)
    
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


def verify_data_structure(data_dir='data'):
    """Verify that the data directory structure is correct."""
    
    print("\n" + "="*80)
    print("Verifying data structure...")
    print("="*80)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        return False
    
    required_structure = {
        'train': ['healthy', 'diseased'],
        'val': ['healthy', 'diseased'],
        'test': ['healthy', 'diseased']
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
                num_images = len(list(category_path.glob('*.jpg')))
                print(f"✓ {split}/{category}: {num_images:,} images")
    
    if all_good:
        print("\n✅ Data structure is valid!")
    else:
        print("\n❌ Data structure is incomplete")
    
    return all_good


def main():
    """Main function to prepare data."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Plant Health Classification Data Setup                     ║
║                                                                              ║
║  This script downloads the PlantVillage dataset from TensorFlow Datasets    ║
║  and prepares it for binary classification (healthy vs diseased).           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check if data already exists
    data_dir = 'data'
    if os.path.exists(data_dir) and any(Path(data_dir).iterdir()):
        print(f"\n⚠️  Data directory '{data_dir}' already exists.")
        response = input("Do you want to delete and re-download? (y/N): ")
        if response.lower() == 'y':
            print("Removing existing data...")
            shutil.rmtree(data_dir)
        else:
            print("Keeping existing data. Verifying structure...")
            verify_data_structure(data_dir)
            return
    
    try:
        # Download and prepare data
        counters = download_and_prepare_data(output_dir=data_dir, healthy_only=True)
        
        # Verify structure
        verify_data_structure(data_dir)
        
        print("\n" + "="*80)
        print("Next steps:")
        print("="*80)
        print("\n1. Train models:")
        print("   python train.py")
        print("\n2. Evaluate models:")
        print("   python evaluate.py")
        print("\n3. Compare models:")
        print("   python compare_models.py")
        
    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        print("\nPlease ensure you have tensorflow and tensorflow_datasets installed:")
        print("  pip install tensorflow tensorflow-datasets")
        raise


if __name__ == '__main__':
    main()
