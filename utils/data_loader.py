"""
Data loading and preprocessing utilities for plant health classification.
"""

import os
from collections import Counter

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# Canonical class ordering for the 3-class model. The order is fixed (not
# alphabetical) so that label indices stay stable across PyTorch <-> ONNX <-> C++.
DEFAULT_CLASSES = ['healthy', 'diseased', 'background']


class PlantHealthDataset(Dataset):
    """
    Custom dataset for loading plant leaf images.

    Args:
        root_dir (str): Root directory containing class subdirectories
            ('healthy', 'diseased', 'background').
        transform (callable, optional): Optional transform to be applied on images.
        classes (list[str], optional): Override the class list. Defaults to
            ``DEFAULT_CLASSES``. If a class subdirectory is missing, no samples
            are loaded for that class, but ``self.classes`` and ``class_to_idx``
            are unchanged so indices stay aligned with the 3-class model (e.g.
            legacy ``healthy/`` + ``diseased/`` only still use labels 0 and 1;
            ``background`` remains index 2 with zero samples).
    """

    def __init__(self, root_dir, transform=None, classes=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = list(classes) if classes is not None else list(DEFAULT_CLASSES)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transforms():
    """
    Get training data transformations with augmentation.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def get_val_test_transforms():
    """
    Get validation/test data transformations without augmentation.
    
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def _make_balanced_sampler(dataset):
    """
    Build a WeightedRandomSampler that yields each class with equal probability.

    With three classes of uneven on-disk size (e.g. ~10k healthy / 27k diseased /
    21k background), uniform shuffling would expose ``diseased`` ~2.7x more often
    than ``healthy``. The weighted sampler counters that by weighting each sample
    by ``1 / count[label]`` so the expected per-batch label distribution is
    uniform. ``num_samples`` is kept at ``len(dataset)`` so one "epoch" still
    means roughly one full pass over the data on disk.
    """
    counts = Counter(label for _, label in dataset.samples)
    if not counts:
        return None
    sample_weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def create_data_loaders(
    train_dir,
    val_dir,
    test_dir,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    balanced_sampler=True,
    classes=None,
):
    """
    Create data loaders for training, validation, and testing.

    Args:
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        test_dir (str): Path to test data directory
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        balanced_sampler (bool): If True, the train loader uses a
            ``WeightedRandomSampler`` that samples each class with equal
            probability. Val and test stay sequential so their metrics reflect
            the real class distribution.
        classes (list[str], optional): Class ordering override. Defaults to the
            project-wide ``DEFAULT_CLASSES``.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_dataset = PlantHealthDataset(
        train_dir, transform=get_train_transforms(), classes=classes
    )
    val_dataset = PlantHealthDataset(
        val_dir, transform=get_val_test_transforms(), classes=classes
    )
    test_dataset = PlantHealthDataset(
        test_dir, transform=get_val_test_transforms(), classes=classes
    )

    sampler = _make_balanced_sampler(train_dataset) if balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_stats(data_dir):
    """
    Get statistics about the dataset.
    
    Args:
        data_dir (str): Path to data directory
    
    Returns:
        dict: Dictionary containing dataset statistics
    """
    dataset = PlantHealthDataset(data_dir)
    
    stats = {
        'total_samples': len(dataset),
        'classes': dataset.classes,
        'class_distribution': {}
    }
    
    # Count samples per class
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        stats['class_distribution'][class_name] = \
            stats['class_distribution'].get(class_name, 0) + 1
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Data Loader Configuration:")
    print("=" * 60)
    
    # Show training transforms
    print("\nTraining Transforms:")
    train_transforms = get_train_transforms()
    print(train_transforms)
    
    # Show validation/test transforms
    print("\nValidation/Test Transforms:")
    val_test_transforms = get_val_test_transforms()
    print(val_test_transforms)
    
    print("\nTo use:")
    print("1. Organize data in the following structure:")
    print("   data/")
    print("     train/")
    print("       healthy/")
    print("       diseased/")
    print("       background/")
    print("     val/")
    print("       healthy/")
    print("       diseased/")
    print("       background/")
    print("     test/")
    print("       healthy/")
    print("       diseased/")
    print("       background/")
    print()
    print("2. Create data loaders:")
    print("   train_loader, val_loader, test_loader = create_data_loaders(")
    print("       'data/train', 'data/val', 'data/test',")
    print("       batch_size=32")
    print("   )")


