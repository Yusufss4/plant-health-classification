"""
Data loading and preprocessing utilities for plant health classification.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PlantHealthDataset(Dataset):
    """
    Custom dataset for loading plant leaf images.
    
    Args:
        root_dir (str): Root directory containing 'healthy' and 'diseased' subdirectories
        transform (callable, optional): Optional transform to be applied on images
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Class names and labels
        self.classes = ['healthy', 'diseased']
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


def create_data_loaders(
    train_dir,
    val_dir,
    test_dir,
    batch_size=32,
    num_workers=4,
    pin_memory=True
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
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = PlantHealthDataset(train_dir, transform=get_train_transforms())
    val_dataset = PlantHealthDataset(val_dir, transform=get_val_test_transforms())
    test_dataset = PlantHealthDataset(test_dir, transform=get_val_test_transforms())
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
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
    print("     val/")
    print("       healthy/")
    print("       diseased/")
    print("     test/")
    print("       healthy/")
    print("       diseased/")
    print()
    print("2. Create data loaders:")
    print("   train_loader, val_loader, test_loader = create_data_loaders(")
    print("       'data/train', 'data/val', 'data/test',")
    print("       batch_size=32")
    print("   )")
