"""
Example script demonstrating how to use the plant health classification models.

This script shows:
1. How to create FCNN and ViT models
2. How to perform forward passes
3. Model architecture details
"""

import torch
from models import create_fcnn_model, create_vit_model


def demonstrate_fcnn():
    """Demonstrate FCNN model usage."""
    print("=" * 80)
    print("FULLY CONNECTED NEURAL NETWORK (FCNN) DEMONSTRATION")
    print("=" * 80)
    
    # Create model
    model = create_fcnn_model(num_classes=2, dropout_rate=0.3)
    
    print("\n1. Model Architecture:")
    print("-" * 80)
    print(model)
    
    print("\n2. Model Statistics:")
    print("-" * 80)
    num_params = model.get_num_parameters()
    print(f"Total Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / (1024**3):.2f} GB (float32)")
    
    print("\n3. Layer Breakdown:")
    print("-" * 80)
    for name, param in model.named_parameters():
        print(f"{name:40} {list(param.shape)}")
    
    print("\n4. Forward Pass Example:")
    print("-" * 80)
    
    # Create dummy input batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Channels: 3 (RGB)")
    print(f"  - Height: 224 pixels")
    print(f"  - Width: 224 pixels")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output logits:\n{output}")
    
    # Get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"\nProbabilities (after softmax):")
    for i in range(batch_size):
        print(f"  Sample {i+1}: Healthy={probabilities[i,0]:.4f}, Diseased={probabilities[i,1]:.4f}")
    
    # Get predictions
    predictions = torch.argmax(probabilities, dim=1)
    class_names = ['Healthy', 'Diseased']
    print(f"\nPredictions:")
    for i in range(batch_size):
        print(f"  Sample {i+1}: {class_names[predictions[i]]}")
    
    print("\n" + "=" * 80 + "\n")


def demonstrate_vit():
    """Demonstrate Vision Transformer model usage."""
    print("=" * 80)
    print("VISION TRANSFORMER (ViT) DEMONSTRATION")
    print("=" * 80)
    
    # Create model
    model = create_vit_model(num_classes=2, dropout=0.1)
    
    print("\n1. Model Configuration:")
    print("-" * 80)
    print(f"Image Size: 224×224")
    print(f"Patch Size: 16×16")
    print(f"Number of Patches: {model.patch_embed.num_patches} ({int(224/16)}×{int(224/16)})")
    print(f"Embedding Dimension: 768")
    print(f"Transformer Layers: 12")
    print(f"Attention Heads: 12")
    print(f"MLP Ratio: 4")
    
    print("\n2. Model Statistics:")
    print("-" * 80)
    num_params = model.get_num_parameters()
    print(f"Total Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / (1024**2):.2f} MB (float32)")
    
    print("\n3. Architecture Components:")
    print("-" * 80)
    print("Patch Embedding:")
    print(f"  - Conv2d projection: [3, 768, 16, 16]")
    print(f"  - Learnable class token: [1, 1, 768]")
    print(f"  - Learnable position embeddings: [1, 197, 768] (196 patches + 1 CLS)")
    print("\nTransformer Blocks (×12):")
    print(f"  - Multi-Head Self-Attention (12 heads)")
    print(f"  - Layer Normalization")
    print(f"  - MLP: [768 → 3072 → 768]")
    print(f"  - Residual Connections")
    print("\nClassification Head:")
    print(f"  - Layer Norm")
    print(f"  - Linear: [768 → 2]")
    
    print("\n4. Forward Pass Example:")
    print("-" * 80)
    
    # Create dummy input batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output logits:\n{output}")
    
    # Get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"\nProbabilities (after softmax):")
    for i in range(batch_size):
        print(f"  Sample {i+1}: Healthy={probabilities[i,0]:.4f}, Diseased={probabilities[i,1]:.4f}")
    
    # Get predictions
    predictions = torch.argmax(probabilities, dim=1)
    class_names = ['Healthy', 'Diseased']
    print(f"\nPredictions:")
    for i in range(batch_size):
        print(f"  Sample {i+1}: {class_names[predictions[i]]}")
    
    print("\n" + "=" * 80 + "\n")


def compare_models():
    """Compare FCNN and ViT models."""
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    fcnn = create_fcnn_model(num_classes=2)
    vit = create_vit_model(num_classes=2)
    
    fcnn_params = fcnn.get_num_parameters()
    vit_params = vit.get_num_parameters()
    
    print("\n| Aspect                  | FCNN              | ViT               |")
    print("|" + "-" * 78 + "|")
    print(f"| Parameters              | {fcnn_params:>17,} | {vit_params:>17,} |")
    print(f"| Model Size (float32)    | {fcnn_params * 4 / (1024**3):>14.2f} GB | {vit_params * 4 / (1024**2):>14.2f} MB |")
    print(f"| Input Processing        | {'Flatten to 1D':>17} | {'Patch-based':>17} |")
    print(f"| Architecture            | {'Dense Layers':>17} | {'Transformers':>17} |")
    print(f"| Spatial Info            | {'Lost':>17} | {'Preserved':>17} |")
    
    # Test inference time (rough estimate)
    import time
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # FCNN timing
    fcnn.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = fcnn(dummy_input)
        fcnn_time = (time.time() - start) / 100 * 1000
    
    # ViT timing
    vit.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = vit(dummy_input)
        vit_time = (time.time() - start) / 100 * 1000
    
    print(f"| Inference Time (CPU)    | {fcnn_time:>14.2f} ms | {vit_time:>14.2f} ms |")
    print("=" * 80)
    
    print("\nKey Differences:")
    print("1. FCNN flattens images, losing 2D spatial structure")
    print("2. ViT uses patches, preserving local spatial information")
    print("3. ViT has fewer parameters but is computationally more intensive")
    print("4. ViT uses self-attention to capture global context")
    print("5. FCNN is faster but less accurate for image tasks")


def main():
    """Run all demonstrations."""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "PLANT HEALTH CLASSIFICATION" + " " * 31 + "#")
    print("#" + " " * 24 + "Model Demonstration" + " " * 35 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print("\n")
    
    # Demonstrate FCNN
    demonstrate_fcnn()
    
    # Demonstrate ViT
    demonstrate_vit()
    
    # Compare models
    compare_models()
    
    print("\n")
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nTo train models:")
    print("  python train.py --model fcnn --epochs 50 --batch-size 32")
    print("  python train.py --model vit --epochs 100 --batch-size 16")
    print("\nTo evaluate models:")
    print("  python evaluate.py --model fcnn --weights checkpoints/fcnn_best.pth")
    print("  python evaluate.py --model vit --weights checkpoints/vit_best.pth")
    print("\nTo compare models:")
    print("  python evaluate.py --compare \\")
    print("    --model1 fcnn --weights1 checkpoints/fcnn_best.pth \\")
    print("    --model2 vit --weights2 checkpoints/vit_best.pth")
    print("=" * 80)
    print("\n")


if __name__ == "__main__":
    main()
