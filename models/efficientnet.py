"""
EfficientNet-B0 for Plant Health Classification

This module implements EfficientNet-B0 architecture for binary classification
of plant leaves (healthy vs. diseased).

EfficientNet-B0 is a modern CNN architecture that achieves state-of-the-art accuracy
with significantly fewer parameters through compound scaling and efficient design.

Key advantages:
- Efficient architecture with only ~5.3M parameters
- Uses compound scaling (depth, width, resolution)
- Mobile inverted bottleneck convolutions (MBConv)
- Squeeze-and-Excitation blocks for channel attention
- Pretrained on ImageNet for transfer learning
- Better accuracy-efficiency trade-off than traditional CNNs

Based on: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
Paper: https://arxiv.org/abs/1905.11946
"""

import torch
import torch.nn as nn


class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 model for image classification.
    
    This is a wrapper around torchvision's EfficientNet-B0 implementation,
    adapted for binary plant health classification.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary)
        pretrained (bool): Whether to load ImageNet pretrained weights
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Import here to avoid dependency issues
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            
            # Create base model
            if pretrained:
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                self.backbone = efficientnet_b0(weights=weights)
            else:
                self.backbone = efficientnet_b0(weights=None)
            
            # Get the number of features from the classifier
            num_features = self.backbone.classifier[1].in_features
            
            # Replace classifier for binary classification
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(num_features, num_classes)
            )
            
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not load EfficientNet-B0 from torchvision: {e}\n"
                "Please ensure you have torchvision >= 0.13.0 installed."
            )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_model(num_classes=2, dropout=0.2, pretrained=True):
    """
    Factory function to create EfficientNet-B0 model.
    
    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
        pretrained (bool): Whether to use ImageNet pretrained weights
    
    Returns:
        EfficientNetB0: Initialized EfficientNet-B0 model
    
    Why EfficientNet-B0 over traditional FCNN:
        1. **Spatial Structure**: Preserves 2D image structure (no flattening)
        2. **Parameter Efficient**: ~5.3M parameters with high accuracy
        3. **Compound Scaling**: Balanced scaling of depth, width, and resolution
        4. **Mobile Inverted Bottlenecks**: Efficient convolutions
        5. **SE Blocks**: Channel-wise attention improves feature learning
        6. **Pretrained**: ImageNet weights provide strong visual features
        7. **State-of-the-art**: Better accuracy than traditional CNNs
        8. **Mobile-Friendly**: Optimized for inference on resource-constrained devices
    """
    model = EfficientNetB0(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("=" * 80)
    print("EfficientNet-B0 Model for Plant Health Classification")
    print("=" * 80)
    
    model = create_cnn_model(num_classes=2, pretrained=False)
    
    print("\n1. Model Architecture:")
    print("-" * 80)
    print(f"Model Type: EfficientNet-B0")
    print(f"Purpose: Binary plant health classification (healthy vs. diseased)")
    
    print("\n2. Key Advantages:")
    print("-" * 80)
    print("✓ Preserves spatial structure (no flattening)")
    print("✓ Efficient compound scaling (depth, width, resolution)")
    print("✓ Mobile inverted bottleneck convolutions (MBConv)")
    print("✓ Squeeze-and-Excitation blocks for attention")
    print("✓ Only ~5.3M parameters with state-of-the-art accuracy")
    print("✓ Pretrained on ImageNet for transfer learning")
    print("✓ Better accuracy-efficiency trade-off than traditional FCNN")
    
    print("\n3. Model Statistics:")
    print("-" * 80)
    num_params = model.get_num_parameters()
    print(f"Total Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / (1024**2):.2f} MB (float32)")
    
    print("\n4. Why EfficientNet-B0 over Traditional FCNN:")
    print("-" * 80)
    print("• Traditional FCNN: Flattens images, loses spatial structure, 307M parameters")
    print("• EfficientNet-B0: Preserves structure, efficient design, 5.3M parameters")
    print("• Efficiency gain: ~58x fewer parameters")
    print("• Better accuracy with modern architecture")
    print("• Suitable for mobile deployment")
    print("• Leverages ImageNet pretraining")
    
    print("\n5. Forward Pass Example:")
    print("-" * 80)
    
    # Create dummy input batch
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
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
    
    print("\n" + "=" * 80)
    print("Model ready for training on plant health classification task!")
    print("=" * 80)
