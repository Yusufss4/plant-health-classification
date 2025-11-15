"""
DINOv2 ViT-S/14 for Plant Health Classification

This module implements DINOv2 ViT-S/14 architecture for binary classification
of plant leaves (healthy vs. diseased).

DINOv2 is a self-supervised Vision Transformer trained on a large and diverse dataset,
providing high-quality visual features without labels. The ViT-S/14 variant uses a 
small transformer (21M parameters) with 14x14 patch size.

Key advantages:
1. Self-supervised pretraining on diverse data (ImageNet-22k, web images)
2. High-quality visual features for downstream tasks
3. Robust to domain shift
4. ViT-S/14 variant: 21M parameters, 14x14 patch size
5. Register tokens enhance feature quality
6. Excellent for transfer learning on specialized domains like agriculture

Architecture highlights:
- Pure Vision Transformer (no CNN components)
- 12 transformer layers, 384 embedding dimension
- Patch size 14x14 for fine-grained features
- Self-supervised learning (DINO + iBOT)
- Register tokens improve feature maps
- Supports both frozen features and fine-tuning

Based on: "DINOv2: Learning Robust Visual Features without Supervision"
Paper: https://arxiv.org/abs/2304.07193
"""

import torch
import torch.nn as nn


class DINOv2ViT(nn.Module):
    """
    DINOv2 ViT-S/14 model for image classification.
    
    This loads the pretrained DINOv2 backbone from PyTorch Hub and adds
    a custom classification head for binary plant health classification.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary)
        pretrained (bool): Whether to load pretrained DINOv2 weights (always True for DINOv2)
        use_registers (bool): Whether to use register-enhanced version (dinov2_vits14_reg)
        freeze_backbone (bool): Whether to freeze backbone weights for feature extraction
        dropout (float): Dropout probability for classification head
        use_mlp_head (bool): Use MLP head instead of linear (2-layer with hidden dim)
        gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        use_registers=True,
        freeze_backbone=False,
        dropout=0.1,
        use_mlp_head=False,
        gradient_checkpointing=False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_registers = use_registers
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        
        # Load DINOv2 backbone from PyTorch Hub
        model_name = 'dinov2_vits14_reg' if use_registers else 'dinov2_vits14'
        
        try:
            print(f"Loading {model_name} from PyTorch Hub...")
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
            
            # Get embedding dimension from backbone
            # DINOv2 ViT-S/14 has 384 dimensional embeddings
            embed_dim = self.backbone.embed_dim
            
            # Freeze backbone if requested
            if freeze_backbone:
                print("Freezing backbone weights for feature extraction mode...")
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Enable gradient checkpointing if requested
            if gradient_checkpointing and hasattr(self.backbone, 'set_grad_checkpointing'):
                print("Enabling gradient checkpointing...")
                self.backbone.set_grad_checkpointing(True)
            
            # Create classification head
            if use_mlp_head:
                # 2-layer MLP head: embed_dim -> hidden_dim -> num_classes
                hidden_dim = embed_dim // 2
                self.head = nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, num_classes)
                )
            else:
                # Simple linear head
                self.head = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(embed_dim, num_classes)
                )
            
            print(f"DINOv2 {model_name} loaded successfully!")
            print(f"Embedding dimension: {embed_dim}")
            print(f"Freeze backbone: {freeze_backbone}")
            print(f"Use MLP head: {use_mlp_head}")
            print(f"Gradient checkpointing: {gradient_checkpointing}")
            
        except Exception as e:
            raise RuntimeError(
                f"Could not load DINOv2 model '{model_name}' from PyTorch Hub: {e}\n"
                "Make sure you have internet connection for the first download."
            )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Get features from DINOv2 backbone
        # DINOv2 returns the CLS token embedding
        with torch.set_grad_enabled(not self.freeze_backbone or self.training):
            features = self.backbone(x)
        
        # Apply classification head
        logits = self.head(features)
        
        return logits
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VisionTransformer(DINOv2ViT):
    """
    Alias for DINOv2ViT to maintain backward compatibility.
    
    This allows existing code to work with the new DINOv2 implementation
    while using the VisionTransformer name.
    """
    pass


def create_vit_model(
    num_classes=2, 
    dropout=0.1, 
    pretrained=True, 
    use_registers=True,
    freeze_backbone=False,
    use_mlp_head=False,
    gradient_checkpointing=False
):
    """
    Factory function to create DINOv2 ViT-S/14 model.
    
    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout probability for classification head
        pretrained (bool): Whether to use pretrained DINOv2 weights (always True for DINOv2)
        use_registers (bool): Use register-enhanced version (dinov2_vits14_reg) for better features
        freeze_backbone (bool): Freeze backbone for feature extraction mode
        use_mlp_head (bool): Use 2-layer MLP head instead of linear classifier
        gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency
    
    Returns:
        DINOv2ViT: Initialized DINOv2 ViT-S/14 model
    
    Model Details:
        - DINOv2 ViT-S/14: ~21M parameters (backbone only)
        - Patch size: 14x14 for fine-grained features
        - Embedding dimension: 384
        - Self-supervised pretraining on diverse image data
        - Register tokens enhance feature quality (use_registers=True)
    
    Why DINOv2 over other Vision Transformers:
        1. **Self-Supervised Learning**: Trained without labels, learns robust features
        2. **Domain Robustness**: Better generalization to specialized domains like agriculture
        3. **High-Quality Features**: State-of-the-art feature quality for downstream tasks
        4. **Efficient Transfer Learning**: Excellent for fine-tuning on small datasets
        5. **Register Tokens**: Enhanced feature maps for better classification
        6. **Flexible Modes**: Supports both frozen features and fine-tuning
    
    Usage modes:
        - Fine-tuning (freeze_backbone=False): Train entire model, best accuracy
        - Feature extraction (freeze_backbone=True): Fast training, good for small datasets
        - With registers (use_registers=True): Better feature quality (recommended)
        - MLP head (use_mlp_head=True): More capacity in classification head
    """
    model = DINOv2ViT(
        num_classes=num_classes,
        pretrained=pretrained,
        use_registers=use_registers,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        use_mlp_head=use_mlp_head,
        gradient_checkpointing=gradient_checkpointing
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("=" * 80)
    print("DINOv2 ViT-S/14 Model for Plant Health Classification")
    print("=" * 80)
    
    # Test with pretrained=False to avoid downloading during testing
    model = create_vit_model(num_classes=2, pretrained=False, use_registers=True)
    
    print("\n1. Model Architecture:")
    print("-" * 80)
    print(f"Model Type: DINOv2 ViT-S/14 (with registers)")
    print(f"Purpose: Binary plant health classification (healthy vs. diseased)")
    
    print("\n2. Key Advantages:")
    print("-" * 80)
    print("✓ Self-supervised learning on diverse image data")
    print("✓ High-quality visual features without labels")
    print("✓ Robust to domain shift (great for agriculture)")
    print("✓ ~21M parameters in backbone")
    print("✓ Patch size 14x14 for fine-grained features")
    print("✓ Register tokens enhance feature quality")
    print("✓ Supports both frozen features and fine-tuning")
    
    print("\n3. Model Statistics:")
    print("-" * 80)
    num_params = model.get_num_parameters()
    print(f"Total Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / (1024**2):.2f} MB (float32)")
    
    print("\n4. Why DINOv2 over MobileViT-v2:")
    print("-" * 80)
    print("• Self-supervised pretraining: Better feature quality")
    print("• Domain robustness: Less overfitting to ImageNet biases")
    print("• Transfer learning: Excellent for specialized domains")
    print("• Pure ViT architecture: Better long-range dependencies")
    print("• Register tokens: Enhanced spatial features")
    print("• State-of-the-art: Latest research in self-supervised learning")
    
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
