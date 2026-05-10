"""
DINOv3 ViT-S/14 for Plant Health Classification

This module implements DINOv3 ViT-S/14 architecture for binary classification
of plant leaves (healthy vs. diseased).

DINOv3 is the next generation of self-supervised Vision Transformers, trained on 
even larger and more diverse datasets, providing superior visual features without labels. 
The ViT-S/14 variant uses a small transformer (21M parameters) with 14x14 patch size.

Key advantages:
1. Enhanced self-supervised pretraining on massive diverse datasets
2. Superior visual features and improved downstream task performance
3. Even better robustness to domain shift
4. ViT-S/14 variant: 21M parameters, 14x14 patch size
5. Improved register tokens for better feature quality
6. Excellent for transfer learning on specialized domains like agriculture

Architecture highlights:
- Pure Vision Transformer (no CNN components)
- 12 transformer layers, 384 embedding dimension
- Patch size 14x14 for fine-grained features
- Advanced self-supervised learning techniques
- Enhanced register tokens improve feature maps
- Supports both frozen features and fine-tuning

Based on: "DINOv3: Scaling Self-Supervised Vision Transformers"
Paper: https://arxiv.org/abs/2508.10104
Website: https://ai.meta.com/dinov3/
"""

import torch
import torch.nn as nn


class DINOv3ViT(nn.Module):
    """
    DINOv3 ViT-S/14 model for image classification.
    
    This loads the pretrained DINOv3 backbone from timm library and adds
    a custom classification head for binary plant health classification.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for binary)
        pretrained (bool): Whether to load pretrained DINOv3 weights (always True for DINOv3)
        use_registers (bool): Whether to use register-enhanced version (vit_small_patch14_reg4_dinov3)
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
        
        # Load DINOv3 backbone from timm library
        # DINOv3 models are available in timm >= 1.0.20
        if use_registers:
        # "plus" is the slightly larger small model; you can pick base/large as well
            model_name = "vit_small_plus_patch16_dinov3.lvd1689m"
        else:
            model_name = "vit_small_patch16_dinov3.lvd1689m"
        
        try:
            import timm
            print(f"Loading {model_name} from timm library...")
            
            # Create model without classification head (num_classes=0)
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool=''  # We'll use the CLS token
            )
            
            # Get embedding dimension from backbone
            # DINOv3 ViT-S/14 has 384 dimensional embeddings
            embed_dim = self.backbone.embed_dim
            
            # Freeze backbone if requested
            if freeze_backbone:
                print("Freezing backbone weights for feature extraction mode...")
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Enable gradient checkpointing if requested
            if gradient_checkpointing:
                print("Enabling gradient checkpointing...")
                if hasattr(self.backbone, 'set_grad_checkpointing'):
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
            
            print(f"DINOv3 {model_name} loaded successfully!")
            print(f"Embedding dimension: {embed_dim}")
            print(f"Freeze backbone: {freeze_backbone}")
            print(f"Use MLP head: {use_mlp_head}")
            print(f"Gradient checkpointing: {gradient_checkpointing}")
            
        except ImportError:
            raise ImportError(
                "timm library is required for DINOv3 models. "
                "Install with: pip install timm>=1.0.20"
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load DINOv3 model '{model_name}' from timm: {e}\n"
                "Make sure you have timm>=1.0.20 installed and internet connection for the first download."
            )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Get features from DINOv3 backbone
        # DINOv3 returns features, we need to extract the CLS token
        with torch.set_grad_enabled(not self.freeze_backbone or self.training):
            features = self.backbone.forward_features(x)
            # Extract CLS token (first token)
            if isinstance(features, dict):
                features = features['x_norm_clstoken']
            else:
                features = features[:, 0]  # CLS token is the first token
        
        # Apply classification head
        logits = self.head(features)
        
        return logits
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VisionTransformer(DINOv3ViT):
    """
    Alias for DINOv3ViT to maintain backward compatibility.
    
    This allows existing code to work with the new DINOv3 implementation
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
    Factory function to create DINOv3 ViT-S/14 model.
    
    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout probability for classification head
        pretrained (bool): Whether to use pretrained DINOv3 weights (always True for DINOv3)
        use_registers (bool): Use register-enhanced version for better features
        freeze_backbone (bool): Freeze backbone for feature extraction mode
        use_mlp_head (bool): Use 2-layer MLP head instead of linear classifier
        gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency
    
    Returns:
        DINOv3ViT: Initialized DINOv3 ViT-S/14 model
    """
    model = DINOv3ViT(
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
    print("DINOv3 ViT-S/14 Model for Plant Health Classification")
    print("=" * 80)
    
    # Test with pretrained=False to avoid downloading during testing
    model = create_vit_model(num_classes=2, pretrained=False, use_registers=True)
    
    print("\nModel Statistics:")
    print("-" * 80)
    num_params = model.get_num_parameters()
    print(f"Total Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / (1024**2):.2f} MB (float32)")
    
    print("\nForward Pass Example:")
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
