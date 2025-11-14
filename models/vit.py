"""
Vision Transformer (ViT) for Plant Health Classification

This module implements a Vision Transformer architecture for binary classification
of plant leaves (healthy vs. diseased).

Based on: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them.
    
    Args:
        image_size (int): Size of input image (assumes square)
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels (3 for RGB)
        embed_dim (int): Embedding dimension
    """
    
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolutional layer acts as patch extraction and linear projection
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            Patch embeddings [batch_size, num_patches, embed_dim]
        """
        # x: [B, C, H, W]
        x = self.projection(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, num_patches, embed_dim]
        
        Returns:
            Output tensor [batch_size, num_patches, embed_dim]
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, N, 3*embed_dim]
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        attention_output = attention_probs @ v  # [B, heads, N, head_dim]
        attention_output = attention_output.transpose(1, 2)  # [B, N, heads, head_dim]
        attention_output = attention_output.reshape(batch_size, num_patches, embed_dim)
        
        # Final projection
        output = self.projection(attention_output)
        output = self.projection_dropout(output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Transformer encoder block.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (int): Ratio of MLP hidden dim to embedding dim
        dropout (float): Dropout probability
    """
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, num_patches, embed_dim]
        
        Returns:
            Output tensor [batch_size, num_patches, embed_dim]
        """
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification.
    
    Args:
        image_size (int): Size of input image
        patch_size (int): Size of each patch
        num_classes (int): Number of output classes
        dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        heads (int): Number of attention heads
        mlp_dim (int): MLP hidden dimension
        dropout (float): Dropout probability
        emb_dropout (float): Embedding dropout probability
    """
    
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=2,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(emb_dropout)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim // dim, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.projection.weight)
        nn.init.constant_(self.patch_embed.projection.bias, 0)
        
        # Initialize positional embedding and class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize classification head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            Class logits [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Classification (use class token)
        cls_output = x[:, 0]  # [B, dim]
        logits = self.head(cls_output)  # [B, num_classes]
        
        return logits
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_vit_model(num_classes=2, dropout=0.1):
    """
    Factory function to create Vision Transformer model.
    
    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout probability
    
    Returns:
        VisionTransformer: Initialized ViT model
    """
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=dropout,
        emb_dropout=dropout
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_vit_model(num_classes=2)
    
    # Print model info
    print("Vision Transformer Model:")
    print("=" * 60)
    print(f"Image Size: 224x224")
    print(f"Patch Size: 16x16")
    print(f"Number of Patches: {model.patch_embed.num_patches}")
    print(f"Embedding Dimension: 768")
    print(f"Transformer Layers: 12")
    print(f"Attention Heads: 12")
    print("=" * 60)
    
    # Print parameter count
    num_params = model.get_num_parameters()
    print(f"\nTotal Parameters: {num_params:,}")
    print(f"Model Size: ~{num_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput Shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
    print(f"Output (logits):\n{output}")
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"\nProbabilities:\n{probabilities}")
    
    # Get predictions
    predictions = torch.argmax(probabilities, dim=1)
    print(f"\nPredictions: {predictions}")
