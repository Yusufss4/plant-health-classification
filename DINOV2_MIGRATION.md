# Migration from MobileViT v2 to DINOv2 ViT-S/14

## Overview

This document describes the migration from MobileViT v2 to DINOv2 ViT-S/14 for plant health classification.

## What Changed

### Model Architecture
- **Before**: MobileViT v2 (hybrid CNN-Transformer, ~5M parameters)
- **After**: DINOv2 ViT-S/14 (pure ViT, ~21M parameters in backbone)

### Key Features

#### DINOv2 ViT-S/14
- **Self-supervised learning**: Trained on diverse data without labels
- **High-quality features**: Better generalization to specialized domains
- **Register tokens**: Enhanced feature quality (dinov2_vits14_reg)
- **Flexible modes**: 
  - Full fine-tuning (freeze_backbone=False)
  - Feature extraction (freeze_backbone=True)
- **Memory efficient**: Gradient checkpointing option
- **Patch size**: 14x14 for fine-grained features

## API Changes

### Model Creation

#### Before (MobileViT v2)
```python
from models import create_vit_model

model = create_vit_model(
    num_classes=2,
    dropout=0.1,
    pretrained=True,
    variant='100'
)
```

#### After (DINOv2)
```python
from models import create_vit_model

# Basic usage (same as before)
model = create_vit_model(
    num_classes=2,
    dropout=0.1,
    pretrained=True
)

# Advanced usage with new features
model = create_vit_model(
    num_classes=2,
    dropout=0.1,
    pretrained=True,
    use_registers=True,              # NEW: Use register-enhanced version
    freeze_backbone=False,            # NEW: Freeze for feature extraction
    use_mlp_head=False,               # NEW: Use 2-layer MLP head
    gradient_checkpointing=False      # NEW: Enable for memory efficiency
)
```

### New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_registers` | bool | True | Use register-enhanced version (dinov2_vits14_reg) |
| `freeze_backbone` | bool | False | Freeze backbone for feature extraction mode |
| `use_mlp_head` | bool | False | Use 2-layer MLP head instead of linear |
| `gradient_checkpointing` | bool | False | Enable gradient checkpointing for memory |

### Backward Compatibility

The API is backward compatible - existing code will work without changes:
- `VisionTransformer` alias still exists
- `create_vit_model()` function signature extended (new params are optional)
- Same forward pass interface
- Same training pipeline compatibility

## Training Considerations

### Recommended Settings

#### Full Fine-tuning (Best Accuracy)
```python
model = create_vit_model(
    num_classes=2,
    dropout=0.1,
    freeze_backbone=False,
    use_registers=True,
    gradient_checkpointing=True  # If memory constrained
)

# Lower learning rate recommended
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 25
batch_size = 16
```

#### Feature Extraction (Fast Training)
```python
model = create_vit_model(
    num_classes=2,
    dropout=0.1,
    freeze_backbone=True,
    use_registers=True
)

# Higher learning rate okay for head only
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
batch_size = 32
```

### Training Tips

1. **Learning Rate**: Use lower LR (0.0001) for fine-tuning, higher (0.001) for frozen backbone
2. **Batch Size**: DINOv2 works well with smaller batches (16) for fine-tuning
3. **Epochs**: Typically needs fewer epochs (10-25) than MobileViT
4. **Data Augmentation**: Already configured correctly (224x224, ImageNet normalization)
5. **Gradient Checkpointing**: Enable if running out of memory

## Performance Comparison

### Model Size
- **MobileViT v2**: ~5M parameters
- **DINOv2 ViT-S/14**: ~21M parameters (backbone only)
- **With head**: ~21M + small head (linear: 770 params, MLP: ~74K params)

### Expected Benefits
- Better transfer learning to agricultural domain
- More robust to domain shift
- Higher quality features for classification
- Better generalization on small datasets

## Data Preprocessing

No changes required - preprocessing remains the same:
- Image size: 224x224
- Normalization: ImageNet mean/std
- Augmentation: Already configured correctly

## Files Modified

- `models/vit.py`: Complete rewrite with DINOv2
- `models/__init__.py`: Updated exports
- `train.py`: Updated comments
- `compare_models.py`: Updated model references
- `requirements.txt`: Updated comments
- `README.md`: Updated model names and training details

## Migration Checklist

- [x] Replace MobileViT v2 with DINOv2 implementation
- [x] Add flexible classification head options
- [x] Support frozen/fine-tuning modes
- [x] Enable gradient checkpointing
- [x] Maintain backward compatibility
- [x] Update documentation
- [x] Verify preprocessing (224x224, ImageNet norm)
- [x] Test model creation and forward pass
- [x] Test training compatibility
- [x] Run security checks

## Testing

All tests pass:
- ✅ Model creation with various configurations
- ✅ Forward pass with different batch sizes
- ✅ Backward pass and gradient flow
- ✅ Frozen backbone mode
- ✅ Full fine-tuning mode
- ✅ State dict save/load
- ✅ Device compatibility
- ✅ Integration with existing training pipeline
- ✅ Security scan (CodeQL)

## Support

For issues or questions:
1. Check this migration guide
2. Review model documentation in `models/vit.py`
3. Run tests: `python /tmp/test_dinov2_model.py`
4. Open an issue on GitHub

## References

- DINOv2 Paper: https://arxiv.org/abs/2304.07193
- PyTorch Hub: https://github.com/facebookresearch/dinov2
- Original MobileViT-v2: https://arxiv.org/abs/2206.02680
