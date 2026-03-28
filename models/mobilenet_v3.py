"""
MobileNet-v3-Small for Plant Health Classification

Lightweight architecture for mobile and edge deployment, with ImageNet-pretrained
weights and an adapted classification head for binary healthy vs diseased.
"""

import torch
import torch.nn as nn


class MobileNetV3Small(nn.Module):
    """
    MobileNet-v3-Small from torchvision, adapted for binary classification.

    Args:
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Whether to load ImageNet pretrained weights
        dropout (float): Dropout probability on the classifier head (before final linear)
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        dropout=0.2,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        try:
            from torchvision.models import (
                mobilenet_v3_small,
                MobileNet_V3_Small_Weights,
            )

            if pretrained:
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
                self.backbone = mobilenet_v3_small(weights=weights)
            else:
                self.backbone = mobilenet_v3_small(weights=None)

            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[2] = nn.Dropout(p=dropout, inplace=True)
            self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not load MobileNet-v3-Small from torchvision: {e}\n"
                "Please ensure you have torchvision >= 0.13.0 installed."
            )

    def forward(self, x):
        return self.backbone(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mobilenet_v3_model(num_classes=2, dropout=0.2, pretrained=True):
    """
    Factory for MobileNet-v3-Small.

    Args:
        num_classes: Number of output classes
        dropout: Dropout on the classifier head
        pretrained: ImageNet pretrained weights

    Returns:
        MobileNetV3Small
    """
    return MobileNetV3Small(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("MobileNet-v3-Small — Plant Health Classification")
    print("=" * 80)

    model = create_mobilenet_v3_model(num_classes=2, pretrained=False)

    num_params = model.get_num_parameters()
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Model size (float32): ~{num_params * 4 / (1024**2):.2f} MB")

    batch_size = 4
    dummy = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {dummy.shape}")

    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")

    probs = torch.softmax(out, dim=1)
    names = ["Healthy", "Diseased"]
    preds = torch.argmax(probs, dim=1)
    for i in range(batch_size):
        print(
            f"  Sample {i + 1}: {names[preds[i]]} "
            f"(healthy={probs[i, 0]:.4f}, diseased={probs[i, 1]:.4f})"
        )

    print("\n" + "=" * 80)
