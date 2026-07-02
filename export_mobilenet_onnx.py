"""
Export trained MobileNet-v3-Small to ONNX for C++ / ONNX Runtime inference.

Usage:
    python export_mobilenet_onnx.py [--checkpoint PATH] [--output PATH] [--no-verify]

Class indices (must match utils.data_loader.DEFAULT_CLASSES):
    0 = healthy
    1 = diseased
    2 = background
Input: NCHW float32, batch 1, 3x224x224, ImageNet-normalized (see cpp/README.md).
"""

import argparse
import json
import os

import numpy as np
import onnx
import torch

from models import build_model, get_model_spec
from utils import DEFAULT_CLASSES

MODEL_TYPE = "mobilenet_v3"
DEFAULT_CHECKPOINT = "checkpoints/mobilenet_v3_3cls_best.pth"
DEFAULT_OUTPUT = "checkpoints/mobilenet_v3_3cls.onnx"


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    """Load a training checkpoint dict from disk."""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def checkpoint_class_info(ckpt: dict) -> tuple[int, list[str]]:
    num_classes = int(ckpt.get("num_classes", len(DEFAULT_CLASSES)))
    class_names = list(ckpt.get("class_names", DEFAULT_CLASSES))
    if len(class_names) != num_classes:
        raise ValueError(
            f"Checkpoint class_names length ({len(class_names)}) != "
            f"num_classes ({num_classes})"
        )
    return num_classes, class_names


def load_weights(model, checkpoint_path: str, device: torch.device) -> dict:
    ckpt = load_checkpoint(checkpoint_path, device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return ckpt


def attach_onnx_metadata(
    onnx_path: str,
    class_names: list[str],
    num_classes: int,
) -> None:
    """Embed class labels in ONNX metadata_props for deployment tooling."""
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    props = {
        "num_classes": str(num_classes),
        "class_names": ",".join(class_names),
        "class_names_json": json.dumps(class_names),
    }
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, onnx_path)
    print(
        f"Attached ONNX metadata: num_classes={num_classes}, "
        f"class_names={class_names}"
    )


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset: int = 17,
) -> tuple[int, list[str]]:
    """Export checkpoint weights to ONNX; returns class count and names."""
    device = torch.device("cpu")
    ckpt = load_checkpoint(checkpoint_path, device)
    num_classes, class_names = checkpoint_class_info(ckpt)

    dropout = get_model_spec(MODEL_TYPE).dropout
    model = build_model(
        MODEL_TYPE,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=False,
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    export_kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )
    try:
        torch.onnx.export(
            model,
            dummy,
            output_path,
            dynamo=False,  # TorchScript exporter; avoids onnxscript on newer PyTorch defaults
            **export_kwargs,
        )
    except TypeError:
        torch.onnx.export(model, dummy, output_path, **export_kwargs)
    print(f"Exported ONNX to {output_path}")

    attach_onnx_metadata(output_path, class_names, num_classes)
    return num_classes, class_names


def verify_onnx(
    checkpoint_path: str,
    onnx_path: str,
    num_classes: int,
) -> None:
    """Compare PyTorch vs ONNX Runtime logits on the same random tensor."""
    import onnxruntime as ort

    device = torch.device("cpu")
    ckpt = load_checkpoint(checkpoint_path, device)

    dropout = get_model_spec(MODEL_TYPE).dropout
    model = build_model(
        MODEL_TYPE,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=False,
    )
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    np.random.seed(0)
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        pt_out = model(x).numpy()

    sess = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    ort_out = sess.run(
        None, {"input": x.numpy().astype(np.float32)}
    )[0]

    diff = np.max(np.abs(pt_out - ort_out))
    print(f"PyTorch vs ONNX Runtime max abs diff: {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError(
            f"Parity check failed (max diff {diff} > 1e-3). "
            "Check export settings and onnxruntime version."
        )
    print("Parity check passed.")


def main():
    """CLI entry: export checkpoint to ONNX and optionally verify parity."""
    parser = argparse.ArgumentParser(description="Export MobileNet-v3 to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to mobilenet_v3_3cls_best.pth",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output .onnx path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip PyTorch vs ONNX Runtime parity check",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train with "
            "`python train.py` first."
        )

    num_classes, _ = export_onnx(args.checkpoint, args.output, opset=args.opset)
    if not args.no_verify:
        verify_onnx(args.checkpoint, args.output, num_classes)


if __name__ == "__main__":
    main()
