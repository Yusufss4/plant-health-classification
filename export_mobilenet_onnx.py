"""
Export trained MobileNet-v3-Small to ONNX for C++ / ONNX Runtime inference.

Usage:
    python export_mobilenet_onnx.py [--checkpoint PATH] [--output PATH] [--no-verify]

Class indices: 0 = healthy, 1 = diseased (matches PlantHealthDataset).
Input: NCHW float32, batch 1, 3x224x224, ImageNet-normalized (see cpp/README.md).
"""

import argparse
import os

import numpy as np
import torch

from models import create_mobilenet_v3_model

# Must match train.py mobilenet_v3 branch
DROPOUT = 0.2
DEFAULT_CHECKPOINT = "checkpoints/mobilenet_v3_best.pth"
DEFAULT_OUTPUT = "checkpoints/mobilenet_v3.onnx"


def load_weights(model, checkpoint_path: str, device: torch.device) -> None:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    opset: int = 17,
) -> None:
    device = torch.device("cpu")
    model = create_mobilenet_v3_model(num_classes=2, dropout=DROPOUT, pretrained=False)
    load_weights(model, checkpoint_path, device)
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


def verify_onnx(checkpoint_path: str, onnx_path: str) -> None:
    """Compare PyTorch vs ONNX Runtime logits on the same random tensor."""
    import onnxruntime as ort

    device = torch.device("cpu")
    model = create_mobilenet_v3_model(num_classes=2, dropout=DROPOUT, pretrained=False)
    load_weights(model, checkpoint_path, device)
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
    parser = argparse.ArgumentParser(description="Export MobileNet-v3 to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to mobilenet_v3_best.pth",
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
            "`python train.py --model mobilenet_v3` first."
        )

    export_onnx(args.checkpoint, args.output, opset=args.opset)
    if not args.no_verify:
        verify_onnx(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
