"""
Dynamic INT8 weight quantization of MobileNet ONNX for smaller/faster inference on Pi.

Requires: pip install onnx onnxruntime

Usage:
    python quantize_mobilenet_onnx.py [--input checkpoints/mobilenet_v3.onnx] \\
        [--output checkpoints/mobilenet_v3_int8.onnx]

Smoke-test on CPU:
    python -c "import onnxruntime as ort; ort.InferenceSession('checkpoints/mobilenet_v3_int8.onnx')"
"""

import argparse
import os

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    p = argparse.ArgumentParser(description="Dynamic quantize MobileNet ONNX (weights to INT8)")
    p.add_argument("--input", default="checkpoints/mobilenet_v3.onnx")
    p.add_argument("--output", default="checkpoints/mobilenet_v3_int8.onnx")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Missing {args.input}. Run export_mobilenet_onnx.py first.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        weight_type=QuantType.QUInt8,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
