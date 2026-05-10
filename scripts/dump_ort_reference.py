#!/usr/bin/env python3
"""
Save NCHW float32 tensor (same layout as cpp/infer_mobilenet --tensor-bin) and
print ONNX Runtime logits — use to validate C++ preprocessing vs PyTorch pipeline.

Usage:
  python scripts/dump_ort_reference.py --image path.jpg --onnx checkpoints/mobilenet_v3.onnx \\
      --tensor-out /tmp/mobilenet_input.bin
"""

import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--onnx", required=True)
    p.add_argument("--tensor-out", default="mobilenet_input.bin")
    args = p.parse_args()

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    img = Image.open(args.image).convert("RGB")
    t = tfm(img).numpy().astype(np.float32)  # 3,224,224
    batch = np.expand_dims(t, axis=0)  # 1,3,224,224
    batch.tofile(args.tensor_out)
    print(f"Wrote {batch.shape} float32 tensor to {args.tensor_out}")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"input": batch})[0]
    print(f"ORT logits: [{logits[0, 0]:.6f}, {logits[0, 1]:.6f}]")


if __name__ == "__main__":
    main()
