#!/usr/bin/env bash
# Compare C++ ORT inference (--tensor-bin) with Python reference on the same tensor.
# Requires: ONNX exported, infer_mobilenet built, python deps, test image.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="${ONNX:-$ROOT/checkpoints/mobilenet_v3.onnx}"
IMAGE="${1:-$ROOT/data/test/healthy/$(ls "$ROOT/data/test/healthy" 2>/dev/null | head -1)}"
BIN="${TMPDIR:-/tmp}/mobilenet_validate_input.bin"

if [[ ! -f "$ONNX" ]]; then
  echo "Missing $ONNX — run: python export_mobilenet_onnx.py"
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "Usage: $0 [path/to/image.jpg]"
  echo "Default image not found. Pass a JPEG/PNG path."
  exit 1
fi

python3 "$ROOT/scripts/dump_ort_reference.py" --image "$IMAGE" --onnx "$ONNX" --tensor-out "$BIN"
echo "--- C++ (should match ORT logits above) ---"
if [[ -z "${ONNXRUNTIME_ROOT:-}" ]]; then
  echo "Set ONNXRUNTIME_ROOT and build cpp/ first (see cpp/README.md)"
  exit 1
fi
"$ROOT/cpp/build/infer_mobilenet" "$ONNX" --tensor-bin "$BIN"
