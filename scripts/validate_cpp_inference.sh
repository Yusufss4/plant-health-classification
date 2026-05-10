#!/usr/bin/env bash
# Compare C++ ORT inference (--tensor-bin) with Python reference on the same tensor.
# Requires: ONNX exported, phc_infer_mobilenet built, python deps, test image.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="${ONNX:-$ROOT/checkpoints/mobilenet_v3.onnx}"
BIN="${TMPDIR:-/tmp}/mobilenet_validate_input.bin"
IMAGE="${1:-}"
INFER_BIN="${2:-}"

usage() {
  cat <<'EOF'
Usage: scripts/validate_cpp_inference.sh <image> <infer_bin>

Compares Python ORT reference logits with C++ logits using the same input tensor.

Environment:
  ONNX               Override model path (default: checkpoints/mobilenet_v3.onnx)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 2 ]]; then
  echo "Expected exactly 2 arguments: <image> <infer_bin>." >&2
  usage
  exit 2
fi

if [[ ! -f "$ONNX" ]]; then
  echo "Missing $ONNX — run: python export_mobilenet_onnx.py"
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "Image file not found: $IMAGE"
  echo "Usage: $0 <image> <infer_bin>"
  exit 1
fi

python3 "$ROOT/scripts/dump_ort_reference.py" --image "$IMAGE" --onnx "$ONNX" --tensor-out "$BIN"
echo "--- C++ (should match ORT logits above) ---"
if [[ -z "${ONNXRUNTIME_ROOT:-}" ]]; then
  echo "Set ONNXRUNTIME_ROOT and build cpp/ first (see cpp/README.md)"
  exit 1
fi

if [[ ! -x "$INFER_BIN" ]]; then
  echo "Missing executable: $INFER_BIN"
  echo "Build it with: cd cpp && cmake --preset local-release && cmake --build --preset local-release"
  echo "Pass the executable path as the second argument."
  exit 1
fi

"$INFER_BIN" "$ONNX" --tensor-bin "$BIN"
