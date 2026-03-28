#!/usr/bin/env bash
# Download ONNX Runtime prebuilt archive (CPU) for local development or Pi.
# Usage:
#   bash scripts/download_onnxruntime.sh linux-x64 [version]
#   bash scripts/download_onnxruntime.sh linux-aarch64 [version]
set -euo pipefail

PLATFORM="${1:-linux-x64}"
VERSION="${2:-1.17.3}"
NAME="onnxruntime-${PLATFORM}-${VERSION}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${NAME}.tgz"
DEST="${ONNXRUNTIME_DOWNLOAD_DIR:-$(pwd)/third_party/onnxruntime}"

mkdir -p "${DEST}"
ARCHIVE="${DEST}/${NAME}.tgz"
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Downloading ${URL}"
  curl -fL -o "${ARCHIVE}" "${URL}"
fi
EXTRACTED="${DEST}/${NAME}"
if [[ ! -d "${EXTRACTED}" ]]; then
  tar -xzf "${ARCHIVE}" -C "${DEST}"
fi
echo "Extracted to: ${EXTRACTED}"
echo "Export ONNXRUNTIME_ROOT for CMake:"
echo "  export ONNXRUNTIME_ROOT=${EXTRACTED}"
