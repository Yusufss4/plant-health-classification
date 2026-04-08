#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Deploy binaries + model + dataset to a Raspberry Pi (Zero 2 W friendly).

What it copies:
  - C++ build outputs (tools + optionally live_infer_web)
  - ONNX model file
  - Dataset folder (expects healthy/ and diseased/ subfolders somewhere under it)
  - Live web page (web/live/index.html) into deploy/artifacts for serving
  - (Optional) ONNX Runtime shared libs into deploy/lib

Usage:
  scripts/deploy_rpi_zero2w.sh --host <pi-host-or-ip> [options]

Required:
  --host <host>            Pi hostname or IP

Common options:
  --user <user>            SSH username (default: pi)
  --port <port>            SSH port (default: 22)
  --dest <dir>             Remote deploy dir (default: ~/phc_deploy)
  --build-dir <dir>        Local build dir (default: cpp/build/rpi-zero2w-release)
  --model <file.onnx>      Local ONNX model (default: checkpoints/mobilenet_v3.onnx)
  --data <dir>             Local dataset root (default: data/test)
  --ort-root <dir>         Local ONNX Runtime root; if set, copies libonnxruntime.so* to dest/lib
  --no-delete              Do not delete remote files not present locally
  --dry-run                Show what would be transferred
  -h, --help               Show help

Examples:
  scripts/deploy_rpi_zero2w.sh --host rpi-zero2w.local
  scripts/deploy_rpi_zero2w.sh --host 192.168.1.50 --dest ~/phc --data data/test
  scripts/deploy_rpi_zero2w.sh --host rpi-zero2w.local --ort-root third_party/onnxruntime/onnxruntime-linux-aarch64-1.17.3

After deploy (on the Pi):
  cd ~/phc_deploy
  export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH}"
  ./phc_evaluate_mobilenet model/mobilenet_v3.onnx data/test

Live preview (on the Pi):
  ./bin/live_infer_web ./model/mobilenet_v3.onnx ./artifacts
  python3 -m http.server 8080 --bind 0.0.0.0 --directory ./artifacts
EOF
}

HOST=""
USER="pi"
PORT="22"
DEST="~/phc_deploy"
BUILD_DIR="cpp/build/rpi-zero2w-release"
MODEL="checkpoints/mobilenet_v3.onnx"
DATA="data/test"
WEB_DIR="web/live"
ORT_ROOT=""
DELETE_FLAG="--delete"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --user) USER="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --data) DATA="${2:-}"; shift 2 ;;
    --ort-root) ORT_ROOT="${2:-}"; shift 2 ;;
    --no-delete) DELETE_FLAG=""; shift ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${HOST}" ]]; then
  echo "--host is required" >&2
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${REPO_ROOT}/${BUILD_DIR}"
MODEL="${REPO_ROOT}/${MODEL}"
DATA="${REPO_ROOT}/${DATA}"
WEB_DIR="${REPO_ROOT}/${WEB_DIR}"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "Build dir not found: ${BUILD_DIR}" >&2
  exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "Model not found: ${MODEL}" >&2
  exit 2
fi
if [[ ! -d "${DATA}" ]]; then
  echo "Dataset dir not found: ${DATA}" >&2
  exit 2
fi
if [[ ! -f "${WEB_DIR}/index.html" ]]; then
  echo "Live web page not found: ${WEB_DIR}/index.html" >&2
  exit 2
fi

REMOTE="${USER}@${HOST}"
SSH="ssh -p ${PORT}"
RSYNC_SSH=(-e "ssh -p ${PORT}")

echo "==> Preparing remote directories at ${REMOTE}:${DEST}"
# NOTE: don't quote ${DEST}/... here; if DEST contains "~", quoting prevents tilde expansion
# and would create directories under a literal "~" folder.
${SSH} "${REMOTE}" "mkdir -p ${DEST}/bin ${DEST}/model ${DEST}/data ${DEST}/lib ${DEST}/artifacts"

echo "==> Syncing binaries from ${BUILD_DIR}"
# Copy only the executables we care about; ignore intermediate build files.
BIN_FILES=(
  phc_infer_mobilenet
  phc_evaluate_mobilenet
  live_infer_web
  phc_tests
)

TMP_BIN_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_BIN_DIR}"; }
trap cleanup EXIT

for b in "${BIN_FILES[@]}"; do
  if [[ -f "${BUILD_DIR}/${b}" ]]; then
    cp -f "${BUILD_DIR}/${b}" "${TMP_BIN_DIR}/"
  fi
done

if [[ -z "$(ls -A "${TMP_BIN_DIR}" 2>/dev/null || true)" ]]; then
  echo "No binaries found in ${BUILD_DIR}. Did you build the preset?" >&2
  exit 2
fi

rsync -avz ${DRY_RUN} ${DELETE_FLAG} "${RSYNC_SSH[@]}" \
  "${TMP_BIN_DIR}/" "${REMOTE}:${DEST}/bin/"

echo "==> Syncing model"
rsync -avz ${DRY_RUN} "${RSYNC_SSH[@]}" \
  "${MODEL}" "${REMOTE}:${DEST}/model/mobilenet_v3.onnx"

echo "==> Syncing dataset"
rsync -avz ${DRY_RUN} ${DELETE_FLAG} "${RSYNC_SSH[@]}" \
  "${DATA}/" "${REMOTE}:${DEST}/data/test/"

echo "==> Syncing live web page"
rsync -avz ${DRY_RUN} "${RSYNC_SSH[@]}" \
  "${WEB_DIR}/index.html" "${REMOTE}:${DEST}/artifacts/index.html"

if [[ -n "${ORT_ROOT}" ]]; then
  ORT_ROOT="${REPO_ROOT}/${ORT_ROOT}"
  if [[ ! -d "${ORT_ROOT}/lib" ]]; then
    echo "ONNX Runtime lib dir not found: ${ORT_ROOT}/lib" >&2
    exit 2
  fi
  echo "==> Syncing ONNX Runtime libs from ${ORT_ROOT}/lib"
  rsync -avz ${DRY_RUN} "${RSYNC_SSH[@]}" \
    "${ORT_ROOT}/lib/libonnxruntime.so"* "${REMOTE}:${DEST}/lib/" || true
fi

echo "==> Done."
echo "On the Pi:"
echo "  cd ${DEST}"
echo "  export LD_LIBRARY_PATH=\"\$PWD/lib:\${LD_LIBRARY_PATH}\""
echo "  ./bin/phc_evaluate_mobilenet ./model/mobilenet_v3.onnx ./data/test"
echo ""
echo "Live preview (on the Pi):"
echo "  ./bin/live_infer_web ./model/mobilenet_v3.onnx ./artifacts"
echo "  python3 -m http.server 8080 --bind 0.0.0.0 --directory ./artifacts"

