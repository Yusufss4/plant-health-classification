#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Sync a Raspberry Pi sysroot locally and fix absolute symlinks.

Defaults are suitable for Pi Zero 2 W (64-bit / aarch64) sysroots.

Usage:
  scripts/sync_rpi_sysroot.sh --host <pi-host-or-ip> [--user <user>] [--dest <sysroot-dir>]

Options:
  --host <host>     Pi hostname or IP (required)
  --user <user>     SSH username (default: pi)
  --dest <dir>      Local sysroot directory (default: $HOME/sysroots/rpi-zero2w)
  --port <port>     SSH port (default: 22)
  --no-delete       Do not delete files not present on the Pi
  --dry-run         Show what would be transferred
  -h, --help        Show help

Examples:
  scripts/sync_rpi_sysroot.sh --host 192.168.1.50
  scripts/sync_rpi_sysroot.sh --host rpi-zero2w.local --user pi --dest ~/sysroots/rpi-zero2w

After syncing, you can use it with CMake like:
  export RPI_SYSROOT="$HOME/sysroots/rpi-zero2w"
  cmake --preset rpi-zero2w-release
EOF
}

HOST=""
USER="pi"
DEST="${HOME}/sysroots/rpi-zero2w"
PORT="22"
DELETE_FLAG="--delete"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="${2:-}"; shift 2 ;;
    --user) USER="${2:-}"; shift 2 ;;
    --dest) DEST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
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

mkdir -p "${DEST}"

SSH="ssh -p ${PORT}"
REMOTE="${USER}@${HOST}"

COMMON_EXCLUDES=(
  --exclude '/dev/*'
  --exclude '/proc/*'
  --exclude '/sys/*'
  --exclude '/tmp/*'
  --exclude '/run/*'
  --exclude '/mnt/*'
  --exclude '/media/*'
  --exclude '/lost+found'
)

echo "==> Syncing sysroot from ${REMOTE} to ${DEST}"
echo "    (port=${PORT}, delete=${DELETE_FLAG:-no}, dry-run=${DRY_RUN:+yes}${DRY_RUN:-no})"

rsync -avz ${DRY_RUN} ${DELETE_FLAG} --safe-links -e "${SSH}" \
  "${COMMON_EXCLUDES[@]}" \
  "${REMOTE}:/lib" "${DEST}/"

rsync -avz ${DRY_RUN} ${DELETE_FLAG} --safe-links -e "${SSH}" \
  "${COMMON_EXCLUDES[@]}" \
  "${REMOTE}:/usr" "${DEST}/"

rsync -avz ${DRY_RUN} ${DELETE_FLAG} --safe-links -e "${SSH}" \
  "${COMMON_EXCLUDES[@]}" \
  "${REMOTE}:/etc" "${DEST}/"

if [[ -n "${DRY_RUN}" ]]; then
  echo "==> Dry-run complete. No symlink fixes applied."
  exit 0
fi

echo "==> Fixing absolute symlinks inside ${DEST}"
# Convert absolute symlinks like /usr/lib/... to relative ones like ./usr/lib/...
cd "${DEST}"
while IFS= read -r -d '' link; do
  target="$(readlink "${link}")"
  if [[ "${target}" == /* ]]; then
    rm -f "${link}"
    ln -s ".${target}" "${link}"
  fi
done < <(find . -type l -lname '/*' -print0)

echo "==> Done."
