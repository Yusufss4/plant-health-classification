#!/usr/bin/env bash
# Render custom HTML diagrams (mermaid/*.html) to PDFs for LaTeX.
#
# Unlike render-mermaid.sh (mmdc on .mmd files), these are hand-built HTML/CSS
# layouts. Uses headless Chromium print-to-PDF.
#
# Usage:
#   ./render-html-diagrams.sh              # -> figures/*_v2.pdf + *_v2_crop.pdf
#   SUFFIX=_v3 ./render-html-diagrams.sh
#   ./render-html-diagrams.sh --full-only  # letter-sized, no crop variants
#   ./render-html-diagrams.sh --crop-only
#
# Requires: chromium or google-chrome (snap Chromium cannot read /tmp; wrappers
# are written under mermaid/.render/ inside the repo).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SUFFIX="${SUFFIX:-_v2}"
HTML_DIR="mermaid"
RENDER_DIR="${HTML_DIR}/.render"
OUT_DIR="figures"
FULL_ONLY=0
CROP_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --full-only) FULL_ONLY=1 ;;
    --crop-only) CROP_ONLY=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 1
      ;;
  esac
done

if [[ "$FULL_ONLY" -eq 1 && "$CROP_ONLY" -eq 1 ]]; then
  echo "Use at most one of --full-only and --crop-only." >&2
  exit 1
fi

CHROME=""
for c in chromium chromium-browser google-chrome google-chrome-stable; do
  if command -v "$c" >/dev/null 2>&1; then
    CHROME="$c"
    break
  fi
done
if [[ -z "$CHROME" ]]; then
  echo "Need chromium or google-chrome on PATH." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$RENDER_DIR"

# Crop page sizes matched to existing figures/*_crop.pdf (poster layout).
declare -A CROP_PAGE_SIZE=(
  [training_pipeline]="3.31in 9.57in"
  [system_architecture]="6.07in 5.82in"
  [ondevice_runtime]="8.27in 5.2in"
)

chrome_pdf() {
  local html_path="$1"
  local pdf_path="$2"
  "$CHROME" \
    --headless=new \
    --disable-gpu \
    --no-pdf-header-footer \
    --print-to-pdf="$pdf_path" \
    "file://${html_path}" \
    >/dev/null 2>&1 || true
  if [[ ! -s "$pdf_path" ]]; then
    echo "Failed to write $pdf_path (check Chromium / file:// access)." >&2
    exit 1
  fi
}

inject_crop_page() {
  local src_html="$1"
  local page_size="$2"
  local out_html="$3"
  python3 - "$src_html" "$page_size" "$out_html" <<'PY'
import sys
from pathlib import Path

src_path, page_size, out_path = sys.argv[1:4]
text = Path(src_path).read_text(encoding="utf-8")
inject = f"  @page {{ size: {page_size}; margin: 0; }}\n\n"
if "@page" in text.split("</style>", 1)[0]:
    # Replace existing @page (e.g. ondevice_runtime.html).
    import re
    text = re.sub(
        r"@page\s*\{[^}]*\}\s*",
        f"@page {{ size: {page_size}; margin: 0; }}\n\n",
        text,
        count=1,
    )
else:
    text = text.replace("<style>\n", "<style>\n" + inject, 1)
Path(out_path).write_text(text, encoding="utf-8")
PY
}

render_full() {
  local name="$1"
  local src="${ROOT}/${HTML_DIR}/${name}.html"
  local pdf="${ROOT}/${OUT_DIR}/${name}${SUFFIX}.pdf"
  if [[ ! -f "$src" ]]; then
    echo "Missing $src" >&2
    exit 1
  fi
  chrome_pdf "$(realpath "$src")" "$pdf"
  echo "Wrote ${pdf#${ROOT}/}"
}

render_crop() {
  local name="$1"
  local src="${ROOT}/${HTML_DIR}/${name}.html"
  local page_size="${CROP_PAGE_SIZE[$name]:-}"
  if [[ -z "$page_size" ]]; then
    echo "No crop page size for $name" >&2
    exit 1
  fi
  local wrapper="${ROOT}/${RENDER_DIR}/${name}${SUFFIX}_crop.html"
  local pdf="${ROOT}/${OUT_DIR}/${name}${SUFFIX}_crop.pdf"
  inject_crop_page "$src" "$page_size" "$wrapper"
  chrome_pdf "$(realpath "$wrapper")" "$pdf"
  echo "Wrote ${pdf#${ROOT}/}"
}

DIAGRAMS=(system_architecture training_pipeline ondevice_runtime)

for name in "${DIAGRAMS[@]}"; do
  if [[ "$CROP_ONLY" -eq 0 ]]; then
    render_full "$name"
  fi
  if [[ "$FULL_ONLY" -eq 0 ]]; then
    render_crop "$name"
  fi
done

echo "Done. Suffix=${SUFFIX}  (set SUFFIX=... to change output names)"
