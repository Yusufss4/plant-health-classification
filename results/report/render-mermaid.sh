#!/usr/bin/env bash
# Render Mermaid sources under mermaid/ into figures/ for pdflatex.
# Prefers: mmdc (npm i -g @mermaid-js/mermaid-cli), then npx, then Docker
# (minlag/mermaid-cli; image entrypoint is mmdc, pass only -i/-o/flags).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p figures mermaid

render_one() {
  local in_mmd="$1"
  local out_pdf="$2"
  if command -v mmdc >/dev/null 2>&1; then
    mmdc -i "$in_mmd" -o "$out_pdf" -b white
  elif command -v npx >/dev/null 2>&1; then
    npx --yes @mermaid-js/mermaid-cli -i "$in_mmd" -o "$out_pdf" -b white
  elif command -v docker >/dev/null 2>&1; then
    # Writable bind mount (some Docker setups deny root writes without 777 on the dir).
    chmod a+w figures 2>/dev/null || true
    docker run --rm -v "${ROOT}:/data" minlag/mermaid-cli:latest \
      -i "/data/${in_mmd}" -o "/data/${out_pdf}" -b white
  else
    echo "Need mmdc, npx, or docker." >&2
    exit 1
  fi
}

render_one "mermaid/system_architecture.mmd" "figures/system_architecture.pdf"
echo "Wrote figures/system_architecture.pdf"
