#!/usr/bin/env bash
# Continuous PDF: rebuilds when SWE599-Project-Progress-2026S-Savas-Yusuf.tex, references.bib, or inputs change.
set -euo pipefail
cd "$(dirname "$0")"
exec latexmk -pdf -pvc -interaction=nonstopmode SWE599-Project-Progress-2026S-Savas-Yusuf.tex
