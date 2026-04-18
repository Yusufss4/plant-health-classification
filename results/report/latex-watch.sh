#!/usr/bin/env bash
# Continuous PDF: rebuilds when SWE577_578_Report.tex, references.bib, or inputs change.
set -euo pipefail
cd "$(dirname "$0")"
exec latexmk -pdf -pvc -interaction=nonstopmode SWE577_578_Report.tex
