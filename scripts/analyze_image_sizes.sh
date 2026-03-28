#!/usr/bin/env bash
# Analyze image dimensions under a data tree (e.g. data/train/healthy, data/test/diseased).
#
# Usage:
#   bash scripts/analyze_image_sizes.sh [ROOT_DIR]
#
# Default ROOT_DIR is "data" (relative to current working directory).
# Requires: python3, Pillow (pip install Pillow)
#
# Examples:
#   cd /path/to/plant-health-classification && bash scripts/analyze_image_sizes.sh
#   bash scripts/analyze_image_sizes.sh /path/to/custom/images

set -euo pipefail

ROOT="${1:-data}"

if [[ ! -d "$ROOT" ]]; then
  echo "Directory not found: $ROOT" >&2
  exit 1
fi

# Resolve to absolute path for clearer reporting
ROOT_ABS="$(cd "$ROOT" && pwd)"

export ANALYZE_ROOT="$ROOT_ABS"

python3 << 'PY'
import os
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow is required: pip install Pillow", file=sys.stderr)
    sys.exit(1)

root = Path(os.environ["ANALYZE_ROOT"])
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

def iter_images(base: Path):
    if not base.is_dir():
        return
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def size_of(path: Path):
    with Image.open(path) as im:
        return im.size  # (width, height)

rows = []
errors = []
for path in sorted(iter_images(root)):
    try:
        w, h = size_of(path)
        rel = path.relative_to(root)
        rows.append((str(rel), w, h))
    except Exception as e:
        errors.append((str(path.relative_to(root)), str(e)))

if not rows and not errors:
    print(f"No images found under {root} (extensions: {sorted(exts)})")
    sys.exit(0)

widths = [r[1] for r in rows]
heights = [r[2] for r in rows]
pairs = [(r[1], r[2]) for r in rows]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0

print("=" * 72)
print("IMAGE SIZE ANALYSIS")
print("=" * 72)
print(f"Root: {root}")
print(f"Images read successfully: {len(rows)}")
if errors:
    print(f"Failed to read (corrupt/unsupported): {len(errors)}")
print()

print("Width (px)")
print(f"  min:    {min(widths)}")
print(f"  max:    {max(widths)}")
print(f"  mean:   {mean(widths):.2f}")
print(f"  median: {median(widths):.2f}")
print()
print("Height (px)")
print(f"  min:    {min(heights)}")
print(f"  max:    {max(heights)}")
print(f"  mean:   {mean(heights):.2f}")
print(f"  median: {median(heights):.2f}")
print()

area = [w * h for w, h in zip(widths, heights)]
print("Area (pixels)")
print(f"  min:    {min(area):,}")
print(f"  max:    {max(area):,}")
print(f"  mean:   {mean(area):,.2f}")
print()

ctr = Counter(pairs)
print(f"Unique (width × height) combinations: {len(ctr)}")
print()
print("Most common dimensions (count):")
for (w, h), c in ctr.most_common(25):
    pct = 100.0 * c / len(rows)
    print(f"  {w:5d} × {h:5d}  {c:6d}  ({pct:5.1f}%)")

# Optional: breakdown by first path segment (e.g. train / val / test)
by_split = Counter()
for rel, w, h in rows:
    parts = Path(rel).parts
    key = parts[0] if len(parts) > 1 else "(top-level)"
    by_split[key] += 1

if len(by_split) > 1:
    print()
    print("Images per top-level folder under root:")
    for k in sorted(by_split.keys()):
        print(f"  {k}: {by_split[k]}")

if errors:
    print()
    print("-" * 72)
    print("Read errors (first 20):")
    for rel, err in errors[:20]:
        print(f"  {rel}: {err}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")

print("=" * 72)
PY
