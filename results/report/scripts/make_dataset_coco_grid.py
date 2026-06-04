#!/usr/bin/env python3
"""
Build a 5x5 montage of *project background-class* samples for the report figure.

By default uses images already written by prepare_background_data.py:
  - coco_*.jpg  (~90%): filtered COCO (no potted plant / broccoli), short-side
    resize to 256, random 224x224 crop — local scene patches, not full frames.
  - synth_*.jpg (~10%): solid / gradient / blur / noise patches.

Layout (white gaps, 688x673) matches dataset-plant-village.png.

Pass --source raw to fall back to evenly spaced full COCO train2017 frames.
"""

from __future__ import annotations

import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image

CELL_BOXES = [
    (11, 10, 124, 123),
    (149, 10, 263, 123),
    (288, 10, 401, 123),
    (426, 10, 540, 123),
    (565, 10, 678, 123),
    (11, 145, 124, 258),
    (149, 145, 263, 258),
    (288, 145, 401, 258),
    (426, 145, 540, 258),
    (565, 145, 678, 258),
    (11, 280, 124, 393),
    (149, 280, 263, 393),
    (288, 280, 401, 393),
    (426, 280, 540, 393),
    (565, 280, 678, 393),
    (11, 415, 124, 528),
    (149, 415, 263, 528),
    (288, 415, 401, 528),
    (426, 415, 540, 528),
    (565, 415, 678, 528),
    (11, 550, 124, 663),
    (149, 550, 263, 663),
    (288, 550, 401, 663),
    (426, 550, 540, 663),
    (565, 550, 678, 663),
]

CANVAS_SIZE = (688, 673)
CELL_SIZE = 113  # max content width/height in the reference layout
NUM_CELLS = len(CELL_BOXES)

DEFAULT_BACKGROUND_DIR = Path("data/train/background")
DEFAULT_ZIP = Path("data/.coco_cache/train2017.zip")
DEFAULT_OUT = Path("results/report/figures/dataset-coco.png")
SYNTHETIC_FRACTION = 0.10  # match prepare_background_data.py


def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def evenly_spaced(files: list[Path], count: int, rng: random.Random) -> list[Path]:
    if len(files) < count:
        raise ValueError(f"Need at least {count} files, found {len(files)}")
    files = sorted(files)
    if count == 1:
        return [files[0]]
    indices = [int(round(i * (len(files) - 1) / (count - 1))) for i in range(count)]
    # Deterministic shuffle of indices so synth/coco are not always the same IDs.
    order = list(range(count))
    rng.shuffle(order)
    seen: set[int] = set()
    picked: list[Path] = []
    for slot in order:
        idx = indices[slot]
        while idx in seen and idx < len(files) - 1:
            idx += 1
        seen.add(idx)
        picked.append(files[idx])
    return picked


def list_prepared_background(background_dir: Path) -> tuple[list[Path], list[Path]]:
    coco = sorted(background_dir.glob("coco_*.jpg"))
    synth = [
        p
        for p in sorted(background_dir.glob("synth_*.jpg"))
        if Image.open(p).size == (224, 224)
    ]
    if not coco:
        raise FileNotFoundError(f"No coco_*.jpg under {background_dir}")
    return coco, synth


def pick_prepared_samples(
    background_dir: Path,
    count: int,
    seed: int,
    synth_fraction: float,
) -> list[Path]:
    coco_files, synth_files = list_prepared_background(background_dir)
    rng = random.Random(seed)

    n_synth = min(len(synth_files), max(0, round(count * synth_fraction)))
    if synth_files and n_synth == 0:
        n_synth = 1
    n_coco = count - n_synth

    coco_pick = evenly_spaced(coco_files, n_coco, rng)
    synth_pick = (
        evenly_spaced(synth_files, n_synth, random.Random(seed + 1)) if n_synth else []
    )

    # Scatter synth cells (corners + center) so they read as a distinct minority.
    if n_synth >= 3:
        synth_slots = [0, 12, 24]
    elif n_synth == 2:
        synth_slots = [0, 12]
    else:
        synth_slots = [12]

    ordered: list[Path | None] = [None] * count
    for slot, path in zip(synth_slots, synth_pick, strict=True):
        ordered[slot] = path
    coco_slots = [i for i in range(count) if i not in synth_slots]
    for slot, path in zip(coco_slots, coco_pick, strict=True):
        ordered[slot] = path
    if any(p is None for p in ordered):
        raise RuntimeError("Failed to fill montage slots")
    return ordered  # type: ignore[return-value]


def resize_cell(img: Image.Image, size: int) -> Image.Image:
    if img.size != (size, size):
        return img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def build_from_prepared(background_dir: Path, out_path: Path, seed: int) -> None:
    samples = pick_prepared_samples(background_dir, NUM_CELLS, seed, SYNTHETIC_FRACTION)
    canvas = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
    for path, (x0, y0, x1, y1) in zip(samples, CELL_BOXES, strict=True):
        img = Image.open(path).convert("RGB")
        cell = max(x1 - x0, y1 - y0)
        canvas.paste(resize_cell(img, cell), (x0, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    n_synth = sum(1 for p in samples if p.name.startswith("synth_"))
    print(
        f"Wrote {out_path} ({canvas.size[0]}x{canvas.size[1]}) "
        f"from {background_dir}: {len(samples) - n_synth} coco + {n_synth} synth"
    )


def build_from_raw_zip(zip_path: Path, out_path: Path, seed: int) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n
            for n in zf.namelist()
            if n.startswith("train2017/") and n.lower().endswith(".jpg")
        ]
        chosen = [
            str(p)
            for p in evenly_spaced(
                [Path(m) for m in members], NUM_CELLS, random.Random(seed)
            )
        ]
        canvas = Image.new("RGB", CANVAS_SIZE, (255, 255, 255))
        for member, (x0, y0, x1, y1) in zip(chosen, CELL_BOXES, strict=True):
            with zf.open(member) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
            cell = max(x1 - x0, y1 - y0)
            thumb = center_square_crop(img).resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x0, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"Wrote {out_path} ({canvas.size[0]}x{canvas.size[1]}) from raw {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("prepared", "raw"),
        default="prepared",
        help="prepared = data/.../background crops; raw = full COCO frames",
    )
    parser.add_argument("--background-dir", type=Path, default=DEFAULT_BACKGROUND_DIR)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source == "prepared":
        if not args.background_dir.is_dir():
            raise SystemExit(
                f"Background dir not found: {args.background_dir}\n"
                "Run: python prepare_background_data.py"
            )
        build_from_prepared(args.background_dir, args.out, args.seed)
    else:
        if not args.zip.is_file():
            raise SystemExit(f"COCO zip not found: {args.zip}")
        build_from_raw_zip(args.zip, args.out, args.seed)


if __name__ == "__main__":
    main()
