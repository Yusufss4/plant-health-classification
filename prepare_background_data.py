"""
Background-class data preparation for the 3-class plant-health classifier.

Builds `data/{train,val,test}/background/` so the existing PlantHealthDataset can
load three classes: `healthy`, `diseased`, `background`.

Sources:
  - COCO 2017 train (downloaded directly from cocodataset.org, resumable via
    curl -C -). Filtered to exclude images that contain `potted plant` or
    `broccoli` instances so PlantVillage healthy/diseased leaves aren't
    confused with foliage-like COCO content.
  - A small synthetic fraction (~10% per split): solid colors, two-color
    gradients, Gaussian-blurred natural crops, and uniform noise patches.
    These cover trivial-frame deployment cases (lens cap, white wall, etc.).

Why direct download instead of TFDS:
  - TFDS `coco/2017` pulls train+val+test (~25 GB total) even though we only
    need train. Direct download lets us pull just train2017.zip (~18 GB).
  - `curl -C -` can resume across interruptions; TFDS restarts most partial
    downloads from zero.

Preprocessing per COCO image:
  - Short-side resize to 256, then a single random 224x224 crop. The model is
    trained at 224x224, so cropping (instead of full-frame resize) gives the
    network local patches rather than wide-context scenes.

Output layout (compatible with existing PlantHealthDataset):
    data/train/background/coco_*.jpg
    data/train/background/synth_*.jpg
    data/val/background/...
    data/test/background/...

Usage:
    python prepare_background_data.py [--target_train N] [--seed S]
"""

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

OUTPUT_DIR = Path("data")
SPLITS = ("train", "val", "test")

DEFAULT_TARGETS = {
    "train": 21000,
    "val": 4500,
    "test": 4500,
}

SPLIT_RATIOS = (0.70, 0.15, 0.15)

DROP_CATEGORY_NAMES = ("potted plant", "broccoli")

SYNTHETIC_FRACTION = 0.10

IMG_SIZE = 224
RESIZE_SHORT_SIDE = 256

# Direct download URLs (no TFDS). These are the same files TFDS hits internally.
COCO_BASE = "http://images.cocodataset.org"
COCO_TRAIN_IMAGES_URL = f"{COCO_BASE}/zips/train2017.zip"
COCO_TRAINVAL_ANN_URL = f"{COCO_BASE}/annotations/annotations_trainval2017.zip"

# Local cache for the COCO archives. Kept under data/ so it lives next to the
# .gitignore'd dataset content rather than polluting ~/tensorflow_datasets/.
CACHE_DIR = OUTPUT_DIR / ".coco_cache"
TRAIN_ZIP = CACHE_DIR / "train2017.zip"
ANN_ZIP = CACHE_DIR / "annotations_trainval2017.zip"

# Names of the partial download files TFDS may have written previously. We
# scavenge those bytes when present so we don't redo the work.
TFDS_DL_DIR = Path.home() / "tensorflow_datasets" / "downloads" / "coco"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return 0


def _seed_from_tfds_partial(dest: Path, pattern_substr: str) -> None:
    """
    If TFDS already pulled a partial archive matching `pattern_substr`, copy
    those bytes to `dest` (only if dest is smaller). curl -C - resumes from
    `dest`'s current size, so this can save tens of minutes of redownload.
    """
    if not TFDS_DL_DIR.is_dir():
        return
    candidates: list[Path] = []
    for entry in TFDS_DL_DIR.iterdir():
        # TFDS partials can be either .zip.tmp.<hash> files OR directories that
        # contain a single .zip inside (newer TFDS versions). Handle both.
        name = entry.name
        if pattern_substr not in name:
            continue
        if entry.is_file() and ".zip" in name and ".INFO" not in name:
            candidates.append(entry)
        elif entry.is_dir():
            inner = list(entry.glob("*.zip"))
            candidates.extend(inner)
    if not candidates:
        return
    best = max(candidates, key=lambda p: p.stat().st_size)
    if best.stat().st_size <= _file_size(dest):
        return
    print(
        f"  Found TFDS partial {best.name} ({best.stat().st_size / (1024**3):.2f} GiB);"
        f" copying to {dest.name} so curl can resume from there."
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)


def download_with_resume(url: str, dest: Path) -> None:
    """
    Download `url` to `dest` using curl with -C - (resume). Idempotent: if the
    file is already complete, curl returns quickly.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nFetching {url}\n  -> {dest}")
    # --fail makes HTTP errors propagate as a non-zero exit; --location follows
    # redirects; --retry adds light tolerance for connection blips.
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--output",
        str(dest),
        url,
    ]
    # Stream curl's progress directly so the user sees a live percentage bar.
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(
            f"curl failed for {url} (exit {res.returncode}). "
            "Re-run the script to resume."
        )


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def load_bad_image_ids(ann_zip_path: Path) -> set[int]:
    """
    Parse instances_train2017.json out of the annotations zip and return the
    set of image_ids that have at least one annotation in any
    DROP_CATEGORY_NAMES category.
    """
    print(f"\nParsing annotations from {ann_zip_path.name} ...")
    with zipfile.ZipFile(ann_zip_path) as zf:
        member = "annotations/instances_train2017.json"
        if member not in zf.namelist():
            raise RuntimeError(f"{member} not found in {ann_zip_path}")
        with zf.open(member) as fh:
            data = json.load(fh)

    cat_name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    drop_cat_ids: set[int] = set()
    for n in DROP_CATEGORY_NAMES:
        if n in cat_name_to_id:
            drop_cat_ids.add(cat_name_to_id[n])
        else:
            print(f"  Warning: COCO category '{n}' not in categories list.")
    print(f"  Dropping COCO categories: {DROP_CATEGORY_NAMES} -> ids {sorted(drop_cat_ids)}")

    bad_ids: set[int] = set()
    for ann in data["annotations"]:
        if ann["category_id"] in drop_cat_ids:
            bad_ids.add(ann["image_id"])
    print(f"  Total train images with disallowed categories: {len(bad_ids):,}")
    print(f"  Total train annotations:                       {len(data['annotations']):,}")
    print(f"  Total train images:                            {len(data['images']):,}")
    return bad_ids


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    if w <= h:
        new_w = short_side
        new_h = max(short_side, int(round(h * short_side / max(w, 1))))
    else:
        new_h = short_side
        new_w = max(short_side, int(round(w * short_side / max(h, 1))))
    return img.resize((new_w, new_h), Image.BILINEAR)


def random_crop(img: Image.Image, size: int, rng: random.Random) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        img = resize_short_side(img, size)
        w, h = img.size
    left = rng.randint(0, w - size)
    top = rng.randint(0, h - size)
    return img.crop((left, top, left + size, top + size))


# ---------------------------------------------------------------------------
# Synthetic backgrounds
# ---------------------------------------------------------------------------

def make_solid(rng: random.Random) -> Image.Image:
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    return Image.new("RGB", (IMG_SIZE, IMG_SIZE), color)


def make_gradient(rng: random.Random) -> Image.Image:
    c0 = np.array(
        [rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)],
        dtype=np.float32,
    )
    c1 = np.array(
        [rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)],
        dtype=np.float32,
    )
    axis = rng.choice(("h", "v", "d"))
    t = np.linspace(0.0, 1.0, IMG_SIZE, dtype=np.float32)
    if axis == "h":
        ramp = t[None, :]
    elif axis == "v":
        ramp = t[:, None]
    else:
        ramp = (t[None, :] + t[:, None]) / 2.0
    ramp = ramp[..., None]
    arr = c0[None, None, :] * (1.0 - ramp) + c1[None, None, :] * ramp
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def make_noise(rng: random.Random) -> Image.Image:
    arr = np.random.default_rng(rng.randint(0, 2**32 - 1)).integers(
        0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
    )
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=1.5))


def make_blurred(seed_img: Image.Image, rng: random.Random) -> Image.Image:
    radius = rng.uniform(6.0, 18.0)
    return seed_img.filter(ImageFilter.GaussianBlur(radius=radius))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def pick_split(rng: random.Random) -> str:
    r = rng.random()
    if r < SPLIT_RATIOS[0]:
        return "train"
    if r < SPLIT_RATIOS[0] + SPLIT_RATIOS[1]:
        return "val"
    return "test"


def ensure_dirs() -> None:
    for split in SPLITS:
        (OUTPUT_DIR / split / "background").mkdir(parents=True, exist_ok=True)


def coco_targets(targets: dict) -> dict:
    return {
        s: max(0, int(round(targets[s] * (1.0 - SYNTHETIC_FRACTION))))
        for s in SPLITS
    }


def synth_targets(targets: dict) -> dict:
    return {
        s: max(0, int(round(targets[s] * SYNTHETIC_FRACTION)))
        for s in SPLITS
    }


def process_coco_images(
    train_zip_path: Path,
    bad_ids: set[int],
    targets: dict,
    rng: random.Random,
) -> dict:
    coco_quota = coco_targets(targets)
    print(f"\nCOCO quotas per split: {coco_quota}")
    counters = {s: 0 for s in SPLITS}
    blur_pool: list[Image.Image] = []
    scanned = 0
    dropped_bad = 0
    failed = 0

    with zipfile.ZipFile(train_zip_path) as zf:
        members = [
            n for n in zf.namelist()
            if n.startswith("train2017/") and n.lower().endswith(".jpg")
        ]
        print(f"  train2017.zip contains {len(members):,} jpg entries.")
        rng.shuffle(members)

        for member in members:
            scanned += 1
            if scanned % 5000 == 0:
                print(
                    f"  scanned {scanned:,} | "
                    + " ".join(f"{s}={counters[s]}" for s in SPLITS)
                )

            stem = Path(member).stem  # e.g. 000000123456
            try:
                image_id = int(stem)
            except ValueError:
                continue
            if image_id in bad_ids:
                dropped_bad += 1
                continue

            split = pick_split(rng)
            if counters[split] >= coco_quota[split]:
                remaining = [s for s in SPLITS if counters[s] < coco_quota[s]]
                if not remaining:
                    break
                split = rng.choice(remaining)

            try:
                with zf.open(member) as fh:
                    raw = fh.read()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:  # noqa: BLE001 - any decode error is "skip and move on"
                failed += 1
                if failed <= 5:
                    print(f"  Skip (decode error): {member} ({e})")
                continue

            img = resize_short_side(img, RESIZE_SHORT_SIDE)
            img = random_crop(img, IMG_SIZE, rng)

            if len(blur_pool) < 1024:
                blur_pool.append(img.copy())

            idx = counters[split]
            out_path = OUTPUT_DIR / split / "background" / f"coco_{idx:06d}.jpg"
            img.save(out_path, format="JPEG", quality=92)
            counters[split] += 1

            if all(counters[s] >= coco_quota[s] for s in SPLITS):
                break

    print(
        f"\nCOCO pass done. Scanned={scanned:,}  DroppedBad={dropped_bad:,}  "
        f"DecodeErrors={failed:,}  "
        + " ".join(f"{s}={counters[s]}" for s in SPLITS)
    )
    return {"counters": counters, "blur_pool": blur_pool}


def prepare_synthetic_backgrounds(
    targets: dict,
    blur_pool: list,
    rng: random.Random,
) -> dict:
    synth_quota = synth_targets(targets)
    print(f"\nSynthetic quotas per split: {synth_quota}")

    counters = {s: 0 for s in SPLITS}
    base_kinds = ["solid", "gradient", "noise"]
    if blur_pool:
        base_kinds.append("blurred")

    for split in SPLITS:
        n = synth_quota[split]
        for i in range(n):
            kind = rng.choice(base_kinds)
            if kind == "solid":
                img = make_solid(rng)
            elif kind == "gradient":
                img = make_gradient(rng)
            elif kind == "noise":
                img = make_noise(rng)
            else:
                img = make_blurred(rng.choice(blur_pool), rng)

            out_path = OUTPUT_DIR / split / "background" / f"synth_{i:05d}.jpg"
            img.save(out_path, format="JPEG", quality=92)
            counters[split] += 1
        print(f"  {split}: wrote {counters[split]} synthetic images")
    return counters


def verify_counts(targets: dict) -> None:
    print("\n" + "=" * 80)
    print("Final dataset counts:")
    print("=" * 80)
    for split in SPLITS:
        for c in ("healthy", "diseased", "background"):
            d = OUTPUT_DIR / split / c
            if d.is_dir():
                n = len([p for p in d.glob("*.jpg")])
                tag = f" (background target {targets[split]:,})" if c == "background" else ""
                print(f"  data/{split}/{c:<11s}: {n:,}{tag}")
            else:
                print(f"  data/{split}/{c:<11s}: <missing>")


def cleanup_cache(remove: bool) -> None:
    if not remove:
        print(
            f"\nKept downloaded archives in {CACHE_DIR} ({_file_size(TRAIN_ZIP)/(1024**3):.2f} GiB train zip)."
            "  Pass --cleanup to delete them."
        )
        return
    if CACHE_DIR.is_dir():
        print(f"\nRemoving cache dir {CACHE_DIR} ...")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare COCO + synthetic background images for the 3-class model."
    )
    parser.add_argument("--target_train", type=int, default=DEFAULT_TARGETS["train"])
    parser.add_argument("--target_val", type=int, default=DEFAULT_TARGETS["val"])
    parser.add_argument("--target_test", type=int, default=DEFAULT_TARGETS["test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove data/.coco_cache after preparation (saves ~18 GiB).",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Assume the COCO archives are already present in data/.coco_cache.",
    )
    args = parser.parse_args()

    targets = {
        "train": args.target_train,
        "val": args.target_val,
        "test": args.target_test,
    }
    print("Target background counts:", targets)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        # Reuse anything TFDS may have pulled in a previous run.
        _seed_from_tfds_partial(TRAIN_ZIP, "train2017")
        _seed_from_tfds_partial(ANN_ZIP, "annotat")
        download_with_resume(COCO_TRAINVAL_ANN_URL, ANN_ZIP)
        download_with_resume(COCO_TRAIN_IMAGES_URL, TRAIN_ZIP)
    else:
        if not ANN_ZIP.is_file() or not TRAIN_ZIP.is_file():
            print(
                f"ERROR: --skip_download set but expected files missing:"
                f"\n  {ANN_ZIP}\n  {TRAIN_ZIP}",
                file=sys.stderr,
            )
            sys.exit(2)

    bad_ids = load_bad_image_ids(ANN_ZIP)
    coco_result = process_coco_images(TRAIN_ZIP, bad_ids, targets, rng)
    prepare_synthetic_backgrounds(targets, coco_result["blur_pool"], rng)
    verify_counts(targets)
    cleanup_cache(args.cleanup)

    print("\nDone. Next: python train.py --model mobilenet_v3")


if __name__ == "__main__":
    main()
