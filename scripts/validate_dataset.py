"""Check the YOLO dataset before training.

    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --root dataset

Verifies the image/label pairing and the YOLO label format (single class 0,
normalized cx cy w h). Missing label files are reported as warnings (they act
as background images); malformed labels and orphan labels are errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def check_label(path: Path) -> list[str]:
    problems = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{path.name}:{lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(f"{path.name}:{lineno}: non-numeric value")
            continue
        if cls != 0:
            problems.append(f"{path.name}:{lineno}: class_id must be 0 (rupee_50), got {cls}")
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            problems.append(f"{path.name}:{lineno}: coordinates must be normalized 0..1")
        if w <= 0 or h <= 0:
            problems.append(f"{path.name}:{lineno}: width/height must be > 0")
        if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
            problems.append(f"{path.name}:{lineno}: box extends outside the image")
    return problems


def validate_split(root: Path, split: str) -> tuple[int, int, int, list[str], list[str]]:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS) if img_dir.is_dir() else []
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []

    errors: list[str] = []
    warnings: list[str] = []
    positives = negatives = 0

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    for image in images:
        label = lbl_dir / f"{image.stem}.txt"
        if not label.is_file():
            warnings.append(f"{split}/{image.name}: no label file (treated as background)")
            continue
        content = label.read_text(encoding="utf-8").strip()
        if content:
            positives += 1
        else:
            negatives += 1
        errors.extend(f"{split}/{e}" for e in check_label(label))

    for stem in sorted(label_stems - image_stems):
        errors.append(f"{split}: label without image: {stem}.txt")

    print(f"{split:5s}  images={len(images):4d}  labels={len(labels):4d}  "
          f"positives={positives:4d}  empty/background={negatives:4d}")
    return len(images), positives, negatives, errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    args = ap.parse_args()

    root = args.root
    if not (root / "data.yaml").is_file():
        print(f"[error] {root / 'data.yaml'} not found", file=sys.stderr)
        return 1

    print(f"Dataset: {root}\n")
    total_images = total_pos = total_neg = 0
    errors: list[str] = []
    warnings: list[str] = []
    for split in SPLITS:
        n_img, pos, neg, err, warn = validate_split(root, split)
        total_images += n_img
        total_pos += pos
        total_neg += neg
        errors += err
        warnings += warn

    print(f"\ntotal images: {total_images}  positives: {total_pos}  background: {total_neg}")
    if total_images and total_pos == 0:
        warnings.append("no positive ₹50 labels found; the model cannot learn the note")
    if total_images == 0:
        print("\n[warning] dataset is empty - add your own photos (see dataset/README.md)")
    for warning in warnings:
        print(f"[warn] {warning}")
    for error in errors:
        print(f"[error] {error}")

    if errors:
        print(f"\nFAILED: {len(errors)} error(s)")
        return 1
    print("\nOK: dataset structure and labels are valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
