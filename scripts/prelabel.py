"""Bootstrap YOLO labels for new photos with the existing model.

    python scripts/prelabel.py --source photos/
    python scripts/prelabel.py --source photos/ --val-fraction 0.2 --conf 0.5

Images are copied into dataset/images/{train,val}/ and one YOLO label file is
written per image into dataset/labels/{train,val}/ (empty = background).
A stable hash decides the train/val split, so adding more photos does not
reshuffle earlier ones.

These are only starting labels: open them in a YOLO annotator and fix missed
₹50 notes (an image with a missed note gets an empty label, which teaches the
model that the note is background). Review every file before training.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import DEFAULT_MODEL_PATH, DetectorError, RupeeDetector  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pre-label a folder of photos with the existing ONNX model.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="folder of raw photos")
    p.add_argument("--model", default=None, help="ONNX model (default: model.onnx)")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (default 0.5)")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--split", choices=("train", "val", "test"),
                   help="force every image into one split (default: hash-based train/val)")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="fraction sent to val when --split is not given (default 0.2)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing copied images and label files")
    return p.parse_args(argv)


def resolve_model(path_arg: str | None) -> Path:
    if path_arg:
        candidates = [Path(path_arg), ROOT / path_arg, ROOT / "models" / Path(path_arg).name]
    else:
        candidates = [ROOT / "model.onnx", DEFAULT_MODEL_PATH]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise DetectorError(f"ONNX model not found; tried: {', '.join(map(str, candidates))}")


def split_for(stem: str, val_fraction: float) -> str:
    if val_fraction <= 0:
        return "train"
    bucket = hashlib.sha1(stem.encode("utf-8")).digest()[0] / 255.0
    return "val" if bucket < val_fraction else "train"


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.source.is_dir():
        print(f"[error] source folder not found: {args.source}", file=sys.stderr)
        return 1

    images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[error] no images found in {args.source}", file=sys.stderr)
        return 1

    try:
        detector = RupeeDetector(resolve_model(args.model), conf_threshold=args.conf,
                                 iou_threshold=args.iou)
    except DetectorError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    copied = positives = backgrounds = skipped = unreadable = 0
    for path in images:
        split = args.split or split_for(path.stem, args.val_fraction)
        img_dest = ROOT / "dataset" / "images" / split / path.name
        lbl_dest = ROOT / "dataset" / "labels" / split / f"{path.stem}.txt"
        if not args.overwrite and (img_dest.exists() or lbl_dest.exists()):
            skipped += 1
            continue

        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[skip] unreadable: {path.name}")
            unreadable += 1
            continue

        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        lines = []
        for d in detections:
            cx, cy = (d.x1 + d.x2) / 2 / w, (d.y1 + d.y2) / 2 / h
            bw, bh = (d.x2 - d.x1) / w, (d.y2 - d.y1) / h
            if bw > 0 and bh > 0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        img_dest.parent.mkdir(parents=True, exist_ok=True)
        lbl_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, img_dest)
        lbl_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        copied += 1
        if lines:
            positives += 1
        else:
            backgrounds += 1
            print(f"[review] no detection: {path.name} -> empty label (background)")

    print(f"\nimages used: {copied}  positives: {positives}  background: {backgrounds}  "
          f"skipped: {skipped}  unreadable: {unreadable}")
    print("Review the labels in an annotator before training "
          "(see dataset/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
