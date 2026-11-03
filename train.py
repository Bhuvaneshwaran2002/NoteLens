"""Train the new-₹50 YOLOv8 detector and export an OpenCV-ready ONNX model.

    pip install ultralytics
    python train.py --data dataset/data.yaml --epochs 100 --imgsz 512

The dataset must already contain your own photos and YOLO labels; see
dataset/README.md. Training writes runs/<name>/weights/best.pt and exports
best.onnx to ./model.onnx (and models/model.onnx) for `python test.py`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train YOLOv8 on the Rs.50 dataset and export ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset/data.yaml", help="dataset yaml (default: dataset/data.yaml)")
    p.add_argument("--model", default="yolov8n.pt", help="starting weights (default: yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="early-stopping patience")
    p.add_argument("--imgsz", type=int, default=512, help="square training/export size")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="cpu, 0, 0,1, ... (default: auto)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs", help="Ultralytics output folder")
    p.add_argument("--name", default="rs50", help="run name inside --project")
    p.add_argument("--dest", default="model.onnx", help="exported ONNX path")
    p.add_argument("--no-export", action="store_true", help="train only, skip ONNX export")
    return p.parse_args(argv)


def dataset_is_ready(data_yaml: Path) -> tuple[bool, str]:
    dataset_root = data_yaml.parent
    train_dir = dataset_root / "images" / "train"
    images = [p for p in train_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS] if train_dir.is_dir() else []
    if not images:
        return False, (f"No training images found in {train_dir}. Add your own photos and YOLO "
                       f"labels first (see dataset/README.md).")
    labels_dir = dataset_root / "labels" / "train"
    labelled = sum(1 for p in images if (labels_dir / f"{p.stem}.txt").is_file())
    if labelled == 0:
        return False, (f"None of the {len(images)} training images has a label file in {labels_dir}. "
                       f"Label the ₹50 notes and create empty .txt files for background photos.")
    return True, f"{len(images)} training image(s), {labelled} with labels"


def main(argv=None) -> int:
    args = parse_args(argv)

    data_yaml = Path(args.data)
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    if not data_yaml.is_file():
        print(f"[error] dataset yaml not found: {data_yaml}", file=sys.stderr)
        return 1

    ready, detail = dataset_is_ready(data_yaml)
    if not ready:
        print(f"[error] {detail}", file=sys.stderr)
        return 1
    print(f"[info] {detail}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[error] ultralytics is required for training:\n  pip install ultralytics",
              file=sys.stderr)
        return 1

    train_kwargs = dict(
        data=str(data_yaml), epochs=args.epochs, patience=args.patience,
        imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        project=str(ROOT / args.project), name=args.name, exist_ok=True,
        seed=42, deterministic=True, plots=True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    print(f"[info] training {args.model} for {args.epochs} epochs at {args.imgsz}px")
    results = YOLO(args.model).train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", ROOT / args.project / args.name))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"[error] training finished but {best} was not found", file=sys.stderr)
        return 1
    print(f"[info] best weights: {best}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.copy2(best, models_dir / "best.pt")

    if args.no_export:
        print("[info] --no-export given, skipping ONNX export")
        return 0

    try:
        exported = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    except Exception as exc:  # export can raise several backend-specific errors
        print(f"[error] ONNX export failed: {exc}\n"
              f"Install the export extras with: pip install ultralytics onnx onnxslim", file=sys.stderr)
        return 1

    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, dest)
    shutil.copy2(exported, models_dir / "model.onnx")
    print(f"[info] exported ONNX: {dest}")
    print("\nRun it with:\n  python test.py --model model.onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
