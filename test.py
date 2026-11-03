"""Live ₹50 note detector (ZED Digital FEDS - Problem Statement 3).

Opens the webcam with OpenCV, runs the exported YOLOv8 ONNX model, and draws a
box labelled "₹50 <confidence>" when the new light blue-green ₹50 note is in
front of the camera.

    python test.py --model model.onnx
    python test.py                       # uses ./model.onnx or ./models/model.onnx
    python test.py --camera 1 --conf 0.5
    python test.py --source photo.jpg --save
    python test.py --source samples/ --save
    python test.py --source clip.mp4 --save

Keys while the webcam window is focused:
    q / Esc  quit
    s        save the annotated frame to outputs/
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import DEFAULT_MODEL_PATH, PROJECT_ROOT, DetectorError, RupeeDetector


def _utf8_console() -> None:
    """Avoid UnicodeEncodeError on Windows consoles that cannot print ₹."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_utf8_console()

OUTPUT_DIR = PROJECT_ROOT / "outputs"
WINDOW_NAME = "Rs.50 Note Detector  |  q = quit, s = save"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Detect the new Rs.50 note from a webcam (or an image/video with --source).")
    p.add_argument("--model", default=None,
                   help="ONNX model path (default: ./model.onnx, then ./models/model.onnx)")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--conf", type=float, default=0.6, help="confidence threshold (default 0.6)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    p.add_argument("--source", type=Path,
                   help="optional image, folder of images, or video instead of the webcam")
    p.add_argument("--save", action="store_true",
                   help="save annotated output under outputs/ (with --source)")
    p.add_argument("--show", action="store_true",
                   help="show a window for --source input (off by default)")
    p.add_argument("--width", type=int, default=1280, help="requested webcam width")
    p.add_argument("--height", type=int, default=720, help="requested webcam height")
    p.add_argument("--cuda", action="store_true", help="use the OpenCV CUDA backend if available")
    return p.parse_args(argv)


def resolve_model(path_arg: str | None) -> Path:
    """Find the model for the command in the assignment (`--model model.onnx`)."""
    if path_arg:
        candidates = [Path(path_arg)]
        if not Path(path_arg).is_absolute():
            candidates += [
                PROJECT_ROOT / path_arg,
                PROJECT_ROOT / "models" / Path(path_arg).name,
            ]
        if Path(path_arg).suffix.lower() != ".onnx":
            candidates.append(PROJECT_ROOT / "models" / (Path(path_arg).stem + ".onnx"))
    else:
        candidates = [PROJECT_ROOT / "model.onnx", DEFAULT_MODEL_PATH]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise DetectorError(
        f"ONNX model not found. Tried:\n  {tried}\n"
        f"Put model.onnx next to test.py or pass --model path/to/model.onnx")


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # DirectShow opens much faster than MSMF on most Windows webcams.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Check that it is connected, not used by another "
            f"app, and that camera access is allowed in Windows Privacy settings. "
            f"Try --camera 1 if you have more than one camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def annotate(detector: RupeeDetector, frame, detections, fps: float | None = None):
    out = detector.draw(frame, detections)
    text = f"notes: {len(detections)}"
    if fps is not None:
        text = f"FPS {fps:.1f} | {text}"
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_frame(frame, prefix: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def show(frame) -> int:
    cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF


def run_source(detector: RupeeDetector, source: Path, save: bool, show_window: bool) -> int:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"[error] Could not read image: {source}", file=sys.stderr)
            return 1
        detections = detector.detect(frame)
        annotated = annotate(detector, frame, detections)
        summary = ", ".join(f"{d.confidence:.2f}" for d in detections) or "none"
        print(f"{source.name}: notes={len(detections)} conf=[{summary}]")
        if save:
            print(f"[info] saved {save_frame(annotated, source.stem)}")
        if show_window:
            cv2.imshow(WINDOW_NAME, annotated)
            cv2.waitKey(0)
        return 0

    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"[error] No images found in {source}", file=sys.stderr)
            return 1
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        total = 0
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"[skip] unreadable: {path.name}")
                continue
            detections = detector.detect(frame)
            total += len(detections)
            annotated = annotate(detector, frame, detections)
            summary = ", ".join(f"{d.confidence:.2f}" for d in detections) or "none"
            print(f"{path.name[:60]:60s} notes={len(detections)} conf=[{summary}]")
            if save:
                cv2.imwrite(str(OUTPUT_DIR / f"{path.stem}_pred.jpg"), annotated)
        print(f"[info] processed {len(images)} image(s), {total} detection(s)")
        return 0

    if not source.is_file():
        print(f"[error] Source not found: {source}", file=sys.stderr)
        return 1

    # Anything else that OpenCV can open is treated as a video.
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[error] Could not open video: {source}", file=sys.stderr)
        return 1

    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if save:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = OUTPUT_DIR / f"{source.stem}_pred.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))
            print(f"[info] writing {out_path}")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            detections = detector.detect(frame)
            annotated = annotate(detector, frame, detections)
            if writer is not None:
                writer.write(annotated)
            if show_window:
                if show(annotated) in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return 0


def run_webcam(detector: RupeeDetector, args) -> int:
    cap = open_camera(args.camera, args.width, args.height)
    fps = 0.0
    last = time.perf_counter()
    failed_reads = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads > 30:
                    print("[error] Camera stopped returning frames.", file=sys.stderr)
                    return 1
                continue
            failed_reads = 0

            try:
                detections = detector.detect(frame)
            except DetectorError as exc:
                print(f"[warn] {exc}", file=sys.stderr)
                detections = []
            now = time.perf_counter()
            instant = 1.0 / max(now - last, 1e-6)
            fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
            last = now

            annotated = annotate(detector, frame, detections, fps)
            try:
                key = show(annotated)
            except cv2.error as exc:
                print(f"[error] Could not open a display window: {exc}", file=sys.stderr)
                return 1

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                print(f"[info] saved {save_frame(annotated, 'snapshot')}")
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        model_path = resolve_model(args.model)
        detector = RupeeDetector(model_path, conf_threshold=args.conf,
                                 iou_threshold=args.iou, use_cuda=args.cuda)
    except DetectorError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[info] model: {detector.model_path}")
    print(f"[info] input: {detector.input_hw}  classes: {detector.class_names}  "
          f"conf: {detector.conf_threshold}")

    if args.source:
        return run_source(detector, args.source, save=args.save, show_window=args.show)

    try:
        return run_webcam(detector, args)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
