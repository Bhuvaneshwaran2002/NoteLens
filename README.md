# ₹50 Note Detector — ZED Digital (FEDS Problem Statement 3)

Detects **only the new light blue-green ₹50 note** (front and back) in a live
webcam feed and draws a box labelled **“₹50”** with the confidence score.
The trained YOLOv8 model is exported to **ONNX** and runs through **OpenCV DNN**,
so no PyTorch/Ultralytics install is needed to run it.

## Install and run

```bash
python -m pip install -r requirements.txt
python test.py --model model.onnx
```

`model.onnx` is at the repository root. If `--model` is omitted, `test.py`
looks for `./model.onnx` and then `./models/model.onnx`.

Keys: **q / Esc** quit, **s** save a snapshot to `outputs/`.

Useful options:

```bash
python test.py --camera 1 --conf 0.6 --iou 0.45
python test.py --source photo.jpg --save      # test an image
python test.py --source samples/ --save       # test a folder of images
python test.py --source clip.mp4 --save       # test a video
```

## Train

The dataset folder is ready, but **you must add your own photos and labels**
(see [`dataset/README.md`](dataset/README.md)). Then:

```bash
pip install ultralytics
python scripts/prelabel.py --source photos/      # optional first pass, then review
python scripts/validate_dataset.py
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 100 --imgsz 512
```

Training writes `runs/rs50/weights/best.pt` and exports the ONNX model to
`model.onnx` (and `models/model.onnx`). The export uses a **512×512** input and a
single class, `rupee_50`.

## What is in the repo

| Path | Purpose |
| --- | --- |
| `test.py` | Webcam test script required by the assignment (`₹50` + confidence) |
| `train.py` | Training + ONNX export |
| `model.onnx` | Trained ONNX model (OpenCV-ready) |
| `dataset/` | Placeholder layout + `data.yaml`; add your own photos/labels |
| `src/` | Shared ONNX preprocessing, decoding, NMS, drawing |
| `scripts/self_check.py` | Offline correctness checks without a webcam |
| `scripts/validate_dataset.py` | Dataset/label sanity checks |
| `scripts/prelabel.py` | First-pass labels from raw photos (review before training) |
| `model_improvement/` | Optional experiment harness and evaluation tools |
| `backend/`, `frontend/` | Optional browser demo (not needed for the assignment) |

## Model details

- YOLOv8n, single class `0: rupee_50`, input `512×512`, output `[1, 5, 5376]`
- Confidence threshold `0.6`, NMS IoU `0.45` (configurable with `--conf` / `--iou`)
- Pre/post-processing (letterbox, decode, class-aware NMS) in `src/onnx_utils.py`

## Quick offline verification

```bash
python scripts/self_check.py --model model.onnx
python scripts/self_check.py --source backend/tests/fixtures/rupee50_train_sample.jpg
```
