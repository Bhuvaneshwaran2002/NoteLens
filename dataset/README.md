# Dataset — new ₹50 note

Put **your own photos** here. The assignment rules are strict:

- Only the **new light blue-green ₹50 note** (front: Mahatma Gandhi portrait,
  back: Hampi stone chariot). Older ₹50 notes are **not** part of this class.
- Also photograph other notes (**₹10, ₹20, ₹100, ₹200, ₹500**) so the model
  learns what is **not** ₹50.
- Do **not** download note images from the internet.
- Do **not** print, photocopy, or create fake notes.

## Layout

```
dataset/
  data.yaml
  images/
    train/   <-- training photos (.jpg/.jpeg/.png)
    val/     <-- validation photos
    test/    <-- optional test photos
  labels/
    train/   <-- one YOLO .txt per training image
    val/
    test/
```

A label file has the **same stem** as the image, e.g.
`images/train/note_001.jpg` -> `labels/train/note_001.txt`.

## YOLO label format (single class)

One line per ₹50 note:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four coordinates are normalized to `0..1`. `class_id` is always `0`
(`rupee_50`). Example:

```
0 0.512 0.480 0.640 0.290
```

If an image contains **no** ₹50 note (e.g. a ₹500 note), create an **empty**
label file with the same stem:

```
labels/train/rupee500_01.txt   (0 bytes)
```

This teaches the model that the object is not ₹50. Missing label files are also
treated as background by Ultralytics, but empty files make the intent explicit
and keep the dataset easy to audit.

## Suggested capture plan

- ₹50 front and back: vary lighting (bright/dim/warm), angle (flat, tilted,
  rotated), distance, background (table, hand, clutter), and partial occlusion.
- Other notes: same variety, including ₹50 look-alikes under bad lighting.
- Aim for roughly **300–600 images**, with about **70–80% train / 20–30% val**.
  Keep a few unseen photos for `test/`.
- Recommended negative ratio: at least one non-₹50 image for every ₹50 image.

## Labeling tools

Any YOLO-format annotator works, for example
[LabelImg](https://github.com/HumanSignal/labelImg) or
[CVAT](https://www.cvat.ai/). Use a single class named `rupee_50`.

### Speed up the first pass

Pre-label a folder of raw photos with the existing model, then correct the
result in your annotator:

```bash
python scripts/prelabel.py --source photos/ --val-fraction 0.2
```

It copies photos into `dataset/images/{train,val}/` (stable hash split) and
writes matching labels into `dataset/labels/{train,val}/`. **Always review the
output**: a note the model missed becomes an empty (background) label, which
would teach the wrong thing if left uncorrected.

## Check the dataset

From the repository root:

```bash
python scripts/validate_dataset.py
```

Then train and export the ONNX model:

```bash
python train.py --data dataset/data.yaml --epochs 100 --imgsz 512
```
