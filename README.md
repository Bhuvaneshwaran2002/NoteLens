# NoteLens – ₹50 Indian Currency Note Detection

## Project Overview

NoteLens is a real-time Indian currency detection system designed to identify the **new light blue-green ₹50 banknote** (front and back) using a webcam feed.

The system uses a trained **YOLOv8 object detection model** exported to **ONNX format** and performs inference using **OpenCV DNN**. It detects ₹50 currency notes and displays a bounding box labelled **₹50** along with the confidence score.

## Key Features

- Real-time ₹50 currency note detection using a webcam.
- Supports both front and back sides of the new ₹50 note.
- YOLOv8-based object detection.
- ONNX model integration for lightweight inference.
- OpenCV DNN-based detection without requiring PyTorch or Ultralytics at runtime.
- Displays bounding boxes and confidence scores.
- Supports saving detection snapshots.

## Technologies Used

- **Programming Language:** Python
- **Object Detection:** YOLOv8
- **Model Format:** ONNX
- **Computer Vision:** OpenCV DNN
- **Camera Input:** OpenCV VideoCapture

## Project Structure

```text
NoteLens/
│
├── backend/
│   ├── models/
│   │   └── model.onnx
│   │
│   ├── outputs/
│   │
│   ├── test.py
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Bhuvaneshwaran2002/NoteLens.git
```

### 2. Navigate to the Backend Directory

```bash
cd NoteLens/backend
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Application

Connect your webcam and execute:

```bash
python test.py --model model.onnx
```

The application will start the webcam feed and detect the supported ₹50 currency note in real time.

**Model Location:**

The `model.onnx` file should be placed in the backend root directory or inside the `models/` directory.

If the `--model` argument is omitted, the application looks for `./model.onnx` and then `./models/model.onnx`.

## Keyboard Controls

| Key | Function |
|---|---|
| `Q` | Quit the application |
| `ESC` | Exit the application |
| `S` | Save a detection snapshot |

Saved snapshots are stored in the `outputs/` directory.

## Model Information

- **Model Architecture:** YOLOv8
- **Model Format:** ONNX
- **Inference Engine:** OpenCV DNN
- **Detection Target:** New light blue-green ₹50 Indian banknote
- **Detection Views:** Front and back

## Expected Output

The application displays a live webcam feed with:
- Bounding boxes around detected ₹50 notes.
- ₹50 denomination labels.
- Confidence scores for detected objects.

## Future Enhancements

- Support for additional Indian currency denominations.
- Fake currency detection and authenticity verification.
- Improved detection under different lighting conditions.
- Mobile application integration.
- Currency recognition using image uploads.

## Author

**Bhuvaneshwaran R**

GitHub: https://github.com/Bhuvaneshwaran2002

Project: NoteLens – ₹50 Indian Currency Note Detection

---
