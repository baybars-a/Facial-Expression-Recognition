# Facial Expression Recognition
Detects faces via webcam and classifies emotions in real-time.

## Emotions Detected
`happy` `sad` `angry` `surprise` `fear` `disgust` `neutral`

## Features
- Real-time emotion detection from webcam
- Probability bars showing confidence for each emotion
- Facial landmark overlay (eyes, eyebrows, iris, lips, nose) — toggle with **B**

## Requirements
Install dependencies with:
```
pip install fer mediapipe opencv-python
```

> **Note:** `fer` also requires TensorFlow and PyTorch. If you get missing dependency errors, run:
> ```
> pip install tensorflow torch
> ```

## Usage
```
python facial.py
```

## Controls
| Key | Action |
|-----|--------|
| `B` | Toggle facial landmark overlay |
| `Q` / `ESC` | Quit |

## Model File
The landmark overlay requires `face_landmarker.task` (included in this repo).
It is the [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) model.
