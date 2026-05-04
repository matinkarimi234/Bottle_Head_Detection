# Bottle Mouth Detection Comparison with Model Downloader

This project compares multiple ways to detect the mouth/head of capless bottles in video and export pixel locations.

## Files

- `bottle_mouth_detection_comparison_with_models.ipynb` — main notebook
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores data, models, outputs, and generated files

## What the notebook can download

- A pretrained Ultralytics YOLO test model
- A MediaPipe sample EfficientDet-Lite0 `.tflite` model

These downloaded models are only for pipeline testing. They are not trained specifically for bottle mouths.

## Recommended final model

Train your own YOLO model with one class:

```text
bottle_mouth
```

Then put the trained file here:

```text
models/yolo_bottle_mouth.pt
```

## Basic usage

```bash
pip install -r requirements.txt
jupyter notebook bottle_mouth_detection_comparison_with_models.ipynb
```

Put your video here:

```text
data/input_video.mp4
```

The notebook outputs CSV files and preview images under:

```text
outputs/
```
