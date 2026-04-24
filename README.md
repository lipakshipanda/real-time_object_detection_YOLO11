# Advanced Real-Time YOLO11 Detector

A production-grade real-time object detector built exclusively on **YOLO11**
(the latest Ultralytics architecture), with a full feature set far beyond the
original starter project.

---

## What's new vs the original

| Feature | Original | Advanced |
|---------|----------|----------|
| YOLO version | v8n (original project) | **YOLO11** (n / s / m / l / x) |
| Tasks | detect only | **detect / segment / pose / OBB** |
| Tracker | basic | ByteTrack **+** BoT-SORT (switchable) |
| FPS counter | single-frame | **30-frame rolling average** |
| Confidence control | fixed | **live ±0.05 with + / − keys** |
| Class filter | ✗ | **`--classes person car …`** |
| ROI zone | ✗ | **Polygon ROI with alpha overlay** |
| Virtual line counter | ✗ | **IN / OUT crossing counter** |
| Object trails | ✗ | **Alpha-faded centroid trails** |
| Per-class HUD | ✗ | **Live count panel per class** |
| Segmentation masks | ✗ | **Coloured alpha overlays** |
| Pose keypoints | ✗ | **Drawn on body joints** |
| Screenshot | ✗ | **S key → timestamped PNG** |
| Video recording | ✗ | **R key → MP4 toggle** |
| Pause / resume | ✗ | **P key** |
| Device auto-select | ✗ | **CUDA → MPS → CPU** |
| Frame skipping | ✗ | **`--skip N` for perf tuning** |
| Config file | ✗ | **`configs/default.ini`** |
| End-of-run stats | ✗ | **Total detections, avg ms** |

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Webcam with YOLO11n (default – fastest)
python main.py

# Video file with YOLO11s (better accuracy)
python main.py --source data/sample.mp4 --model yolo11s.pt

# Only track people and cars
python main.py --classes person car

# Instance segmentation
python main.py --model yolo11n-seg.pt --task segment

# Pose estimation (skeleton keypoints)
python main.py --model yolo11n-pose.pt --task pose

# Count objects crossing a horizontal line at y=360
python main.py --line 0,360 1280,360

# Restrict detection to a rectangular ROI
python main.py --roi 200,100 1000,100 1000,600 200,600

# Save output video immediately
python main.py --record exports/output.mp4

# Limit display width + skip every other frame (slow hardware)
python main.py --width 640 --skip 1
```

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| S | Save screenshot to `exports/` |
| R | Toggle video recording on/off |
| P | Pause / resume |
| H | Toggle help overlay |
| F | Toggle object trails |
| + | Raise confidence threshold +0.05 |
| − | Lower confidence threshold −0.05 |

---

## Supported YOLO11 models (auto-download on first run)

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolo11n.pt` | 5.4 MB | ★★★★★ | ★★☆ |
| `yolo11s.pt` | 18 MB  | ★★★★☆ | ★★★ |
| `yolo11m.pt` | 38 MB  | ★★★☆☆ | ★★★★ |
| `yolo11l.pt` | 49 MB  | ★★☆☆☆ | ★★★★ |
| `yolo11x.pt` | 109 MB | ★☆☆☆☆ | ★★★★★ |
| `yolo11n-seg.pt` | — | fast | segmentation |
| `yolo11n-pose.pt` | — | fast | pose |
| `yolo11n-obb.pt` | — | fast | rotated boxes |

---

## Project structure

```
realtime_yolo_advanced/
├── main.py
├── requirements.txt
├── configs/default.ini
├── models/
├── src/
│   ├── detector.py
│   └── utils.py
├── data/
├── exports/
└── logs/
```

---

## Tips

- **GPU (CUDA)**: swap the torch line in `requirements.txt` for the CUDA wheel.
- **Apple Silicon**: MPS selected automatically.
- **CPU slow hardware**: `python main.py --model yolo11n.pt --skip 1 --width 640`
