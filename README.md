# Person Multi-Object Tracking with YOLOv8n + SORT

A beginner-readable, end-to-end multi-object tracking pipeline for pedestrians.
It uses a pretrained **YOLOv8n** detector and a hand-built **SORT** tracker
evaluated on the **MOT17-04-FRCNN** benchmark sequence.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Architecture Overview](#2-architecture-overview)
3. [Dataset](#3-dataset)
4. [Detector](#4-detector)
5. [SORT Tracker](#5-sort-tracker)
6. [Project Structure](#6-project-structure)
7. [Installation & Setup](#7-installation--setup)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Visualization](#9-visualization)
10. [MOTChallenge Export](#10-motchallenge-export)
11. [Evaluation](#11-evaluation)
12. [Failure Analysis](#12-failure-analysis)
13. [Measured Results](#13-measured-results)
14. [Limitations](#14-limitations)

---

## 1. Project Objective

Given a video sequence of pedestrians, the system must:

- Detect every person in each frame using a neural network.
- Assign a **persistent numeric ID** to each person across frames.
- Handle brief occlusions (temporary disappearances).
- Quantify tracking accuracy with standard MOT metrics.
- Identify and visualize representative failure cases.

The implementation deliberately favours **simplicity and readability** over
maximum benchmark performance, because every design decision must be explainable
during evaluation.

---

## 2. Architecture Overview

```
┌─────────────┐       detections         ┌─────────────┐      tracks
│  Video      │  [x1,y1,x2,y2,conf]      │    SORT     │  [x1,y1,x2,y2,id]
│  frames     │ ───────────────────────► │   Tracker   │ ─────────────────►  Results
└─────────────┘                          └─────────────┘
       │                                       │
  YOLOv8n                            Kalman filter (predict)
  (COCO pretrained)                  IoU matrix (iou_batch)
  class=0 only                       Hungarian algorithm (assign)
                                     Track create / delete
```

**Per-frame flow:**

1. Read the next frame as a NumPy array with OpenCV.
2. `YoloDetector.detect(frame)` → bounding boxes for all persons (`conf ≥ 0.3`).
3. `Sort.update(detections)` →
   a. Each existing track's Kalman filter **predicts** where it will be.
   b. The **IoU matrix** between predicted positions and new detections is computed.
   c. The **Hungarian algorithm** finds the globally optimal detection-to-track assignment.
   d. Matched tracks are updated; unmatched detections become new tracks; tracks
      absent for more than `max_age` frames are deleted.
4. The resulting `[x1, y1, x2, y2, track_id]` rows are stored for every frame.

---

## 3. Dataset

| Property | Value |
|---|---|
| Dataset | [MOT17](https://motchallenge.net/data/MOT17/) |
| Sequence used | **MOT17-04-FRCNN** |
| Frames | 1,050 |
| Resolution | 1920 × 1080 |
| Frame rate | 30 fps |
| Annotations | 83 distinct pedestrian IDs |

MOT17-04 is a static-camera sequence of a busy pedestrian area.  Ground-truth
annotations follow the MOTChallenge format:
```
frame, id, left, top, width, height, active, class, visibility
```
Only rows where `active == 1` are counted in the evaluation.

---

## 4. Detector

**File:** `src/detector.py`

```python
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_weight='yolov8n.pt', conf_thresh=0.3):
        self.model = YOLO(model_weight)       # downloads weights on first run
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        # classes=[0] restricts to COCO class 0 = "person"
        results = self.model(frame, classes=[0], conf=self.conf_thresh, verbose=False)
        ...
        return dets   # shape (N, 5): [x1, y1, x2, y2, conf]
```

**Key choices:**

| Choice | Reason |
|---|---|
| YOLOv8n ("nano") | Smallest/fastest YOLOv8 variant; straightforward to explain |
| COCO pretrained | No fine-tuning required; persons are a native COCO class |
| `conf_thresh = 0.3` | Keeps recall high; SORT's track-age mechanism suppresses brief false positives |
| `classes=[0]` | Eliminates all non-person detections before they reach the tracker |

The detector returns boxes in `[x1, y1, x2, y2, conf]` format, which is the
format `Sort.update()` expects directly.

---

## 5. SORT Tracker

**File:** `src/tracker.py`

SORT (Simple Online and Realtime Tracking) uses two sub-components:

### 5.1 Kalman Filter (`KalmanBoxTracker`)

Each active track owns a `filterpy.kalman.KalmanFilter` with a **constant-velocity
motion model**.

**State vector (7 values):**

```
x = [cx, cy, s, r, cx', cy', s']
```

| Variable | Meaning |
|---|---|
| `cx`, `cy` | Centre of the bounding box |
| `s` | Scale (area = width × height) |
| `r` | Aspect ratio (width / height, held constant) |
| `cx'`, `cy'`, `s'` | Velocities of the above |

**Why this representation?**  
Tracking centre + scale + aspect ratio is more numerically stable than tracking
four corner coordinates independently.  `r` is treated as constant because
pedestrian body proportions do not change rapidly.

**Predict step** (called once per frame before matching):
```
x_predicted = F @ x_prev
```
The state transition matrix `F` adds the velocity to the position, giving a
linear extrapolation of where the person will be.

**Update step** (called after a detection is matched to this track):
```
x_updated = x_predicted + K @ (z - H @ x_predicted)
```
The Kalman gain `K` blends the prediction with the observed measurement `z`.

### 5.2 IoU Matrix (`iou_batch`)

Before matching, the IoU between every pair of (predicted track box, new
detection) is computed efficiently with NumPy broadcasting:

```python
# bb_test: (N, 4), bb_gt: (M, 4)
bb_test_exp = np.expand_dims(bb_test, 1)  # (N, 1, 4)
bb_gt_exp   = np.expand_dims(bb_gt,   0)  # (1, M, 4)
# element-wise min/max gives the (N, M) intersection corners
```

The result is an `(N, M)` matrix where entry `[i, j]` is IoU(track_i, det_j).

### 5.3 Hungarian Algorithm (`associate_detections_to_trackers`)

The IoU matrix is passed to `scipy.optimize.linear_sum_assignment` which finds
the globally optimal one-to-one assignment that **maximises total IoU**.
Any pair whose IoU falls below `iou_threshold = 0.3` is discarded even if it
was selected by the algorithm.

```python
row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # negate: minimise cost
```

### 5.4 Track Creation and Deletion (`Sort`)

| Event | Rule |
|---|---|
| **New track** | An unmatched detection spawns a new `KalmanBoxTracker` |
| **Track returned** | Only after `hit_streak >= min_hits` (default 3) consecutive matches — suppresses brief false positives |
| **Track coasting** | A matched track that receives no detection is kept alive for `max_age` frames, predicted forward by the Kalman filter alone |
| **Track deleted** | Removed when `time_since_update > max_age` (default 1) |

Default SORT parameters: `max_age=1`, `min_hits=3`, `iou_threshold=0.3`.

---

## 6. Project Structure

```
person-tracking-yolo-sort/
│
├── data/
│   └── MOT17/train/MOT17-04-FRCNN/
│       ├── img1/          # 1,050 JPEG frames
│       └── gt/gt.txt      # Ground-truth annotations
│
├── src/
│   ├── detector.py        # YOLOv8n wrapper → [x1,y1,x2,y2,conf] per frame
│   ├── tracker.py         # Custom SORT: Kalman filter + Hungarian matching
│   ├── pipeline.py        # Orchestrates detector + tracker over a sequence
│   ├── evaluate.py        # motmetrics integration (MOTA, IDF1, …)
│   ├── export.py          # Saves results in MOTChallenge .txt format
│   └── visualize.py       # Draws boxes + IDs onto frames, writes MP4
│
├── tests/
│   ├── test_tracker.py    # Unit tests: IoU, assignment, track lifecycle
│   ├── test_evaluate.py
│   ├── test_export.py
│   ├── test_pipeline.py
│   └── test_visualize.py
│
├── analysis/
│   └── analyze_failures.py  # Identifies representative ID-switch events
│
├── outputs/
│   ├── mot17_04_tracks.txt          # Raw tracking output (MOTChallenge format)
│   ├── mot17_04_evaluation.csv      # Metric summary
│   ├── id_switch_events.csv         # 10 representative events selected from the identified ID-switch cases.
│   └── id_switch_frames/            # Annotated images of those frames
│
├── verify_detector.py       # Smoke-test: run detector on a single image
├── verify_pipeline.py       # Smoke-test: run pipeline on a handful of frames
├── export_mot17_04.py       # Entry point: run full pipeline and save tracks
├── evaluate_mot17_04.py     # Entry point: compute and print metrics
├── visualize_mot17_04.py    # Entry point: render the annotated MP4
├── yolov8n.pt               # Downloaded model weights
├── requirements.txt
└── README.md
```

---

## 7. Installation & Setup

### Prerequisites

- Python 3.12 (the project was developed and tested with CPython 3.12.15)
- [`uv`](https://github.com/astral-sh/uv) package manager (fast, recommended)

### 1. Clone the repository

```bash
git clone <repo-url>
cd person-tracking-yolo-sort
```

### 2. Install `uv`

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$HOME\.local\bin;" + $env:Path
```

### 3. Create a virtual environment with Python 3.12

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux
```

### 4. Install dependencies

PyTorch and torchvision must be installed first from the CPU-only wheel index
(avoids downloading a large CUDA build if you do not have a GPU):

```powershell
uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install scipy opencv-python ultralytics filterpy pytest
```

> **Note:** `numpy<2` is required for compatibility with `torch==2.3.1`.
> `uv` resolves this automatically. If you install manually with pip, pin
> `numpy<2` explicitly.

### 5. Download the MOT17-04 sequence

Download MOT17 from [motchallenge.net](https://motchallenge.net/data/MOT17/) and
place the `MOT17-04-FRCNN` folder at:

```
data/MOT17/train/MOT17-04-FRCNN/
```

YOLOv8n weights (`yolov8n.pt`) are downloaded automatically on first run if
not already present in the project root.

---

## 8. Running the Pipeline

### Run tracker tests first

```powershell
python -m pytest tests/test_tracker.py -v
```

All 4 tests (IoU, partial overlap, assignment, lifecycle) should pass before
running the full pipeline.

### Run the full MOT17-04 tracking pipeline

```powershell
python export_mot17_04.py
```

This reads every frame in `data/MOT17/train/MOT17-04-FRCNN/img1/`, runs
YOLOv8n detection, updates SORT, and writes results to
`outputs/mot17_04_tracks.txt` in MOTChallenge format:

```
frame, track_id, left, top, width, height, conf, -1, -1, -1
```

### Quick smoke-test (first few frames only)

```powershell
python verify_pipeline.py
```

---

## 9. Visualization

```powershell
python visualize_mot17_04.py
```

Reads `outputs/mot17_04_tracks.txt` and the source frames, draws a colored
bounding box and `ID <n>` label for each active track, and writes:

```
outputs/mot17_04_first_100_tracks.mp4
```

Each track ID is assigned a deterministic color using simple arithmetic
(`color_for_track` in `src/visualize.py`) so the same person always appears in
the same color across runs.

---

## 10. MOTChallenge Export

```powershell
python export_mot17_04.py
```

`src/export.py` converts the `[x1, y1, x2, y2, track_id]` pipeline output to
the ten-column MOTChallenge submission format required by `motmetrics`:

```
frame, id, left, top, width, height, 1, -1, -1, -1
```

---

## 11. Evaluation

```powershell
python evaluate_mot17_04.py
```

`src/evaluate.py` uses `motmetrics.utils.CLEAR_MOT_M` to compare
`outputs/mot17_04_tracks.txt` against the MOT17-04 ground truth.  The evaluator:

- Filters GT to **active pedestrians** (`active == 1`) only.
- Uses an **IoU threshold of 0.5** for matching (standard MOTChallenge convention).
- Reports MOTA, IDF1, ID Switches, False Positives, False Negatives,
  Mostly Tracked, and Mostly Lost.

Results are saved to `outputs/mot17_04_evaluation.csv`.

---

## 12. Failure Analysis

```powershell
python -m analysis.analyze_failures
```

`analysis/analyze_failures.py` works entirely from the **already-generated**
`outputs/mot17_04_tracks.txt` and the ground-truth file — it does **not** rerun
the detector or tracker.

**Method:**

1. Parse both files frame-by-frame.
2. For each frame, greedily match GT boxes to predicted boxes by descending IoU
   (threshold 0.5 — same convention as the evaluator).
3. Record each GT pedestrian's last matched predicted ID.
4. Flag frames where the matched predicted ID changes for the same GT pedestrian
   — these are ID-switch events under this script's own matching logic.

The script identified **210 switch events** under its per-frame greedy matching.
This number differs from the evaluator's reported **192 ID switches** because
`motmetrics` uses a more sophisticated accumulation and deduplication strategy.
The two figures measure related but not identical things; the script's output is
best used to locate representative frames for qualitative inspection.

Outputs:

- `outputs/id_switch_events.csv` — 10 representative events (frame, GT ID,
  previous predicted ID, new predicted ID, IoU at switch time).
- `outputs/id_switch_frames/frame_XXXXXX.jpg` — annotated images: yellow box =
  GT pedestrian involved in the switch, green = other GT pedestrians, red = predictions.

**Representative events:**

| Frame | GT ID | Prev Pred ID | New Pred ID | IoU at switch |
|------:|------:|-------------:|------------:|:---:|
| 13    | 74    | 20           | 27          | 0.830 |
| 92    | 88    | 47           | 76          | 0.844 |
| 181   | 74    | 114          | 129         | 0.686 |
| 306   | 2     | 1            | 183         | 0.819 |
| 409   | 92    | 92           | 255         | 0.838 |
| 487   | 98    | 243          | 315         | 0.834 |
| 561   | 88    | 287          | 364         | 0.639 |
| 699   | 68    | 402          | 443         | 0.819 |
| 814   | 86    | 493          | 517         | 0.909 |
| 920   | 111   | 512          | 568         | 0.906 |

The high IoU values at switch time (0.64–0.91) show that the detection box is
accurate at the moment of the switch.  The switch is caused by Kalman-filter
drift: the old track's predicted position has drifted far enough from the true
position that the Hungarian algorithm reassigns the detection to a newer, closer
track.  The same GT IDs (74, 88) switch repeatedly across the sequence,
confirming that long-running pedestrians are the most affected.

---

## 13. Measured Results

Evaluated on **MOT17-04-FRCNN** (1,050 frames, training split ground truth).
All numbers are measured; none are estimated or interpolated.

| Metric | Value |
|---|---|
| **MOTA** (Multiple Object Tracking Accuracy) | **0.3228** (32.28%) |
| **IDF1** (Identity F1) | **0.3551** (35.51%) |
| **ID Switches** | **192** |
| **False Positives** | **1,427** |
| **False Negatives (Misses)** | **30,588** |
| Mostly Tracked (≥80% of lifespan) | 10 |
| Mostly Lost (≤20% of lifespan) | 40 |

**Interpretation:**

- **High false negatives (30,588)** are the dominant error.  MOT17-04 has many
  pedestrians that are distant, partially out-of-frame, or heavily occluded.
  YOLOv8n, which was not fine-tuned on MOT17, misses a large fraction of these.
- **Low false positives (1,427)** indicate the detector is conservative;
  non-person objects are rarely flagged.
- **192 ID switches** over 1,050 frames (≈ 0.18 per frame) is consistent with
  SORT's known limitation: without appearance features, any ambiguity in the IoU
  matching can cause identity loss, especially when tracks drift during coasting.
- **Mostly Lost: 40** reflects the same detection gap — many GT pedestrians are
  simply never detected often enough to be tracked.

---

## 14. Limitations

| Limitation | Explanation |
|---|---|
| **No appearance re-identification** | SORT uses position alone. Two people who cross paths, or a person who re-enters the frame, will receive a new ID. Appearance-based trackers (DeepSORT, ByteTrack) mitigate this. |
| **Short `max_age`** | Default `max_age=1` means a track is deleted after just one missed frame. Increasing it handles longer occlusions but raises the risk of ghost tracks. |
| **Detector not fine-tuned** | YOLOv8n is pretrained on COCO. Pedestrians at long range or in non-standard poses may be missed, driving the high false-negative count. |
| **Single sequence** | Results are reported on MOT17-04-FRCNN only. Performance varies across sequences with different crowd densities, camera motion, and lighting. |
| **CPU inference** | The pipeline runs on CPU (`torch+cpu`). Per-frame inference is slow (~1–3 seconds per frame); real-time use requires a GPU. |
| **Hungarian algorithm is O(n³)** | For very large numbers of simultaneous tracks this becomes a bottleneck, though it is not an issue for MOT17-04's crowd size. |
