# Project Context

## Project

Person Multi-Object Tracking Pipeline using YOLOv8n + custom SORT.

The goal is a beginner-readable implementation for MOT17, initially
focusing on MOT17-04.

## Current Environment

- Python 3.12.15
- NumPy 1.26.4
- PyTorch 2.3.1+cpu
- torchvision 0.18.1+cpu
- Ultralytics 8.4.173
- Local virtual environment: `.venv`

## Current Progress

- Project structure created.
- Custom SORT implementation created in `src/tracker.py`.
- `tests/test_tracker.py` created.
- SORT tests: 4/4 passing.
- YOLOv8n detector wrapper created in `src/detector.py`.
- `verify_detector.py` created.
- Detector verification succeeds.
- YOLOv8n successfully detects people and outputs:
  `[x1, y1, x2, y2, confidence]`.

## Architecture

YOLOv8n:
image/frame
→ person detections
→ bounding boxes + confidence

SORT:
detections
→ IoU matching
→ Hungarian assignment
→ Kalman filter updates
→ persistent track IDs

## Important Constraint

This is an educational project.

The student must be able to explain the code during evaluation.
Prefer simple, explicit implementations over sophisticated abstractions.

Do not rewrite working components without a clear reason.

Do not implement everything in one giant step. Work component-by-component,
run tests after changes, and explain important design decisions.

## Next Planned Work

1. Inspect and validate detector/tracker interfaces.
2. Implement `pipeline.py`.
3. Download/use MOT17-04.
4. Run tracking.
5. Implement evaluation with `motmetrics`.
6. Implement visualization.
7. Implement failure/ID-switch analysis.
8. Extend evaluation to additional MOT17 sequences.
9. Complete README and technical explanation.