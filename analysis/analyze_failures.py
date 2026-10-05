"""Identify and report representative ID-switch events from existing tracking outputs.

This script does NOT run the YOLO detector or SORT tracker.  It reads the
already-generated prediction file and the MOT17 ground-truth file, matches
detections frame-by-frame with IoU >= 0.5 (same convention as the evaluator),
then identifies frames where a GT pedestrian that was previously matched to one
predicted ID is suddenly matched to a different predicted ID.

Outputs
-------
outputs/id_switch_events.csv
    One row per representative ID-switch event (up to 10).
outputs/id_switch_frames/
    Annotated JPEG crops of each representative frame (if the source images are
    available).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# File paths – relative to the project root
# ---------------------------------------------------------------------------
TRACKS_FILE = Path("outputs/mot17_04_tracks.txt")
GT_FILE = Path("data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt")
IMAGES_DIR = Path("data/MOT17/train/MOT17-04-FRCNN/img1")
OUTPUT_CSV = Path("outputs/id_switch_events.csv")
OUTPUT_IMAGES_DIR = Path("outputs/id_switch_frames")

IOU_THRESHOLD = 0.5       # Match threshold – same as the evaluator
MAX_REPRESENTATIVE = 10   # How many events to report


# ---------------------------------------------------------------------------
# Step 1 – Parsing helpers
# ---------------------------------------------------------------------------

def parse_mot_file(path, active_only=False):
    """Read a MOTChallenge CSV into a dict: frame -> list of (id, x1, y1, x2, y2).

    The MOTChallenge format is:
        frame, id, left, top, width, height, active, class, ...

    ``active_only=True`` filters ground-truth rows to active pedestrians
    (column 6 == 1), matching the convention used by motmetrics.
    """
    cols = ["frame", "id", "left", "top", "width", "height", "active"]
    df = pd.read_csv(path, header=None, usecols=range(7), names=cols)

    if active_only:
        # Keep only active rows (pedestrians that count toward metrics)
        df = df[df["active"] == 1]

    # Convert (left, top, width, height) to (x1, y1, x2, y2)
    df["x1"] = df["left"]
    df["y1"] = df["top"]
    df["x2"] = df["left"] + df["width"]
    df["y2"] = df["top"] + df["height"]

    # Group into a dict for fast per-frame access
    by_frame = {}
    for frame_no, group in df.groupby("frame"):
        by_frame[int(frame_no)] = list(
            zip(
                group["id"].astype(int),
                group["x1"].astype(float),
                group["y1"].astype(float),
                group["x2"].astype(float),
                group["y2"].astype(float),
            )
        )
    return by_frame


# ---------------------------------------------------------------------------
# Step 2 – IoU between two single boxes [x1, y1, x2, y2]
# ---------------------------------------------------------------------------

def iou(box_a, box_b):
    """Compute IoU between two boxes given as (x1, y1, x2, y2) tuples."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    # Union
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Step 3 – Per-frame greedy matching (GT → prediction, highest IoU first)
# ---------------------------------------------------------------------------

def match_frame(gt_boxes, pred_boxes, iou_threshold=IOU_THRESHOLD):
    """Match GT boxes to prediction boxes greedily by descending IoU.

    Returns a list of (gt_id, pred_id, matched_iou) tuples for matched pairs.
    Unmatched GT/pred rows are ignored for the purpose of switch detection.
    """
    matches = []

    # Build all (gt_idx, pred_idx, iou) triples and sort by IoU descending
    candidates = []
    for gi, (gt_id, *gt_box) in enumerate(gt_boxes):
        for pi, (pred_id, *pred_box) in enumerate(pred_boxes):
            score = iou(gt_box, pred_box)
            if score >= iou_threshold:
                candidates.append((score, gi, pi, gt_id, pred_id))

    candidates.sort(key=lambda t: -t[0])

    used_gt = set()
    used_pred = set()
    for score, gi, pi, gt_id, pred_id in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        matches.append((gt_id, pred_id, score))
        used_gt.add(gi)
        used_pred.add(pi)

    return matches


# ---------------------------------------------------------------------------
# Step 4 – Scan all frames and detect ID-switch events
# ---------------------------------------------------------------------------

def find_id_switches(gt_by_frame, pred_by_frame):
    """Return a list of dicts, one for each detected ID-switch event.

    An ID switch occurs when a GT pedestrian (identified by GT ID) that was
    matched to predicted ID X in a previous frame is matched to a different
    predicted ID Y in the current frame.
    """
    # gt_id -> last predicted id it was matched to
    gt_to_last_pred = {}
    switch_events = []

    for frame_no in sorted(gt_by_frame.keys()):
        gt_boxes = gt_by_frame.get(frame_no, [])
        pred_boxes = pred_by_frame.get(frame_no, [])

        if not gt_boxes or not pred_boxes:
            continue  # Nothing to match this frame

        matches = match_frame(gt_boxes, pred_boxes)

        for gt_id, pred_id, matched_iou in matches:
            last_pred = gt_to_last_pred.get(gt_id)

            if last_pred is not None and last_pred != pred_id:
                # The same GT pedestrian was previously tracked under a
                # different predicted ID – this is an ID switch.
                switch_events.append(
                    {
                        "frame": frame_no,
                        "gt_id": gt_id,
                        "prev_pred_id": last_pred,
                        "new_pred_id": pred_id,
                        "iou": round(matched_iou, 4),
                    }
                )

            gt_to_last_pred[gt_id] = pred_id

    return switch_events


# ---------------------------------------------------------------------------
# Step 5 – Annotate and save a representative frame image
# ---------------------------------------------------------------------------

def save_annotated_frame(frame_no, gt_boxes, pred_boxes, switch_gt_ids, output_dir):
    """Draw GT (green) and pred (red) boxes on the frame image and save it.

    GT boxes involved in an ID switch are highlighted in yellow.
    """
    image_path = IMAGES_DIR / f"{frame_no:06d}.jpg"
    if not image_path.exists():
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    # Draw all GT boxes (green; yellow if involved in a switch)
    for gt_id, x1, y1, x2, y2 in gt_boxes:
        color = (0, 255, 255) if gt_id in switch_gt_ids else (0, 200, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            img, f"GT{gt_id}", (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    # Draw all prediction boxes (red)
    for pred_id, x1, y1, x2, y2 in pred_boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 220), 2)
        cv2.putText(
            img, f"P{pred_id}", (int(x1), int(y2) + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2, cv2.LINE_AA,
        )

    # Legend
    cv2.putText(img, f"Frame {frame_no}  (yellow=GT switch, green=GT ok, red=pred)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    output_path = output_dir / f"frame_{frame_no:06d}.jpg"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Parsing ground truth ...")
    gt_by_frame = parse_mot_file(GT_FILE, active_only=True)

    print("Parsing predictions ...")
    pred_by_frame = parse_mot_file(TRACKS_FILE, active_only=False)

    print("Scanning for ID-switch events ...")
    all_switches = find_id_switches(gt_by_frame, pred_by_frame)
    print(f"  Total ID-switch events found: {len(all_switches)}")

    if not all_switches:
        print("No ID switches detected.")
        return

    # Pick representative events spread across the sequence
    step = max(1, len(all_switches) // MAX_REPRESENTATIVE)
    representatives = all_switches[::step][:MAX_REPRESENTATIVE]

    print(f"\n{'Frame':>6}  {'GT ID':>5}  {'Prev Pred':>9}  {'New Pred':>8}  {'IoU':>5}")
    print("-" * 44)
    for ev in representatives:
        print(
            f"{ev['frame']:>6}  {ev['gt_id']:>5}  "
            f"{ev['prev_pred_id']:>9}  {ev['new_pred_id']:>8}  {ev['iou']:>5.3f}"
        )

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(representatives).to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CSV -> {OUTPUT_CSV}")

    # Save annotated images
    print("Saving annotated frame images ...")
    for ev in representatives:
        frame_no = ev["frame"]
        gt_boxes = gt_by_frame.get(frame_no, [])
        pred_boxes = pred_by_frame.get(frame_no, [])
        switch_gt_ids = {ev["gt_id"]}
        out = save_annotated_frame(frame_no, gt_boxes, pred_boxes,
                                   switch_gt_ids, OUTPUT_IMAGES_DIR)
        if out:
            print(f"  Saved frame {frame_no} -> {out}")
        else:
            print(f"  Skipped frame {frame_no} (image not found)")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
