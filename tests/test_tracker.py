import numpy as np
import pytest
from src.tracker import iou_batch, associate_detections_to_trackers, Sort, KalmanBoxTracker

def test_iou_batch():
    # Bbox format: [x1, y1, x2, y2]
    # bb_test has one bounding box
    bb_test = np.array([[10, 10, 20, 20]])
    
    # bb_gt has two bounding boxes. First matches exactly, second has no overlap.
    bb_gt = np.array([
        [10, 10, 20, 20],
        [30, 30, 40, 40]
    ])
    
    iou = iou_batch(bb_test, bb_gt)
    
    assert iou.shape == (1, 2)
    assert np.isclose(iou[0, 0], 1.0)
    assert np.isclose(iou[0, 1], 0.0)

def test_iou_partial_overlap():
    bb1 = np.array([[0, 0, 10, 10]])
    bb2 = np.array([[5, 5, 15, 15]])
    
    iou = iou_batch(bb1, bb2)
    
    # Intersection is [5, 5, 10, 10] => area = 25
    # Union is 100 + 100 - 25 = 175
    expected_iou = 25 / 175
    assert np.isclose(iou[0, 0], expected_iou)

def test_associate_detections_to_trackers():
    detections = np.array([
        [10, 10, 20, 20],
        [50, 50, 60, 60],
        [100, 100, 110, 110]
    ])
    
    trackers = np.array([
        [11, 11, 21, 21], # Matches det 0
        [100, 100, 110, 110] # Matches det 2
    ])
    
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers, iou_threshold=0.3)
    
    # Should have 2 matches
    assert matches.shape == (2, 2)
    
    # Det 1 is unmatched
    assert len(unmatched_dets) == 1
    assert unmatched_dets[0] == 1
    
    # No unmatched trackers
    assert len(unmatched_trks) == 0

def test_sort_lifecycle():
    tracker = Sort(max_age=1, min_hits=1)
    
    # Frame 1: 1 detection
    dets1 = np.array([[10, 10, 20, 20, 0.9]])
    tracks1 = tracker.update(dets1)
    
    assert len(tracks1) == 1
    track_id1 = tracks1[0][4]
    
    # Frame 2: Same detection moved slightly
    dets2 = np.array([[11, 11, 21, 21, 0.9]])
    tracks2 = tracker.update(dets2)
    
    assert len(tracks2) == 1
    assert tracks2[0][4] == track_id1
    
    # Frame 3: No detections (track is coasting, but max_age=1 so it won't be returned since hits < min_hits, wait min_hits=1 so it might be deleted)
    # Actually SORT typically requires a minimum number of hits to return a track initially. 
    # Let's verify it gets deleted if we miss it for max_age frames.
    dets3 = np.empty((0, 5))
    tracks3 = tracker.update(dets3)
    assert len(tracks3) == 0
    
    # Frame 4: No detections. By now max_age is reached, track should be deleted.
    dets4 = np.empty((0, 5))
    tracks4 = tracker.update(dets4)
    assert len(tracker.trackers) == 0
