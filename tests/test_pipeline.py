import cv2
import numpy as np

from src.pipeline import track_mot17_sequence


class FakeDetector:
    """Supplies controlled detections so this test does not load YOLO weights."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return np.empty((0, 5))


class FakeTracker:
    """Records every update, including updates with no detections."""

    def __init__(self):
        self.detections_per_frame = []

    def update(self, detections):
        self.detections_per_frame.append(detections)
        return np.empty((0, 5))


def test_pipeline_reads_numbered_images_in_order_and_updates_every_frame(tmp_path):
    image_dir = tmp_path / "MOT17-04-DP" / "img1"
    image_dir.mkdir(parents=True)

    # Create deliberately non-lexical input order; pipeline must use numbers.
    for frame_number in (10, 1, 2):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{frame_number:06d}.jpg"), image)

    detector = FakeDetector()
    tracker = FakeTracker()
    results = track_mot17_sequence(image_dir.parent, detector, tracker, max_frames=3)

    assert [result["frame_number"] for result in results] == [1, 2, 10]
    assert detector.calls == 3
    assert len(tracker.detections_per_frame) == 3
    assert all(detections.shape == (0, 5) for detections in tracker.detections_per_frame)
