"""Run person detection and SORT tracking over a MOT17 image sequence.

MOT17 stores a sequence as numbered JPEG files in an ``img1`` directory.
This module deliberately handles that format directly instead of treating the
sequence as a video file.
"""

from pathlib import Path

import cv2


def get_mot17_frame_paths(sequence_dir):
    """Return MOT17 ``img1`` JPEG frames ordered by their frame number.

    ``sequence_dir`` is the directory that contains ``img1``; for example,
    ``data/MOT17/train/MOT17-04-DP``.  MOT17 names frames ``000001.jpg``,
    ``000002.jpg``, and so on, so sorting the numeric filename stem gives the
    chronological order.
    """
    image_dir = Path(sequence_dir) / "img1"
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find an MOT17 img1 directory at: {image_dir}"
        )

    frame_paths = list(image_dir.glob("*.jpg"))
    return sorted(frame_paths, key=lambda path: int(path.stem))


def track_mot17_sequence(sequence_dir, detector, tracker, max_frames=None):
    """Track people in a MOT17 image sequence.

    Args:
        sequence_dir: MOT17 sequence directory containing an ``img1`` folder.
        detector: A ``YoloDetector`` (or a test double with ``detect``).
        tracker: A ``Sort`` instance (or a test double with ``update``).
        max_frames: Optional positive limit useful for quick verification.

    Returns:
        A list of dictionaries.  Each dictionary contains the MOT17 frame
        number, image path, and an ``(N, 5)`` NumPy array of tracks in
        ``[x1, y1, x2, y2, track_id]`` format.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be a positive integer or None")

    frame_paths = get_mot17_frame_paths(sequence_dir)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    results = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Could not read image: {frame_path}")

        detections = detector.detect(frame)
        # This call stays inside the loop so SORT also advances on empty frames.
        tracks = tracker.update(detections)

        results.append(
            {
                "frame_number": int(frame_path.stem),
                "frame_path": str(frame_path),
                "tracks": tracks.copy(),
            }
        )

    return results


def create_default_pipeline(conf_thresh=0.3):
    """Create the project detector and tracker with beginner-friendly defaults."""
    # Import here rather than at module load time.  This keeps the simple frame
    # ordering helpers usable in tests that inject lightweight test doubles.
    from src.detector import YoloDetector
    from src.tracker import Sort

    detector = YoloDetector(conf_thresh=conf_thresh)
    tracker = Sort()
    return detector, tracker
