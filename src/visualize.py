"""Simple OpenCV drawing helpers for person-tracking results."""

from pathlib import Path

import cv2


def color_for_track(track_id):
    """Return a repeatable BGR color for one SORT track ID.

    The arithmetic is intentionally simple: it gives each ID a stable color
    without needing to store a color table between frames.
    """
    return (
        (37 * int(track_id)) % 256,
        (17 * int(track_id) + 80) % 256,
        (29 * int(track_id) + 160) % 256,
    )


def draw_tracks(frame, tracks):
    """Draw ``[x1, y1, x2, y2, track_id]`` SORT tracks on a frame."""
    annotated_frame = frame.copy()

    for x1, y1, x2, y2, track_id in tracks:
        track_id = int(track_id)
        color = color_for_track(track_id)
        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))

        cv2.rectangle(annotated_frame, top_left, bottom_right, color, 2)
        cv2.putText(
            annotated_frame,
            f"ID {track_id}",
            (top_left[0], max(20, top_left[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


def write_tracking_video(results, output_path, fps=30):
    """Write annotated pipeline results to an MP4 file.

    ``results`` is the list returned by ``track_mot17_sequence``.  The source
    images are read again here only for drawing; the pipeline remains focused
    on detection and tracking.
    """
    if not results:
        raise ValueError("Cannot create a video from an empty result list")

    first_frame = cv2.imread(results[0]["frame_path"])
    if first_frame is None:
        raise ValueError(f"Could not read image: {results[0]['frame_path']}")

    height, width = first_frame.shape[:2]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")

    total_tracks = 0
    try:
        for result in results:
            frame = cv2.imread(result["frame_path"])
            if frame is None:
                raise ValueError(f"Could not read image: {result['frame_path']}")

            tracks = result["tracks"]
            total_tracks += len(tracks)
            writer.write(draw_tracks(frame, tracks))
    finally:
        writer.release()

    return {
        "output_path": str(output_path),
        "frame_count": len(results),
        "width": width,
        "height": height,
        "fps": fps,
        "total_active_tracks": total_tracks,
    }
