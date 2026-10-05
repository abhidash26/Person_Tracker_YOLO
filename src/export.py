"""Write pipeline tracking results in the MOTChallenge result-file format."""

from pathlib import Path


# Sort.update returns only [x1, y1, x2, y2, track_id].  Its output does not
# retain the YOLO detection confidence after association.  MOTChallenge result
# files conventionally use confidence 1 for a valid tracker hypothesis.
TRACK_CONFIDENCE = 1


def write_motchallenge_results(results, output_path):
    """Save pipeline results as MOTChallenge tracking predictions.

    Each line has: frame, track_id, x, y, width, height, confidence,
    -1, -1, -1.  Frames with no active tracks correctly have no output row.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    frame_numbers_with_tracks = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for result in results:
            frame_number = result["frame_number"]
            tracks = result["tracks"]

            if len(tracks) > 0:
                frame_numbers_with_tracks.append(frame_number)

            for x1, y1, x2, y2, track_id in tracks:
                width = x2 - x1
                height = y2 - y1
                output_file.write(
                    f"{frame_number},{int(track_id)},{x1:.2f},{y1:.2f},"
                    f"{width:.2f},{height:.2f},{TRACK_CONFIDENCE},-1,-1,-1\n"
                )
                line_count += 1

    return {
        "output_path": str(output_path),
        "line_count": line_count,
        "first_track_frame": (
            min(frame_numbers_with_tracks) if frame_numbers_with_tracks else None
        ),
        "last_track_frame": (
            max(frame_numbers_with_tracks) if frame_numbers_with_tracks else None
        ),
    }
