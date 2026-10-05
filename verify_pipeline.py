"""Quickly run the MOT17 detection-and-tracking pipeline on a few frames.

Example (after placing MOT17-04 on disk):
    python verify_pipeline.py data/MOT17/train/MOT17-04-DP --max-frames 5
"""

import argparse
from pathlib import Path

from src.pipeline import create_default_pipeline, track_mot17_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence_dir",
        help="MOT17-04 sequence directory that contains img1",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Number of frames to process (default: 5)",
    )
    args = parser.parse_args()

    image_dir = Path(args.sequence_dir) / "img1"
    if not image_dir.is_dir():
        print(f"MOT17 images not found: {image_dir}")
        print("Place a MOT17-04 sequence there, then run this command again.")
        return

    detector, tracker = create_default_pipeline()
    results = track_mot17_sequence(
        args.sequence_dir, detector, tracker, max_frames=args.max_frames
    )

    print(f"Processed {len(results)} frame(s).")
    for result in results:
        print(
            f"Frame {result['frame_number']}: "
            f"{len(result['tracks'])} active track(s)"
        )


if __name__ == "__main__":
    main()
