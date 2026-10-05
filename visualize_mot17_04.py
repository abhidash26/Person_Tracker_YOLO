"""Create an annotated MP4 from the first MOT17-04 training frames.

Run from the project root:
    python visualize_mot17_04.py --max-frames 100
"""

import argparse
from pathlib import Path

from src.pipeline import create_default_pipeline, track_mot17_sequence
from src.visualize import write_tracking_video


DEFAULT_SEQUENCE = Path("data/MOT17/train/MOT17-04-FRCNN")
DEFAULT_OUTPUT = Path("outputs/mot17_04_first_100_tracks.mp4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    detector, tracker = create_default_pipeline()
    results = track_mot17_sequence(
        args.sequence_dir, detector, tracker, max_frames=args.max_frames
    )
    statistics = write_tracking_video(results, args.output, fps=30)

    print("Created annotated video:")
    for name, value in statistics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
