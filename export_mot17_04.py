"""Run YOLOv8n + SORT over all MOT17-04 frames and export predictions."""

from pathlib import Path
from time import perf_counter

from src.export import write_motchallenge_results
from src.pipeline import create_default_pipeline, track_mot17_sequence


SEQUENCE_DIR = Path("data/MOT17/train/MOT17-04-FRCNN")
OUTPUT_PATH = Path("outputs/mot17_04_tracks.txt")


def main():
    detector, tracker = create_default_pipeline()

    start_time = perf_counter()
    results = track_mot17_sequence(SEQUENCE_DIR, detector, tracker)
    export_statistics = write_motchallenge_results(results, OUTPUT_PATH)
    processing_seconds = perf_counter() - start_time

    frame_count = len(results)
    average_fps = frame_count / processing_seconds if processing_seconds else 0.0

    print(f"Processed frames: {frame_count}")
    print(f"Processing time (seconds): {processing_seconds:.2f}")
    print(f"Average FPS: {average_fps:.2f}")
    for name, value in export_statistics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
