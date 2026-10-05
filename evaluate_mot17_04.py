"""Evaluate exported MOT17-04 SORT tracks against the training ground truth."""

from pathlib import Path

from src.evaluate import evaluate_motchallenge_sequence, save_evaluation_results


SEQUENCE_DIR = Path("data/MOT17/train/MOT17-04-FRCNN")
PREDICTION_PATH = Path("outputs/mot17_04_tracks.txt")
OUTPUT_PATH = Path("outputs/mot17_04_evaluation.csv")


def main():
    summary = evaluate_motchallenge_sequence(
        SEQUENCE_DIR / "gt" / "gt.txt",
        PREDICTION_PATH,
        SEQUENCE_DIR / "seqinfo.ini",
    )
    save_evaluation_results(summary, OUTPUT_PATH)

    metrics = summary.loc["MOT17-04-FRCNN"]
    print(f"MOTA: {metrics['mota']:.6f}")
    print(f"IDF1: {metrics['idf1']:.6f}")
    print(f"IDSW: {int(metrics['num_switches'])}")
    print(f"FP: {int(metrics['num_false_positives'])}")
    print(f"FN: {int(metrics['num_misses'])}")
    print(f"MT: {int(metrics['mostly_tracked'])}")
    print(f"ML: {int(metrics['mostly_lost'])}")
    print(f"Saved evaluation: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
