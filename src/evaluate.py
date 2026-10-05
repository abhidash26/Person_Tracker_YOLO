"""Evaluate MOTChallenge tracking results with motmetrics."""

from pathlib import Path

import motmetrics as mm


METRICS = [
    "mota",
    "idf1",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "mostly_tracked",
    "mostly_lost",
]


def evaluate_motchallenge_sequence(ground_truth_path, prediction_path, seqinfo_path):
    """Return MOTChallenge metrics for one sequence.

    ``CLEAR_MOT_M`` is motmetrics' MOTChallenge-aware comparison helper.  It
    uses the sequence metadata, filters GT to active class-1 pedestrians, and
    removes matched hypotheses on annotated distractors/ignored objects before
    calculating metrics.  Matching uses the standard IoU threshold of 0.5.
    """
    ground_truth_path = Path(ground_truth_path)
    prediction_path = Path(prediction_path)
    seqinfo_path = Path(seqinfo_path)

    for path in (ground_truth_path, prediction_path, seqinfo_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation file not found: {path}")

    # Keep every GT row here. CLEAR_MOT_M needs ignored/distractor rows during
    # its preprocessing step, then selects active pedestrian GT internally.
    ground_truth = mm.io.loadtxt(
        str(ground_truth_path), fmt=mm.io.Format.MOT16, min_confidence=-1
    )
    predictions = mm.io.loadtxt(
        str(prediction_path), fmt=mm.io.Format.MOT16, min_confidence=0
    )

    accumulator, _ = mm.utils.CLEAR_MOT_M(
        ground_truth,
        predictions,
        str(seqinfo_path),
        dist="iou",
        distth=0.5,
    )

    metrics_host = mm.metrics.create()
    return metrics_host.compute(accumulator, metrics=METRICS, name="MOT17-04-FRCNN")


def save_evaluation_results(summary, output_path):
    """Save the one-row metric summary as a readable CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    return output_path
