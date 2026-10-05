import numpy as np

from src.evaluate import evaluate_motchallenge_sequence


def test_evaluator_ignores_distractor_match(tmp_path):
    ground_truth = tmp_path / "gt.txt"
    predictions = tmp_path / "tracks.txt"
    seqinfo = tmp_path / "seqinfo.ini"

    # One valid pedestrian and one annotated static-person distractor.
    ground_truth.write_text(
        "1,1,10,10,10,10,1,1,1\n"
        "1,2,40,40,10,10,0,7,1\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "1,11,10,10,10,10,1,-1,-1,-1\n"
        "1,12,40,40,10,10,1,-1,-1,-1\n",
        encoding="utf-8",
    )
    seqinfo.write_text("[Sequence]\nseqLength=1\n", encoding="utf-8")

    summary = evaluate_motchallenge_sequence(ground_truth, predictions, seqinfo)
    metrics = summary.loc["MOT17-04-FRCNN"]

    assert np.isclose(metrics["mota"], 1.0)
    assert metrics["num_false_positives"] == 0
