import numpy as np

from src.export import write_motchallenge_results


def test_export_writes_motchallenge_rows(tmp_path):
    results = [
        {
            "frame_number": 1,
            "tracks": np.array([[10.0, 20.0, 30.0, 50.0, 4.0]]),
        },
        {"frame_number": 2, "tracks": np.empty((0, 5))},
    ]
    output_path = tmp_path / "tracks.txt"

    statistics = write_motchallenge_results(results, output_path)

    assert output_path.read_text(encoding="utf-8") == "1,4,10.00,20.00,20.00,30.00,1,-1,-1,-1\n"
    assert statistics["line_count"] == 1
    assert statistics["first_track_frame"] == 1
    assert statistics["last_track_frame"] == 1
