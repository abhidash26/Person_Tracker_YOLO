from src.visualize import color_for_track


def test_track_color_is_deterministic():
    assert color_for_track(7) == color_for_track(7)
    assert color_for_track(7) != color_for_track(8)
