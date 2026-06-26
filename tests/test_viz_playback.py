"""Tests for the HoloViews viz builders and the headless playback guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import eztrack as ez
from eztrack.playback import _marker_track, opencv_is_headless, play_window


@pytest.fixture
def tracked(synthetic_video):
    s = ez.Session(dpath=str(synthetic_video.parent), file=synthetic_video.name)
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_pct=99, window=None), progress=False)
    return s, df


def test_heatmap_and_trace_build_elements(tracked):
    s, df = tracked
    assert type(ez.heatmap(s, df)).__name__ == "Image"
    assert type(ez.trace(s, df)).__name__ == "Overlay"


def test_outlier_plot_builds_layout(tracked):
    s, df = tracked
    filtered = ez.hampel_filter(df, s, window=3, sigma=3.0)
    assert type(ez.outlier_plot(df, filtered)).__name__ == "Layout"


def test_threshold_preview_builds_layout(tracked):
    s, _df = tracked
    layout = ez.threshold_preview(
        s, ez.TrackParams(threshold_pct=99, window=None), examples=2, sample=0
    )
    assert type(layout).__name__ == "Layout"


def test_threshold_preview_shows_undetected_frames_unbiased(tracked):
    """With an impossible cutoff every shown frame is undetected -- the preview must
    surface those failures, not search for the rare success, and still return."""
    s, _df = tracked
    layout = ez.threshold_preview(
        s, ez.TrackParams(threshold_abs=1000, window=None), examples=2, sample=0
    )
    assert type(layout).__name__ == "Layout"
    assert len(layout) == 4  # 2 examples x (original | difference), shown undetected


def test_detection_scan_reports_rate(tracked):
    """detection_scan returns a consistent isolated-frame detection rate."""
    s, _df = tracked
    rate = ez.detection_scan(s, ez.TrackParams(threshold_pct=99, window=None), n=30)
    assert rate["n_detected"] + rate["n_failed"] == rate["n_frames"]
    assert 0.0 <= rate["pct_detected"] <= 100.0

    # An unreachable cutoff detects nothing -> 0%.
    none = ez.detection_scan(s, ez.TrackParams(threshold_abs=1000, window=None), n=10)
    assert none["pct_detected"] == 0.0


def test_threshold_preview_prints_scan(tracked, capsys):
    """When sample > 0 the preview prints the detection-rate scan above the panels."""
    s, _df = tracked
    ez.threshold_preview(s, ez.TrackParams(threshold_pct=99, window=None), examples=1, sample=8)
    assert "Detection scan" in capsys.readouterr().out


def test_marker_track_aligns_by_frame_not_row():
    """Marker lookup follows frame numbers, so a sparse track stays in sync."""
    # A sparse track (every 4th frame), like temporal_downsample=4 produces.
    df = pd.DataFrame({"frame": [0, 4, 8], "x": [10.0, 50.0, 90.0], "y": [1.0, 2.0, 3.0]})
    pos = _marker_track(df, range(0, 10))

    # One marker position per video frame, indexed by absolute frame number.
    assert len(pos) == 10
    assert pos.loc[0, "x"] == 10.0 and pos.loc[4, "x"] == 50.0 and pos.loc[8, "x"] == 90.0
    # Skipped frames hold the last known position (not the next row's value).
    assert pos.loc[3, "x"] == 10.0 and pos.loc[7, "x"] == 50.0
    assert pos.loc[9, "x"] == 90.0


def test_marker_track_handles_offset_start():
    """A segment starting mid-session looks up the right frames (not row offsets)."""
    df = pd.DataFrame({"frame": [100, 101, 102], "x": [1.0, 2.0, 3.0], "y": [0.0, 0.0, 0.0]})
    pos = _marker_track(df, range(101, 103))
    assert list(pos["x"]) == [2.0, 3.0]  # frame 101 -> x=2, frame 102 -> x=3
    assert np.isnan(_marker_track(df, range(98, 100))["x"]).all()  # before first track -> NaN


def test_play_window_guarded_on_headless_build(synthetic_video):
    s = ez.Session(dpath=str(synthetic_video.parent), file=synthetic_video.name)
    if not opencv_is_headless():
        pytest.skip("GUI OpenCV build available; guard does not apply")
    with pytest.raises(RuntimeError, match="headless"):
        play_window(s, ez.PlayParams(), None)
