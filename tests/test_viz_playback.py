"""Tests for the HoloViews viz builders and the headless playback guard."""

from __future__ import annotations

import pytest

import eztrack as ez
from eztrack.playback import opencv_is_headless, play_window


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


def test_threshold_preview_builds_layout(tracked):
    s, _df = tracked
    layout = ez.threshold_preview(s, ez.TrackParams(threshold_pct=99, window=None), examples=2)
    assert type(layout).__name__ == "Layout"


def test_play_window_guarded_on_headless_build(synthetic_video):
    s = ez.Session(dpath=str(synthetic_video.parent), file=synthetic_video.name)
    if not opencv_is_headless():
        pytest.skip("GUI OpenCV build available; guard does not apply")
    with pytest.raises(RuntimeError, match="headless"):
        play_window(s, ez.PlayParams(), None)
