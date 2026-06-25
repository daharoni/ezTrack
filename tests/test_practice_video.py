"""Golden tests on the bundled practice clip with programmatic selections.

These exercise the real video decode + reference + tracking path. Selections are
set programmatically (not via widgets), which is exactly what makes the
interactive pipeline testable headlessly.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import PRACTICE_VIDEO

import eztrack as ez

pytestmark = pytest.mark.skipif(not PRACTICE_VIDEO.exists(), reason="practice video missing")


def _session(end):
    return ez.Session(
        dpath=str(PRACTICE_VIDEO.parent),
        file=PRACTICE_VIDEO.name,
        end=end,
        region_names=["left", "right"],
        selections=ez.Selections(
            crop=ez.Crop(0, 0, 320, 240),
            rois=ez.ROIs(
                names=["left", "right"],
                polygons=[
                    [[0, 0], [160, 0], [160, 240], [0, 240]],
                    [[160, 0], [320, 0], [320, 240], [160, 240]],
                ],
            ),
        ),
    )


def test_practice_clip_tracks_and_is_in_bounds():
    s = _session(end=150)
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(method="abs"), progress=False)

    assert isinstance(df, pd.DataFrame)
    for col in ("frame", "x", "y", "detected", "distance_px", "left", "right", "roi"):
        assert col in df.columns
    assert "ROI_coordinates" not in df.columns  # geometry lives in attrs, not every row
    assert 0 < len(df) <= 150
    h, w = s.reference.shape
    assert df["x"].between(0, w).all()
    assert df["y"].between(0, h).all()
    assert df["distance_px"].iloc[0] == 0
    assert df.attrs["method"] == "abs"


def test_practice_clip_tracking_is_deterministic():
    """Same selections + params -> identical track (a real regression net)."""
    frames = []
    for _ in range(2):
        s = _session(end=120)
        ez.reference_frame(s, num_frames=20, frames=list(range(0, 100, 5)))
        frames.append(ez.track(s, ez.TrackParams(window=None), progress=False))
    pd.testing.assert_frame_equal(frames[0], frames[1])


@pytest.mark.slow
def test_practice_clip_full_run():
    s = _session(end=None)
    ez.reference_frame(s, num_frames=50)
    df = ez.track(s, ez.TrackParams(), progress=False)
    assert len(df) > 200
    h, w = s.reference.shape
    assert df["x"].between(0, w).all()
    assert df["y"].between(0, h).all()
