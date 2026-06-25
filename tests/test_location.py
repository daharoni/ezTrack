"""Headless smoke tests for the modernized LocationTracking pipeline.

These exercise the code paths most likely to break on a modern stack
(scipy ``center_of_mass``, ``cv2.resize`` interpolation, pandas-3 ROI labelling
and transition logic) without requiring any interactive HoloViews rendering.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eztrack.location as lt

PRACTICE_VIDEO = (
    Path(__file__).resolve().parent.parent / "PracticeVideos" / "LocationTracking_Clip.mp4"
)

TRACKING_PARAMS = {
    "loc_thresh": 99,
    "use_window": False,
    "window_size": 100,
    "window_weight": 0.9,
    "method": "abs",
    "rmv_wire": False,
    "wire_krn": 5,
}


def _video_dict(end):
    vd = {
        "dpath": str(PRACTICE_VIDEO.parent),
        "file": PRACTICE_VIDEO.name,
        "start": 0,
        "end": end,
        "region_names": None,
        "dsmpl": 1,
        "stretch": {"width": 1, "height": 1},
    }
    vd["fpath"] = os.path.join(vd["dpath"], vd["file"])
    return vd


@pytest.mark.skipif(not PRACTICE_VIDEO.exists(), reason="practice video missing")
def test_track_location_short_clip():
    """Reference + TrackLocation produce a sane frame-by-frame dataframe."""
    vd = _video_dict(end=150)
    reference, _img = lt.Reference(vd, num_frames=20)
    vd["reference"] = reference
    assert reference.ndim == 2 and reference.size > 0

    df = lt.TrackLocation(vd, TRACKING_PARAMS)

    assert isinstance(df, pd.DataFrame)
    for col in ("Frame", "X", "Y", "Distance_px"):
        assert col in df.columns
    assert 0 < len(df) <= 150
    h, w = reference.shape
    assert df["X"].between(0, w).all()
    assert df["Y"].between(0, h).all()
    # first frame has no prior frame, so distance is zero
    assert df["Distance_px"].iloc[0] == 0
    assert (df["Distance_px"] >= 0).all()


def test_roi_linearize_multi_region():
    """ROI_linearize labels frames, joining names when an animal is in >1 ROI."""
    rois = pd.DataFrame(
        {
            "left": [True, False, False, True],
            "right": [False, True, False, True],
        }
    )
    out = lt.ROI_linearize(rois)
    assert list(out) == ["left", "right", "non_roi", "left_right"]


def test_roi_transitions():
    """ROI_transitions flags frames where the region label changes."""
    regions = pd.Series(["a", "a", "b", "b", "a"])
    assert list(lt.ROI_transitions(regions)) == [False, False, True, False, True]
    assert lt.ROI_transitions(regions, include_first=True).iloc[0]


def test_roi_location_membership():
    """ROI_Location marks per-frame membership using an int32 polygon mask."""

    class _Stream:
        data = {"xs": [[1, 1, 8, 8]], "ys": [[1, 8, 8, 1]]}

    vd = {
        "region_names": ["box"],
        "roi_stream": _Stream(),
        "reference": np.zeros((10, 10)),
    }
    location = pd.DataFrame({"Frame": [0, 1], "X": [5, 9], "Y": [5, 9]})
    out = lt.ROI_Location(vd, location)
    assert out["box"].tolist() == [True, False]
    assert "ROI_coordinates" in out.columns


def test_playvideo_ext_guard_on_headless_build():
    """PlayVideo_ext fails fast with a friendly error on a headless OpenCV build."""
    if not lt._opencv_is_headless():
        pytest.skip("GUI OpenCV build available; guard does not apply")
    with pytest.raises(RuntimeError, match="headless"):
        lt.PlayVideo_ext({}, {}, None)


@pytest.mark.slow
@pytest.mark.skipif(not PRACTICE_VIDEO.exists(), reason="practice video missing")
def test_track_location_full_clip():
    """Full-length run over the practice clip completes and stays in-bounds."""
    vd = _video_dict(end=None)
    reference, _img = lt.Reference(vd, num_frames=50)
    vd["reference"] = reference
    df = lt.TrackLocation(vd, TRACKING_PARAMS)
    h, w = reference.shape
    assert len(df) > 200
    assert df["X"].between(0, w).all()
    assert df["Y"].between(0, h).all()
