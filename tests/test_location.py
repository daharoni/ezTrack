"""Headless smoke tests for the LocationTracking pipeline (typed-config API).

These exercise the code paths most likely to break on a modern stack
(scipy ``center_of_mass``, ``cv2.resize`` interpolation, pandas-3 ROI labelling
and transitions, the viz/summary builders) without requiring interactive
HoloViews rendering.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eztrack.location as lt
from eztrack import Session, TrackingParams

PRACTICE_VIDEO = (
    Path(__file__).resolve().parent.parent / "PracticeVideos" / "LocationTracking_Clip.mp4"
)

PARAMS = TrackingParams(loc_thresh=99, use_window=False, method="abs", rmv_wire=False)


def _session(end):
    s = Session(dpath=str(PRACTICE_VIDEO.parent), file=PRACTICE_VIDEO.name, start=0, end=end)
    s.fpath = os.path.join(s.dpath, s.file)
    return s


@pytest.mark.skipif(not PRACTICE_VIDEO.exists(), reason="practice video missing")
def test_track_location_short_clip():
    """make_reference + track_location produce a sane frame-by-frame dataframe."""
    s = _session(end=150)
    lt.make_reference(s, num_frames=20)
    assert s.reference is not None and s.reference.ndim == 2

    df = lt.track_location(s, PARAMS)

    assert isinstance(df, pd.DataFrame)
    for col in ("Frame", "X", "Y", "Distance_px"):
        assert col in df.columns
    assert 0 < len(df) <= 150
    h, w = s.reference.shape
    assert df["X"].between(0, w).all()
    assert df["Y"].between(0, h).all()
    assert df["Distance_px"].iloc[0] == 0
    assert (df["Distance_px"] >= 0).all()


def test_roi_linearize_multi_region():
    """roi_linearize labels frames, joining names when an animal is in >1 ROI."""
    rois = pd.DataFrame({"left": [True, False, False, True], "right": [False, True, False, True]})
    assert list(lt.roi_linearize(rois)) == ["left", "right", "non_roi", "left_right"]


def test_roi_transitions():
    """roi_transitions flags frames where the region label changes."""
    regions = pd.Series(["a", "a", "b", "b", "a"])
    assert list(lt.roi_transitions(regions)) == [False, False, True, False, True]
    assert lt.roi_transitions(regions, include_first=True).iloc[0]


def test_roi_location_membership():
    """roi_location marks per-frame membership using an int32 polygon mask."""

    class _Stream:
        data = {"xs": [[1, 1, 8, 8]], "ys": [[1, 8, 8, 1]]}

    s = Session(dpath=".", region_names=["box"])
    s.reference = np.zeros((10, 10))
    s.roi_stream = _Stream()
    location = pd.DataFrame({"Frame": [0, 1], "X": [5, 9], "Y": [5, 9]})
    out = lt.roi_location(s, location)
    assert out["box"].tolist() == [True, False]
    assert "ROI_coordinates" in out.columns


def test_heatmap_and_show_trace():
    """heatmap and show_trace build HoloViews elements from a tracked dataframe."""
    s = Session(dpath=".")
    s.reference = np.zeros((20, 20))
    df = pd.DataFrame({"Frame": range(5), "X": [1, 2, 3, 4, 5], "Y": [5, 4, 3, 2, 1]})
    assert type(lt.heatmap(s, df)).__name__ == "Image"
    assert type(lt.show_trace(s, df)).__name__ == "Overlay"


def test_summarize_location_whole_session():
    """summarize_location collapses a tracked dataframe to one row when bin_dict is None."""
    s = Session(dpath=".", region_names=None)
    df = pd.DataFrame(
        {
            "File": "clip.mp4",
            "Frame": range(10),
            "X": np.arange(10.0),
            "Y": np.arange(10.0),
            "Distance_px": np.ones(10),
        }
    )
    out = lt.summarize_location(df, s, bin_dict=None)
    assert len(out) == 1
    assert out["Distance_px"].iloc[0] == df["Distance_px"].sum()


def test_session_from_dict_roundtrip():
    """Session.from_dict accepts a legacy video_dict (nested dicts converted)."""
    s = Session.from_dict(
        {
            "dpath": "/data",
            "file": "v.mp4",
            "start": 5,
            "end": 100,
            "region_names": ["a", "b"],
            "dsmpl": 0.5,
            "stretch": {"width": 2, "height": 1},
            "scale": {"px_distance": 10, "true_distance": 100, "true_scale": "cm"},
            "FileNames": ["ignored unknown key"],  # unknown keys are dropped
        }
    )
    assert s.dpath == "/data" and s.start == 5 and s.dsmpl == 0.5
    assert s.stretch.width == 2 and s.stretch.height == 1
    assert s.scale.true_scale == "cm" and s.scale.px_distance == 10


def test_play_video_ext_guard_on_headless_build():
    """play_video_ext fails fast with a friendly error on a headless OpenCV build."""
    from eztrack.playback import _opencv_is_headless

    if not _opencv_is_headless():
        pytest.skip("GUI OpenCV build available; guard does not apply")
    with pytest.raises(RuntimeError, match="headless"):
        lt.play_video_ext(Session(dpath="."), lt.DisplayParams(), None)


@pytest.mark.slow
@pytest.mark.skipif(not PRACTICE_VIDEO.exists(), reason="practice video missing")
def test_track_location_full_clip():
    """Full-length run over the practice clip completes and stays in-bounds."""
    s = _session(end=None)
    lt.make_reference(s, num_frames=50)
    df = lt.track_location(s, PARAMS)
    h, w = s.reference.shape
    assert len(df) > 200
    assert df["X"].between(0, w).all()
    assert df["Y"].between(0, h).all()
