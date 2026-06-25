"""End-to-end pipeline tests on the synthetic ground-truth video."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SYN_N, SYN_Y, syn_truth_x

import eztrack as ez


def _session(video_path, **kw):
    return ez.Session(dpath=str(video_path.parent), file=video_path.name, **kw)


def _track(video_path, **kw):
    s = _session(video_path, **kw)
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_pct=99, window=None), progress=False)
    return s, df


def test_track_recovers_known_trajectory(synthetic_video):
    """The recovered path must follow the square's true left-to-right motion."""
    _s, df = _track(synthetic_video)
    assert len(df) >= SYN_N - 1
    assert df["detected"].mean() > 0.8

    det = df[df["detected"]]
    truth = syn_truth_x()[: len(df)][df["detected"].to_numpy()]
    assert np.abs(det["x"].to_numpy() - truth).mean() < 3.0  # tracks the truth
    assert det["y"].to_numpy() == pytest.approx(SYN_Y, abs=2.0)
    assert df["x"].iloc[-1] > df["x"].iloc[0] + 50  # moved rightward


def test_track_requires_reference(synthetic_video):
    s = _session(synthetic_video)
    with pytest.raises(ValueError, match="reference"):
        ez.track(s, ez.TrackParams(), progress=False)


def test_roi_membership_over_session(synthetic_video):
    s = _session(synthetic_video)
    s.region_names = ["left"]
    s.selections.rois = ez.ROIs(names=["left"], polygons=[[[0, 0], [60, 0], [60, 80], [0, 80]]])
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_pct=99, window=None), progress=False)

    assert "left" in df.columns and "roi" in df.columns
    assert bool(df["left"].iloc[0]) is True  # starts at x=20 (left)
    assert bool(df["left"].iloc[-1]) is False  # ends at x=100 (right)


def test_scale_adds_real_world_distance(synthetic_video):
    s = _session(synthetic_video)
    s.selections.scale = ez.Scale(px_distance=10.0, real_distance=20.0, unit="cm")
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_pct=99, window=None), progress=False)
    assert "distance_cm" in df.columns
    np.testing.assert_allclose(df["distance_cm"], df["distance_px"] * 2.0)


def test_saved_selections_replay_identically(synthetic_video, tmp_path):
    """Selections via object vs. saved-then-loaded JSON must give identical tracks."""
    sel = ez.Selections(
        crop=ez.Crop(10, 10, 110, 70),
        rois=ez.ROIs(names=["left"], polygons=[[[0, 0], [50, 0], [50, 60], [0, 60]]]),
    )

    s1 = _session(synthetic_video)
    s1.selections = sel
    ez.reference_frame(s1, num_frames=20)
    df1 = ez.track(s1, ez.TrackParams(threshold_pct=99, window=None), progress=False)

    path = tmp_path / "sel.json"
    sel.save(str(path))
    s2 = _session(synthetic_video)
    s2.selections = ez.Selections.load(str(path))
    ez.reference_frame(s2, num_frames=20)
    df2 = ez.track(s2, ez.TrackParams(threshold_pct=99, window=None), progress=False)

    pd.testing.assert_frame_equal(df1, df2)


def test_summarize_whole_session_and_bins(synthetic_video):
    s = _session(synthetic_video)
    s.region_names = ["left"]
    s.selections.rois = ez.ROIs(names=["left"], polygons=[[[0, 0], [60, 0], [60, 80], [0, 80]]])
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_pct=99, window=None), progress=False)

    whole = ez.summarize(df, s)
    assert len(whole) == 1
    assert whole["distance_px"].iloc[0] > 0
    assert 0.0 <= whole["prop_left"].iloc[0] <= 1.0

    mid = int(df["frame"].median())
    binned = ez.summarize(df, s, bins={"early": (0, mid), "late": (mid + 1, 999)})
    assert list(binned["bin"]) == ["early", "late"]
    assert binned["prop_left"].iloc[0] > binned["prop_left"].iloc[1]  # left early, right late
