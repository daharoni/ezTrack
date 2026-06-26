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


def test_spatial_downsample_is_speed_only_not_a_coordinate_change(synthetic_video):
    """spatial_downsample must not rescale outputs: x/y stay in original pixels."""
    params = ez.TrackParams(threshold_pct=99, window=None)

    s_full = _session(synthetic_video)
    ez.reference_frame(s_full, num_frames=20)
    df_full = ez.track(s_full, params, progress=False)

    s_half = _session(synthetic_video, spatial_downsample=2)  # 2x reduction
    ez.reference_frame(s_half, num_frames=20)
    df_half = ez.track(s_half, params, progress=False)

    # The reference is kept full-resolution regardless of the downsample factor.
    assert s_half.reference.shape == s_full.reference.shape

    # Downsampled run still recovers the ground-truth path in *original* pixels.
    truth = syn_truth_x()[: len(df_half)]
    det = df_half["detected"].to_numpy()
    assert det.mean() > 0.7
    assert np.abs(df_half["x"].to_numpy()[det] - truth[det]).mean() < 4.0
    assert df_half["y"].to_numpy()[det] == pytest.approx(SYN_Y, abs=2.0)

    # Full- and half-res tracks agree in the same space (not scaled by the factor).
    common = min(len(df_full), len(df_half))
    xf, xh = df_full["x"].to_numpy()[:common], df_half["x"].to_numpy()[:common]
    assert np.abs(xh - xf).mean() < 4.0  # same coordinate space
    assert np.abs(xh - xf / 2).mean() > 10.0  # would be ~0 if half-res leaked into output


def test_temporal_downsample_emits_only_tracked_frames(synthetic_video):
    """temporal_downsample outputs the real tracked frames -- no invented rows."""
    params = ez.TrackParams(threshold_pct=99, window=None)

    s_full = _session(synthetic_video)
    ez.reference_frame(s_full, num_frames=20)
    df_full = ez.track(s_full, params, progress=False)

    s_temp = _session(synthetic_video, temporal_downsample=4)  # every 4th frame
    ez.reference_frame(s_temp, num_frames=20)
    df_temp = ez.track(s_temp, params, progress=False)

    # Roughly a quarter of the rows, not the full count (positions aren't fabricated).
    assert len(df_temp) == pytest.approx(len(df_full) / 4, abs=1)
    # The frame column carries the real (strided) frame numbers.
    assert list(df_temp["frame"]) == list(range(0, len(df_full), 4))

    # Each emitted position is the genuine tracked location for that frame.
    truth = syn_truth_x()
    err = [abs(x - truth[f]) for f, x in zip(df_temp["frame"], df_temp["x"], strict=True)]
    assert np.mean(err) < 4.0


def test_hampel_filter_removes_position_outlier(synthetic_video):
    """hampel_filter (post-tracking) pulls a spike back and recomputes distance/ROIs."""
    params = ez.TrackParams(threshold_pct=99, window=None)
    s = _session(synthetic_video)
    ez.reference_frame(s, num_frames=20)

    clean = ez.track(s, params, progress=False)
    spiked = clean.copy()
    mid = len(spiked) // 2
    spiked.loc[mid, "x"] += 50  # inject an outlier jump

    fixed = ez.hampel_filter(spiked, s, window=3, sigma=3.0)
    # the spike is corrected back near the original trajectory...
    assert abs(fixed["x"].iloc[mid] - clean["x"].iloc[mid]) < 5.0
    # ...while non-outlier points are left essentially untouched...
    keep = [i for i in range(len(fixed)) if i != mid]
    assert np.allclose(fixed["x"].to_numpy()[keep], spiked["x"].to_numpy()[keep], atol=1e-9)
    # ...and the input dataframe is not mutated.
    assert spiked["x"].iloc[mid] == clean["x"].iloc[mid] + 50
    # distance_px is recomputed to stay consistent with the cleaned positions.
    expected = np.hypot(np.diff(fixed["x"].to_numpy()), np.diff(fixed["y"].to_numpy()))
    assert np.allclose(fixed["distance_px"].to_numpy()[1:], expected)


def test_hampel_filter_is_joint_in_xy(synthetic_video):
    """An outlier flagged by its (x, y) jump has BOTH coordinates replaced together."""
    s = _session(synthetic_video)
    ez.reference_frame(s, num_frames=20)
    clean = ez.track(s, ez.TrackParams(threshold_pct=99, window=None), progress=False)

    spiked = clean.copy()
    mid = len(spiked) // 2
    spiked.loc[mid, "x"] += 40  # the jump is only in x ...
    spiked.loc[mid, "y"] -= 25  # ... but y is also off at that same frame

    fixed = ez.hampel_filter(spiked, s, window=3, sigma=3.0)
    # both coordinates of the flagged frame snap back toward the local path,
    # even though y on its own might not have looked like an outlier.
    assert abs(fixed["x"].iloc[mid] - clean["x"].iloc[mid]) < 5.0
    assert abs(fixed["y"].iloc[mid] - clean["y"].iloc[mid]) < 5.0


def test_absolute_threshold_tracks_trajectory(synthetic_video):
    """An absolute difference cutoff tracks the known path just like the percentile."""
    s = _session(synthetic_video)
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_abs=30, window=None), progress=False)
    assert df.attrs["threshold_abs"] == 30
    assert df["detected"].mean() > 0.8
    assert df["x"].iloc[-1] > df["x"].iloc[0] + 50  # recovers the rightward motion


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


def test_save_tracking_round_trips_data_and_metadata(synthetic_video, tmp_path):
    """save_tracking + load_tracking must restore both the per-frame data and the
    df.attrs metadata that a bare to_csv/read_csv would drop."""
    s = _session(synthetic_video)
    s.selections.rois = ez.ROIs(names=["left"], polygons=[[[0, 0], [60, 0], [60, 80], [0, 80]]])
    s.selections.scale = ez.Scale(px_distance=10.0, real_distance=20.0, unit="cm")
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_abs=30, method="light", window=None), progress=False)

    csv_path = tmp_path / "vid_tracking.csv"
    meta_path = ez.save_tracking(df, str(csv_path))
    assert meta_path.endswith(".json") and (tmp_path / "vid_tracking.json").exists()

    loaded = ez.load_tracking(str(csv_path))
    pd.testing.assert_frame_equal(loaded, df, check_dtype=False, check_exact=False, atol=1e-6)
    # Metadata that to_csv alone would have lost survives the round trip.
    assert loaded.attrs["threshold_abs"] == 30
    assert loaded.attrs["method"] == "light"
    assert loaded.attrs["roi_coordinates"]["names"] == ["left"]


def test_summarize_reports_detection_rate(synthetic_video):
    s = _session(synthetic_video)
    ez.reference_frame(s, num_frames=20)
    df = ez.track(s, ez.TrackParams(threshold_pct=99, window=None), progress=False)

    whole = ez.summarize(df, s).iloc[0]
    assert whole["n_frames"] == len(df)
    assert whole["n_detected"] + whole["n_failed"] == len(df)
    assert 0.0 <= whole["pct_detected"] <= 100.0

    rate = ez.detection_rate(df)
    assert rate["n_detected"] == int(df["detected"].sum())
