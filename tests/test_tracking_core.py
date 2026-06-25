"""Tests for the per-frame locate() math, using synthetic frames with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from eztrack import TrackParams, Window
from eztrack.tracking import locate


def _frame_with_blob(cx, cy, value=255, shape=(80, 120), side=12):
    """Black frame with a bright square centered at (cx, cy)."""
    frame = np.zeros(shape, dtype=np.uint8)
    half = side // 2
    frame[cy - half : cy + half, cx - half : cx + half] = value
    return frame


def test_locate_finds_blob_center():
    ref = np.zeros((80, 120), dtype=np.uint8)
    frame = _frame_with_blob(70, 40)
    loc = locate(frame, ref, TrackParams(threshold_pct=99, window=None))
    assert loc.detected
    assert loc.x == pytest.approx(70, abs=1.0)
    assert loc.y == pytest.approx(40, abs=1.0)


def test_locate_light_vs_dark_methods():
    # animal darker than background: bright reference, dark blob
    ref = np.full((80, 120), 200, dtype=np.uint8)
    frame = np.full((80, 120), 200, dtype=np.uint8)
    frame[34:46, 64:76] = 0  # dark square at (70, 40)

    dark = locate(frame, ref, TrackParams(method="dark", threshold_pct=99, window=None))
    assert dark.detected and dark.x == pytest.approx(70, abs=1.0)

    # "light" should find nothing here (animal is darker, not lighter)
    light = locate(frame, ref, TrackParams(method="light", threshold_pct=99, window=None))
    assert not light.detected


def test_locate_reports_not_detected_when_frame_matches_reference():
    ref = np.full((80, 120), 50, dtype=np.uint8)
    loc = locate(ref.copy(), ref, TrackParams(window=None), prior=(11.0, 22.0))
    assert not loc.detected
    assert (loc.x, loc.y) == (11.0, 22.0)  # prior reused, no NaN


def test_locate_first_frame_miss_falls_back_to_center():
    ref = np.full((80, 120), 50, dtype=np.uint8)
    loc = locate(ref.copy(), ref, TrackParams(window=None))
    assert not loc.detected
    assert (loc.x, loc.y) == (60.0, 40.0)  # frame center


def test_mask_excludes_region():
    ref = np.zeros((80, 120), dtype=np.uint8)
    frame = _frame_with_blob(70, 40)
    mask = np.zeros((80, 120), dtype=bool)
    mask[30:50, 60:80] = True  # cover the blob
    loc = locate(frame, ref, TrackParams(threshold_pct=99, window=None), mask=mask)
    assert not loc.detected


def test_window_biases_toward_prior_blob():
    """With two blobs, windowing should keep the COM near the prior, not midway."""
    ref = np.zeros((80, 120), dtype=np.uint8)
    frame = _frame_with_blob(20, 40) | _frame_with_blob(100, 40)
    params = TrackParams(threshold_pct=99, window=Window(size=30, weight=0.95))
    loc = locate(frame, ref, params, prior=(20.0, 40.0))
    assert loc.x == pytest.approx(20, abs=3.0)  # not pulled to ~60 (midpoint)
