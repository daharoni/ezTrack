"""Tests for the per-frame locate() math, using synthetic frames with known answers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import draw_square

from eztrack import TrackParams, Window
from eztrack.tracking import locate


def _frame_with_blob(cx, cy, value=255, shape=(80, 120), side=12):
    """Black frame with a bright square centered at (cx, cy)."""
    return draw_square(np.zeros(shape, dtype=np.uint8), cx, cy, side, value)


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


def test_denoise_drops_speck_keeps_animal():
    """A 1px speck should be removed by opening while the animal blob survives."""
    ref = np.zeros((80, 120), dtype=np.uint8)
    frame = _frame_with_blob(70, 40, side=12)  # the "animal"
    frame[10, 10] = 255  # a lone speck far from the animal

    params = TrackParams(threshold_pct=99, window=None, denoise=True, denoise_kernel=3)
    loc = locate(frame, ref, params)
    assert loc.detected
    assert loc.x == pytest.approx(70, abs=1.0)  # speck gone, COM stays on the animal
    assert loc.y == pytest.approx(40, abs=1.0)


def test_denoise_fails_frame_when_it_would_erase_everything():
    """A kernel larger than the animal erases the whole mask -> the frame is reported
    not-detected, rather than reverting to the un-opened (noisy) mask."""
    ref = np.zeros((80, 120), dtype=np.uint8)
    frame = _frame_with_blob(70, 40, side=4)  # small blob, smaller than the kernel
    params = TrackParams(threshold_pct=99, window=None, denoise=True, denoise_kernel=15)
    loc = locate(frame, ref, params)
    assert not loc.detected  # over-aggressive denoise drops the frame, no specks resurrected
    assert loc.dif.max() == 0  # mask left empty, not reverted to the original


def test_denoise_is_monotonic_in_kernel():
    """Increasing the kernel must never resurrect specks: a kernel past the animal's
    size empties the mask instead of falling back to the noisy one."""
    ref = np.zeros((80, 120), dtype=np.uint8)
    frame = _frame_with_blob(70, 40, side=7)  # 7px animal
    frame[10, 10] = 255  # a lone speck
    params = TrackParams(threshold_pct=99, window=None, denoise=True)

    kept = locate(frame, ref, replace(params, denoise_kernel=3))
    erased = locate(frame, ref, replace(params, denoise_kernel=11))
    assert kept.detected and kept.x == pytest.approx(70, abs=1.0)  # speck removed, animal kept
    assert not erased.detected  # bigger kernel empties the mask, never brings the speck back


def test_threshold_abs_keeps_only_changes_above_cutoff():
    """An absolute cutoff (not a percentile) drops sub-cutoff changes regardless of
    how the rest of the frame looks."""
    ref = np.zeros((80, 120), dtype=np.uint8)
    frame = ref.copy()
    frame[34:46, 64:76] = 40  # a +40 blob (the animal) at (70, 40)
    frame[34:46, 14:26] = 10  # a faint +10 blob elsewhere

    # cutoff between the two blobs: only the strong one survives -> COM on it.
    loc = locate(frame, ref, TrackParams(method="light", threshold_abs=25, window=None))
    assert loc.detected
    assert loc.x == pytest.approx(70, abs=1.0)
    assert loc.y == pytest.approx(40, abs=1.0)

    # raise the cutoff above both -> nothing survives -> not detected.
    none = locate(frame, ref, TrackParams(method="light", threshold_abs=60, window=None))
    assert not none.detected


def test_threshold_abs_overrides_percentile():
    """When threshold_abs is set, threshold_pct is ignored entirely."""
    ref = np.zeros((40, 40), dtype=np.uint8)
    frame = ref.copy()
    frame[18:22, 18:22] = 30  # a single +30 blob

    # A percentile of 0 would normally keep almost everything; the absolute cutoff
    # of 50 wins and erases the only blob -> not detected.
    loc = locate(
        frame, ref, TrackParams(method="abs", threshold_pct=0, threshold_abs=50, window=None)
    )
    assert not loc.detected


def test_threshold_abs_validated():
    with pytest.raises(ValueError, match="threshold_abs"):
        TrackParams(threshold_abs=-1)


def test_threshold_on_raw_uses_pixel_value_not_baseline():
    """Raw mode finds a dark animal over a *dark* baseline that difference-mode misses."""
    # Baseline is already dark where the animal goes, so frame-vs-baseline is tiny...
    ref = np.full((80, 120), 30, dtype=np.uint8)
    frame = ref.copy()
    frame[34:46, 64:76] = 5  # animal at (70, 40): only 25 darker than the baseline

    # ...difference mode with a cutoff of 40 sees nothing (25 < 40).
    diff = locate(frame, ref, TrackParams(method="dark", threshold_abs=40, window=None))
    assert not diff.detected

    # ...raw mode keeps pixels darker than value 15, so the value-5 animal survives.
    raw = locate(
        frame, ref, TrackParams(method="dark", threshold_abs=15, threshold_on="raw", window=None)
    )
    assert raw.detected
    assert raw.x == pytest.approx(70, abs=1.0)
    assert raw.y == pytest.approx(40, abs=1.0)


def test_threshold_on_validated():
    with pytest.raises(ValueError, match="threshold_on"):
        TrackParams(threshold_on="bogus")
    # raw needs a direction -- 'abs' has no baseline to deviate from
    with pytest.raises(ValueError, match="raw"):
        TrackParams(threshold_on="raw", method="abs")


def test_denoise_kernel_validated():
    with pytest.raises(ValueError, match="denoise_kernel"):
        TrackParams(denoise=True, denoise_kernel=0)


def test_center_of_mass_matches_scipy():
    """The fast nonzero center-of-mass must agree with scipy on a thresholded array."""
    from scipy import ndimage

    from eztrack.tracking import _center_of_mass

    arr = np.zeros((40, 60), dtype=np.float32)
    arr[10:14, 20:26] = 7  # one blob
    arr[30:33, 50:54] = 3  # another, different weight
    cy, cx = ndimage.center_of_mass(arr)
    fx, fy = _center_of_mass(arr)
    assert fx == pytest.approx(cx, abs=1e-6)
    assert fy == pytest.approx(cy, abs=1e-6)
    assert _center_of_mass(np.zeros((5, 5))) is None  # nothing survives -> None
