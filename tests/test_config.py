"""Unit tests for the plain config/selection objects (no OpenCV/HoloViews)."""

from __future__ import annotations

import numpy as np
import pytest

from eztrack import Crop, Mask, ROIs, Scale, Selections, TrackParams, Window


def test_crop_bounds_orders_corners():
    # corners given in the "wrong" order still produce a valid box
    crop = Crop(x0=80, y0=60, x1=10, y1=5)
    assert crop.bounds() == (10, 5, 80, 60)


def test_crop_apply_slices_frame():
    frame = np.arange(100).reshape(10, 10)
    out = Crop(x0=2, y0=1, x1=5, y1=4).apply(frame)
    assert out.shape == (3, 3)
    assert np.array_equal(out, frame[1:4, 2:5])


def test_crop_from_boxedit_empty_is_none():
    assert Crop.from_boxedit(None) is None
    assert Crop.from_boxedit({"x0": [], "x1": [], "y0": [], "y1": []}) is None


def test_crop_from_boxedit_takes_first_box():
    crop = Crop.from_boxedit({"x0": [1.5], "y0": [2.5], "x1": [9.0], "y1": [8.0]})
    assert (crop.x0, crop.y0, crop.x1, crop.y1) == (1.5, 2.5, 9.0, 8.0)


def test_polydraw_conversion_for_mask_and_rois():
    data = {"xs": [[0, 10, 10, 0]], "ys": [[0, 0, 10, 10]]}
    mask = Mask.from_polydraw(data)
    assert mask.polygons == [[[0, 0], [10, 0], [10, 10], [0, 10]]]

    rois = ROIs.from_polydraw(["a", "b"], data)  # only one polygon drawn
    assert rois.names == ["a"]  # names matched positionally to drawn polys
    assert len(rois.polygons) == 1


def test_polydraw_rois_default_names_for_undeclared_polygons():
    # three polygons drawn, no names declared -> all get zone_N placeholders
    data = {"xs": [[0, 1, 1]] * 3, "ys": [[0, 0, 1]] * 3}
    rois = ROIs.from_polydraw(None, data)
    assert rois.names == ["zone_1", "zone_2", "zone_3"]

    # partial names -> declared ones kept, the rest filled in positionally
    rois = ROIs.from_polydraw(["left"], data)
    assert rois.names == ["left", "zone_2", "zone_3"]


def test_scale_factor():
    assert Scale().factor is None
    assert Scale(px_distance=10, real_distance=50, unit="cm").factor == 5.0


@pytest.mark.parametrize("bad", [{"method": "sideways"}, {"threshold_pct": 150}])
def test_trackparams_validation_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        TrackParams(**bad)


def test_trackparams_rejects_bad_window_weight():
    with pytest.raises(ValueError):
        TrackParams(window=Window(size=50, weight=2.0))


def test_session_downsample_factor_validation():
    from eztrack import Session

    # factors are reductions >= 1 (2 = 2x), so sub-1 spatial and non-int temporal are errors
    with pytest.raises(ValueError, match="spatial_downsample"):
        Session(dpath=".", spatial_downsample=0.5)
    with pytest.raises(ValueError, match="temporal_downsample"):
        Session(dpath=".", temporal_downsample=0)
    with pytest.raises(ValueError, match="temporal_downsample"):
        Session(dpath=".", temporal_downsample=2.5)
    # valid factors are accepted (temporal coerced to int)
    assert Session(dpath=".", spatial_downsample=2, temporal_downsample=3).temporal_downsample == 3


def test_selections_roundtrip_in_memory():
    sel = Selections(
        crop=Crop(0, 0, 50, 40),
        mask=Mask(polygons=[[[1, 1], [2, 2], [3, 1]]]),
        rois=ROIs(names=["left"], polygons=[[[0, 0], [10, 0], [10, 10]]]),
        scale=Scale(px_distance=12.0, real_distance=24.0, unit="cm"),
    )
    restored = Selections.from_dict(sel.to_dict())
    assert restored == sel


def test_selections_save_load_roundtrip(tmp_path):
    sel = Selections(crop=Crop(1, 2, 3, 4), rois=ROIs(names=["r"], polygons=[[[0, 0], [1, 1]]]))
    path = tmp_path / "sel.json"
    sel.save(str(path))
    assert Selections.load(str(path)) == sel


def test_empty_selections_roundtrip(tmp_path):
    path = tmp_path / "empty.json"
    Selections().save(str(path))
    assert Selections.load(str(path)) == Selections()
