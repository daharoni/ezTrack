"""Unit tests for the pure geometry module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eztrack import Mask, ROIs
from eztrack.regions import linearize, mask_array, rasterize, roi_membership, transitions


def test_rasterize_square():
    poly = [[2, 2], [2, 6], [6, 6], [6, 2]]
    inside = rasterize(poly, (10, 10))
    assert inside.dtype == bool
    assert inside[4, 4]  # center is inside
    assert not inside[0, 0]  # corner is outside


def test_mask_array_unions_polygons_and_handles_empty():
    assert mask_array(Mask(), (10, 10)) is None
    assert mask_array(None, (10, 10)) is None
    mask = Mask(polygons=[[[0, 0], [0, 3], [3, 3], [3, 0]], [[7, 7], [7, 9], [9, 9], [9, 7]]])
    arr = mask_array(mask, (10, 10))
    assert arr[1, 1] and arr[8, 8] and not arr[5, 5]


def test_roi_membership_clips_out_of_bounds():
    rois = ROIs(names=["box"], polygons=[[[0, 0], [0, 5], [5, 5], [5, 0]]])
    # x=99 is out of bounds; must be clipped, not raise IndexError
    membership = roi_membership(rois, np.array([2, 99]), np.array([2, 99]), (10, 10))
    assert membership["box"].tolist() == [True, False]


def test_linearize_joins_and_labels_none():
    df = pd.DataFrame({"left": [True, False, False, True], "right": [False, True, False, True]})
    assert list(linearize(df)) == ["left", "right", "none", "left_right"]


def test_transitions_flags_changes():
    labels = pd.Series(["a", "a", "b", "b", "a"])
    assert list(transitions(labels)) == [False, False, True, False, True]
    assert transitions(labels, include_first=True).iloc[0]
