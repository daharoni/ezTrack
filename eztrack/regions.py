"""Pure geometry: rasterize polygons, test ROI membership, label transitions.

No video or HoloViews here -- everything operates on numpy arrays and the plain
:class:`~eztrack.config.Mask`/:class:`~eztrack.config.ROIs` data objects, so it
is trivially unit-testable headlessly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .config import Mask, ROIs

__all__ = [
    "rasterize",
    "clip_to_indices",
    "mask_array",
    "roi_membership",
    "linearize",
    "transitions",
]


def clip_to_indices(
    xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Clip x/y coordinates to integer, in-bounds ``(xi, yi)`` pixel indices for ``shape``."""
    h, w = shape
    xi = np.clip(np.asarray(xs), 0, w - 1).astype(np.intp)
    yi = np.clip(np.asarray(ys), 0, h - 1).astype(np.intp)
    return xi, yi


def rasterize(polygon: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean array of ``shape`` that is True inside ``polygon``."""
    grid = np.zeros(shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)  # cv2 needs int32 contours
    cv2.fillPoly(grid, pts=[pts], color=1)
    return grid.astype(bool)


def mask_array(mask: Mask | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Boolean exclusion array (union of mask polygons), or ``None`` if empty."""
    if not mask:
        return None
    out = np.zeros(shape, dtype=bool)
    for polygon in mask.polygons:
        out |= rasterize(polygon, shape)
    return out


def roi_membership(
    rois: ROIs | None, xs: np.ndarray, ys: np.ndarray, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Map each ROI name to a boolean per-sample array of "inside that region".

    Coordinates are taken positionally (``ys``/``xs`` as integer pixel indices),
    so the result never depends on the dataframe index.
    """
    if not rois:
        return {}
    xi, yi = clip_to_indices(xs, ys, shape)
    membership = {}
    for name, polygon in zip(rois.names, rois.polygons, strict=False):
        inside = rasterize(polygon, shape)
        membership[name] = inside[yi, xi]
    return membership


def linearize(membership: pd.DataFrame, null_name: str = "none") -> pd.Series:
    """Collapse per-region boolean columns into one label per frame.

    Frames inside several regions get the names joined in column order
    (e.g. ``left_top``); frames in none get ``null_name``.
    """
    cols = list(membership.columns)
    rows = membership.to_numpy(dtype=bool)
    labels = [
        "_".join(c for c, inside in zip(cols, row, strict=True) if inside) or null_name
        for row in rows
    ]
    return pd.Series(labels, index=membership.index, dtype=object)


def transitions(labels: pd.Series, include_first: bool = False) -> pd.Series:
    """Boolean series flagging frames whose ROI label differs from the previous frame."""
    changed = labels != labels.shift(1, fill_value=labels.iloc[0])
    if include_first:
        changed.iloc[0] = True
    return changed
