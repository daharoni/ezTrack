"""HoloViews visualizations: shared image builder, threshold preview, trace, heatmap.

Every plot goes through :func:`image` so the gray / y-inverted / stretched options
live in exactly one place.
"""

from __future__ import annotations

import cv2
import holoviews as hv
import numpy as np
import pandas as pd

from .config import Session, Stretch, TrackParams
from .tracking import locate
from .video import open_capture, preprocess

hv.extension("bokeh")

__all__ = ["image", "threshold_preview", "trace", "heatmap"]


def image(arr: np.ndarray, stretch: Stretch | None = None, title: str = "", **opts) -> hv.Image:
    """Build a stretched, gray, y-inverted HoloViews Image of a 2-D array."""
    stretch = stretch or Stretch()
    img = hv.Image((np.arange(arr.shape[1]), np.arange(arr.shape[0]), arr))
    return img.opts(
        width=int(arr.shape[1] * stretch.width),
        height=int(arr.shape[0] * stretch.height),
        invert_yaxis=True,
        cmap="gray",
        toolbar="below",
        title=title,
        **opts,
    )


def threshold_preview(session: Session, params: TrackParams, examples: int = 4) -> hv.Layout:
    """Show tracking on a few random frames to sanity-check ``params``.

    Each example pairs the original frame with the thresholded difference, both
    marked with the detected center of mass. Windowing is not applied (frames
    are examined in isolation).
    """
    if session.reference is None:
        raise ValueError("No reference frame. Call eztrack.reference_frame(session) first.")
    cap = open_capture(session.fpath)
    try:
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = int(session.end) if session.end is not None else cap_max

        panels = []
        for _ in range(examples):
            loc = None
            while loc is None or not loc.detected:
                frm = int(np.random.randint(session.start, end))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
                ret, frame = cap.read()
                if not ret:
                    continue
                loc = locate(preprocess(frame, session), session.reference, params)
            marker = hv.Points(([loc.x], [loc.y])).opts(
                color="red", size=20, marker="+", line_width=3
            )
            orig = image(preprocess(frame, session), session.stretch, f"Frame {frm}") * marker
            heat = image(loc.dif, session.stretch, f"Frame {frm}").opts(cmap="jet") * marker
            panels.extend([orig, heat])
    finally:
        cap.release()
    return hv.Layout(panels)


def trace(session: Session, df: pd.DataFrame, color: str = "red", alpha: float = 0.8) -> hv.Element:
    """Overlay the traced path (and any ROI outlines) on the reference frame."""
    img = image(session.reference, session.stretch, "Motion Trace")
    rois = session.selections.rois
    if rois:
        polys = hv.Polygons([[(v[0], v[1]) for v in poly] for poly in rois.polygons])
        img = img * polys.opts(fill_alpha=0.1, line_dash="dashed")
    points = hv.Scatter((df["x"], df["y"])).opts(color=color, alpha=alpha, size=3)
    return img * points


def heatmap(session: Session, df: pd.DataFrame, sigma: float | None = None) -> hv.Image:
    """Gaussian-smoothed occupancy heatmap of the animal's location."""
    h, w = session.reference.shape
    yi = np.clip(df["y"].to_numpy().astype(int), 0, h - 1)
    xi = np.clip(df["x"].to_numpy().astype(int), 0, w - 1)
    grid = np.zeros((h, w))
    np.add.at(grid, (yi, xi), 1)  # vectorized occupancy count

    sigma = float(np.mean(grid.shape) * 0.05) if sigma is None else sigma
    grid = cv2.GaussianBlur(grid, (0, 0), sigma)
    if grid.max() > 0:
        grid = grid / grid.max() * 255
    return image(grid, session.stretch, "Heatmap").opts(cmap="jet")
