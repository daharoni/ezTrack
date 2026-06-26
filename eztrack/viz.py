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
from .regions import clip_to_indices
from .tracking import locate
from .video import open_capture, preprocess

hv.extension("bokeh")

__all__ = ["image", "threshold_preview", "trace", "heatmap", "outlier_plot"]


def image(arr: np.ndarray, stretch: Stretch | None = None, title: str = "", **opts) -> hv.Image:
    """Build a stretched, gray, y-inverted HoloViews Image of a 2-D array.

    Defaults the display size to the array's pixel dimensions (scaled by
    ``stretch``); pass ``width``/``height`` to override -- e.g. to fit several
    panels of a high-resolution video side by side.
    """
    stretch = stretch or Stretch()
    img = hv.Image((np.arange(arr.shape[1]), np.arange(arr.shape[0]), arr))
    opts.setdefault("width", int(arr.shape[1] * stretch.width))
    opts.setdefault("height", int(arr.shape[0] * stretch.height))
    opts.setdefault("cmap", "gray")
    return img.opts(
        invert_yaxis=True,
        toolbar="below",
        title=title,
        **opts,
    )


def threshold_preview(
    session: Session, params: TrackParams, examples: int = 4, panel_width: int = 320
) -> hv.Layout:
    """Show tracking on a few random frames to sanity-check ``params``.

    Each example pairs the original frame with the thresholded difference. It
    prefers frames where the animal is detected, but if ``params`` detects nothing
    (e.g. an over-strict ``threshold_abs`` or the wrong ``method``) it stops after a
    bounded search and shows undetected frames -- titled "nothing above threshold"
    with no marker -- rather than looping forever. Windowing is not applied (frames
    are examined in isolation). Panels are laid out two-up (original | difference)
    and scaled to ``panel_width`` px so high-resolution videos don't overflow.
    """
    if session.reference is None:
        raise ValueError("No reference frame. Call eztrack.reference_frame(session) first.")

    # Scale each panel to a fixed display width, preserving aspect ratio, so a
    # 1280px-wide frame doesn't render two-across at native size.
    size = _display_size(session.reference.shape, panel_width)

    cap = open_capture(session.fpath)
    try:
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = int(session.end) if session.end is not None else cap_max

        panels = []
        for _ in range(examples):
            # Look for a detected frame, but cap the attempts so impossible params
            # can't hang the preview -- fall back to the last frame we could read.
            loc = frame = frm = None
            for _attempt in range(25):
                candidate_frm = int(np.random.randint(session.start, end))
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_frm)
                ret, candidate = cap.read()
                if not ret:
                    continue
                frm, frame = candidate_frm, candidate
                loc = locate(preprocess(frame, session), session.reference, params)
                if loc.detected:
                    break
            if frame is None:
                continue  # could not read any frame in this video range

            grey = preprocess(frame, session)
            title = f"Frame {frm}" if loc.detected else f"Frame {frm} — nothing above threshold"
            orig = image(grey, title=title, **size)
            heat = image(loc.dif, title=title, cmap="jet", clim=_signal_clim(loc.dif), **size)
            if loc.detected:
                # Open circle so the pixels directly under the estimate stay visible.
                marker = hv.Points(([loc.x], [loc.y])).opts(
                    color="red", size=20, marker="circle", fill_alpha=0, line_width=2
                )
                orig, heat = orig * marker, heat * marker
            panels.extend([orig, heat])
    finally:
        cap.release()
    return hv.Layout(panels).cols(2)


def _signal_clim(arr: np.ndarray) -> tuple[float, float]:
    """A robust ``(0, high-percentile)`` color range for a thresholded signal image.

    Most of a thresholded difference is exactly zero, and the surviving blob can
    hold a few extreme pixels. Auto-scaling to the raw max then paints almost
    everything at the colormap's low end -- the "all blue" look. Clipping the top
    to the 99th percentile of the *positive* values keeps the animal's blob in the
    warm colors where it actually reads.
    """
    pos = arr[arr > 0]
    hi = float(np.percentile(pos, 99)) if pos.size else 0.0
    return (0.0, hi if hi > 0 else 1.0)


def _display_size(shape: tuple[int, int], width: int) -> dict[str, int]:
    """Width/height opts that fit ``width`` px while preserving aspect ratio."""
    h, w = shape
    return {"width": width, "height": int(width * h / w)}


def trace(
    session: Session, df: pd.DataFrame, cmap: str = "viridis", width: int = 480
) -> hv.Element:
    """Overlay the animal's path (and any ROI outlines) on the reference frame.

    The path is drawn as a connected line colored by frame number, so you can
    read the motion and its direction over time (dark = start, bright = end),
    not just where the animal was.
    """
    size = _display_size(session.reference.shape, width)
    img = image(session.reference, session.stretch, "Motion Trace", **size)
    rois = session.selections.rois
    if rois:
        polys = hv.Polygons([[(v[0], v[1]) for v in poly] for poly in rois.polygons])
        img = img * polys.opts(fill_alpha=0.1, line_dash="dashed")
    path = hv.Path(
        [{"x": df["x"].to_numpy(), "y": df["y"].to_numpy(), "frame": df["frame"].to_numpy()}],
        vdims="frame",
    ).opts(color="frame", cmap=cmap, line_width=2, colorbar=True)
    return img * path


def outlier_plot(before: pd.DataFrame, after: pd.DataFrame) -> hv.Layout:
    """Time view of an outlier filter: ``x``, ``y``, and the correction vs frame.

    The original track is drawn faintly with the filtered track over it (corrected
    frames marked in red), then a third panel plots the **correction distance** --
    how far each frame's position was moved, ``hypot(after - before)``. That is the
    joint deviation the filter acted on (zero where nothing was changed), so you
    can see exactly when and how hard it stepped in. The x-axis is the real frame
    number, so it stays meaningful under temporal downsampling.
    """
    frames = before["frame"].to_numpy()
    correction = np.hypot(
        after["x"].to_numpy() - before["x"].to_numpy(),
        after["y"].to_numpy() - before["y"].to_numpy(),
    )
    changed = correction > 0

    panels = []
    for axis in ("x", "y"):
        b, a = before[axis].to_numpy(), after[axis].to_numpy()
        orig = hv.Curve((frames, b), "frame", axis, label="original").opts(
            color="lightgray", line_width=3
        )
        filt = hv.Curve((frames, a), "frame", axis, label="filtered").opts(color="steelblue")
        marks = hv.Scatter((frames[changed], a[changed])).opts(color="red", size=6)
        panels.append((orig * filt * marks).opts(title=f"{axis} vs frame", width=720, height=200))

    corr = hv.Curve((frames, correction), "frame", "correction (px)").opts(
        color="firebrick", width=720, height=200, title="filter correction vs frame"
    )
    corr_marks = hv.Scatter((frames[changed], correction[changed])).opts(color="red", size=6)
    panels.append(corr * corr_marks)
    return hv.Layout(panels).cols(1)


def heatmap(
    session: Session, df: pd.DataFrame, sigma: float | None = None, width: int = 480
) -> hv.Image:
    """Gaussian-smoothed occupancy heatmap of the animal's location."""
    h, w = session.reference.shape
    xi, yi = clip_to_indices(df["x"].to_numpy(), df["y"].to_numpy(), (h, w))
    grid = np.zeros((h, w))
    np.add.at(grid, (yi, xi), 1)  # vectorized occupancy count

    sigma = float(np.mean(grid.shape) * 0.05) if sigma is None else sigma
    grid = cv2.GaussianBlur(grid, (0, 0), sigma)
    if grid.max() > 0:
        grid = grid / grid.max() * 255
    return image(grid, session.stretch, "Heatmap", **_display_size((h, w), width)).opts(cmap="jet")
