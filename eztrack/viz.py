"""Visualizations: tracking-threshold previews, motion traces, occupancy heatmaps."""

from __future__ import annotations

import cv2
import holoviews as hv
import numpy as np
import pandas as pd

from .config import Session, TrackingParams
from .tracking import locate

hv.extension("bokeh")

__all__ = ["location_thresh_view", "show_trace", "heatmap"]


def location_thresh_view(session: Session, params: TrackingParams, examples: int = 4) -> hv.Layout:
    """Show tracking on a random subset of frames to sanity-check ``params``.

    Original frame (left) and thresholded difference with the center of mass
    marked (right). Windowing is not applied (frames are analysed in isolation).
    """
    cap = cv2.VideoCapture(session.fpath)
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_max = int(session.end) if session.end is not None else cap_max

    images = []
    for _example in range(examples):
        ret = False
        while ret is False:
            frm = np.random.randint(session.start, cap_max)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
            ret, dif, com, frame = locate(cap, params, session)

        image_orig = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
        image_orig.opts(
            width=int(session.reference.shape[1] * session.stretch.width),
            height=int(session.reference.shape[0] * session.stretch.height),
            invert_yaxis=True,
            cmap="gray",
            toolbar="below",
            title="Frame: " + str(frm),
        )
        orig_overlay = image_orig * hv.Points(([com[1]], [com[0]])).opts(
            color="red", size=20, marker="+", line_width=3
        )

        dif = dif * (255 // dif.max())
        image_heat = hv.Image((np.arange(dif.shape[1]), np.arange(dif.shape[0]), dif))
        image_heat.opts(
            width=int(dif.shape[1] * session.stretch.width),
            height=int(dif.shape[0] * session.stretch.height),
            invert_yaxis=True,
            cmap="jet",
            toolbar="below",
            title="Frame: " + str(frm - session.start),
        )
        heat_overlay = image_heat * hv.Points(([com[1]], [com[0]])).opts(
            color="red", size=20, marker="+", line_width=3
        )

        images.extend([orig_overlay, heat_overlay])

    cap.release()
    return hv.Layout(images)


def show_trace(
    session: Session, location: pd.DataFrame, color: str = "red", alpha: float = 0.8, size: int = 3
) -> hv.Element:
    """Overlay the animal's traced path on the reference frame (with ROI outlines)."""
    poly = None
    if session.roi_stream is not None:
        lst = []
        for poly_idx in range(len(session.roi_stream.data["xs"])):
            x = np.array(session.roi_stream.data["xs"][poly_idx])
            y = np.array(session.roi_stream.data["ys"][poly_idx])
            lst.append([(x[vert], y[vert]) for vert in range(len(x))])
        poly = hv.Polygons(lst).opts(fill_alpha=0.1, line_dash="dashed")

    reference = session.reference
    image = hv.Image(
        (np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference)
    ).opts(
        width=int(reference.shape[1] * session.stretch.width),
        height=int(reference.shape[0] * session.stretch.height),
        invert_yaxis=True,
        cmap="gray",
        toolbar="below",
        title="Motion Trace",
    )

    points = hv.Scatter(np.array([location["X"], location["Y"]]).T).opts(
        color=color, alpha=alpha, size=size
    )

    return (image * poly * points) if poly is not None else (image * points)


def heatmap(session: Session, location: pd.DataFrame, sigma: float | None = None) -> hv.Image:
    """Gaussian-smoothed occupancy heatmap of the animal's location over the session."""
    grid = np.zeros(session.reference.shape)
    ys = location["Y"].to_numpy()
    xs = location["X"].to_numpy()
    for frame in range(len(location)):
        grid[int(ys[frame]), int(xs[frame])] += 1

    sigma = np.mean(grid.shape) * 0.05 if sigma is None else sigma
    grid = cv2.GaussianBlur(grid, (0, 0), sigma)
    grid = (grid / grid.max()) * 255

    map_i = hv.Image((np.arange(grid.shape[1]), np.arange(grid.shape[0]), grid))
    map_i.opts(
        width=int(grid.shape[1] * session.stretch.width),
        height=int(grid.shape[0] * session.stretch.height),
        invert_yaxis=True,
        cmap="jet",
        alpha=1,
        colorbar=False,
        toolbar="below",
        title="Heatmap",
    )
    return map_i
