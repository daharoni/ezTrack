"""Interactive HoloViews widgets for picking selections in a notebook.

Each tool draws on the first/reference frame and, as the user draws, writes a
plain serializable object into ``session.selections`` (a :class:`~eztrack.config.Crop`,
:class:`~eztrack.config.Mask`, :class:`~eztrack.config.ROIs`, or
:class:`~eztrack.config.Scale`). Those objects are exactly what the headless core
and the config-file replay path consume -- the widget is just one way to produce
them, which is why the pipeline is fully testable without a browser.
"""

from __future__ import annotations

import functools

import holoviews as hv
import numpy as np
from holoviews import streams

from .config import Crop, Mask, ROIs, Scale, Session
from .video import first_frame
from .viz import image

hv.extension("bokeh")

__all__ = ["crop_tool", "mask_tool", "roi_tool", "distance_tool", "set_scale"]


def crop_tool(session: Session) -> hv.Element:
    """Load the first frame and expose a box-crop tool; writes ``selections.crop``."""
    frame = first_frame(session, crop=False)
    img = image(frame, session.stretch, "First Frame -- crop if desired", colorbar=True)

    box = hv.Polygons([]).opts(alpha=0.5)
    stream = streams.BoxEdit(source=box, num_objects=1)

    def _update(data):
        session.selections.crop = Crop.from_boxedit(data)

    stream.add_subscriber(_update)
    return img * box


def mask_tool(session: Session) -> hv.Element:
    """Draw regions to exclude from analysis; writes ``selections.mask``."""
    frame = first_frame(session, crop=True)
    img = image(frame, session.stretch, "Draw regions to EXCLUDE", colorbar=True)

    poly = hv.Polygons([]).opts(fill_alpha=0.3, active_tools=["poly_draw"])
    stream = streams.PolyDraw(source=poly, drag=True, show_vertices=True)

    def _update(data):
        session.selections.mask = Mask.from_polydraw(data)
        return hv.Labels([])

    dmap = hv.DynamicMap(_update, streams=[stream])
    return img * poly * dmap


def roi_tool(session: Session) -> hv.Element:
    """Draw the regions named in ``session.region_names``; writes ``selections.rois``."""
    names = session.region_names or []
    img = image(
        session.reference,
        session.stretch,
        "Draw regions: " + ", ".join(names) if names else "No regions to draw",
        colorbar=True,
    )
    if not names:
        return img

    poly = hv.Polygons([]).opts(fill_alpha=0.3, active_tools=["poly_draw"])
    stream = streams.PolyDraw(source=poly, drag=True, num_objects=len(names), show_vertices=True)

    def _update(data):
        session.selections.rois = ROIs.from_polydraw(names, data)
        rois = session.selections.rois
        centers_x = [np.mean([v[0] for v in poly]) for poly in rois.polygons]
        centers_y = [np.mean([v[1] for v in poly]) for poly in rois.polygons]
        return hv.Labels((centers_x, centers_y, rois.names))

    dmap = hv.DynamicMap(_update, streams=[stream])
    return img * poly * dmap


def distance_tool(session: Session) -> hv.Element:
    """Click two points of known distance; writes ``selections.scale.px_distance``."""
    session.selections.scale = session.selections.scale or Scale()
    img = image(session.reference, session.stretch, "Select two points", colorbar=True)

    points = hv.Points([]).opts(active_tools=["point_draw"], color="red", size=10)
    stream = streams.PointDraw(source=points, num_objects=2)

    def _update(data, scale):
        xs, ys = (data or {}).get("x", []), (data or {}).get("y", [])
        if len(xs) > 1:
            scale.px_distance = float(np.round(np.hypot(xs[0] - xs[1], ys[0] - ys[1]), 3))
            return hv.Labels((np.mean(xs), np.mean(ys), f"{scale.px_distance} px")).opts(
                text_color="blue", text_font_size="14pt"
            )
        return hv.Labels([])

    dmap = hv.DynamicMap(
        functools.partial(_update, scale=session.selections.scale), streams=[stream]
    )
    return img * points * dmap


def set_scale(session: Session, real_distance: float, unit: str) -> None:
    """Record the real-world distance and unit for the points from :func:`distance_tool`."""
    session.selections.scale = session.selections.scale or Scale()
    session.selections.scale.real_distance = real_distance
    session.selections.scale.unit = unit
