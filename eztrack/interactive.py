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

__all__ = [
    "file_chooser",
    "crop_tool",
    "mask_tool",
    "roi_tool",
    "name_rois",
    "distance_tool",
    "set_scale",
]

# Common behavior-video container formats, used as the default picker filter.
_VIDEO_PATTERNS = ["*.avi", "*.mp4", "*.wmv", "*.mpg", "*.mpeg", "*.mov", "*.mkv"]


def file_chooser(start_dir: str = ".", patterns: list[str] | None = None):
    """Display an in-notebook file picker and return the ``FileChooser`` widget.

    Browses the kernel's filesystem (so it works the same locally or on a remote
    Jupyter server, unlike a native ``tkinter`` dialog). After the user picks a
    file, build a :class:`~eztrack.config.Session` from the selection::

        fc = ez.file_chooser("../../PracticeVideos/")
        fc                                    # shows the picker
        # ...pick a file, then in a later cell:
        session = ez.Session(dpath=fc.selected_path, file=fc.selected_filename)

    ``patterns`` defaults to the common video extensions; pass your own glob list
    to override. ``ipyfilechooser`` is imported lazily so the headless pipeline
    never needs it.
    """
    from ipyfilechooser import FileChooser

    fc = FileChooser(start_dir)
    fc.title = "Select a behavior video"
    fc.filter_pattern = patterns or _VIDEO_PATTERNS
    return fc


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
    """Draw regions of interest; writes ``selections.rois``.

    Draw as many regions as you like -- each is labelled live (with the declared
    ``session.region_names`` if set, otherwise ``zone_1``, ``zone_2`` ...). If
    ``region_names`` is set it also caps how many regions you can draw. Name (or
    rename) them afterwards with :func:`name_rois`.
    """
    names = session.region_names or []
    title = "Draw regions: " + ", ".join(names) if names else "Draw regions (name them next)"
    img = image(session.reference, session.stretch, title, colorbar=True)

    poly = hv.Polygons([]).opts(fill_alpha=0.3, active_tools=["poly_draw"])
    # num_objects=0 means unlimited; cap only when names were declared up front.
    stream = streams.PolyDraw(source=poly, drag=True, num_objects=len(names), show_vertices=True)

    def _update(data):
        session.selections.rois = ROIs.from_polydraw(names, data)
        rois = session.selections.rois
        centers_x = [np.mean([v[0] for v in poly]) for poly in rois.polygons]
        centers_y = [np.mean([v[1] for v in poly]) for poly in rois.polygons]
        return hv.Labels((centers_x, centers_y, rois.names))

    dmap = hv.DynamicMap(_update, streams=[stream])
    return img * poly * dmap


def name_rois(session: Session, names: list[str]) -> hv.Element:
    """Assign names to the regions drawn with :func:`roi_tool`.

    Requires exactly one name per drawn region (raising otherwise), then writes
    them onto ``session.selections.rois`` and mirrors them to
    ``session.region_names``. Returns the reference frame with the named regions
    outlined and labelled, so you can confirm the order is what you intended.
    """
    rois = session.selections.rois
    drawn = len(rois.polygons) if rois else 0
    if drawn == 0:
        raise ValueError("No regions drawn yet -- run roi_tool and draw at least one region first.")
    if len(names) != drawn:
        raise ValueError(
            f"You drew {drawn} region(s) but gave {len(names)} name(s); they must match."
        )

    rois.names = list(names)
    session.region_names = list(names)

    img = image(session.reference, session.stretch, "Named regions", colorbar=True)
    polys = hv.Polygons([[(v[0], v[1]) for v in poly] for poly in rois.polygons]).opts(
        fill_alpha=0.1, line_dash="dashed"
    )
    centers_x = [np.mean([v[0] for v in poly]) for poly in rois.polygons]
    centers_y = [np.mean([v[1] for v in poly]) for poly in rois.polygons]
    labels = hv.Labels((centers_x, centers_y, rois.names))
    return img * polys * labels


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
