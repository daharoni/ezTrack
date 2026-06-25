"""Pixel-to-real-world distance scaling and the interactive measuring tool."""

from __future__ import annotations

import functools as fct

import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import streams

from .config import Scale, Session

hv.extension("bokeh")

__all__ = ["distance_tool", "set_scale", "scale_distance"]


def distance_tool(session: Session) -> hv.Element:
    """Interactive tool to measure the pixel distance between two clicked points.

    Mutates ``session.scale`` (a :class:`~eztrack.config.Scale`), whose
    ``px_distance`` is updated as the user moves the points, and returns the
    element to display.
    """
    reference = session.reference
    image = hv.Image((np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference))
    image.opts(
        width=int(reference.shape[1] * session.stretch.width),
        height=int(reference.shape[0] * session.stretch.height),
        invert_yaxis=True,
        cmap="gray",
        colorbar=True,
        toolbar="below",
        title="Select Points",
    )

    points = hv.Points([]).opts(active_tools=["point_draw"], color="red", size=10)
    point_draw = streams.PointDraw(source=points, num_objects=2)

    scale = Scale()
    session.scale = scale

    def markers(data, scale):
        try:
            x_ls, y_ls = data["x"], data["y"]
        except TypeError:
            x_ls, y_ls = [], []

        x_ctr, y_ctr = np.mean(x_ls), np.mean(y_ls)
        text = ""
        if len(x_ls) > 1:
            x_dist = x_ls[0] - x_ls[1]
            y_dist = y_ls[0] - y_ls[1]
            scale.px_distance = np.around((x_dist**2 + y_dist**2) ** (1 / 2), 3)
            text = f"{scale.px_distance} px"
        return hv.Labels((x_ctr, y_ctr, text)).opts(text_color="blue", text_font_size="14pt")

    markers_ptl = fct.partial(markers, scale=scale)
    dmap = hv.DynamicMap(markers_ptl, streams=[point_draw])
    return image * points * dmap


def set_scale(session: Session, distance: float, scale: str) -> None:
    """Record the real-world distance and unit for the points measured by ``distance_tool``."""
    if session.scale is None:
        session.scale = Scale()
    session.scale.true_distance = distance
    session.scale.true_scale = scale


def scale_distance(session: Session, df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Append a real-world-scaled copy of ``column`` to ``df`` using ``session.scale``.

    Returns ``df`` unchanged when no scale has been defined; prints a warning
    when a scale exists but the pixel distance was never measured.
    """
    if session.scale is None:
        return df
    if session.scale.px_distance is None:
        print(
            f"Distance between reference points undefined. Cannot scale column: {column}. "
            "Returning original dataframe"
        )
        return df

    session.scale.factor = session.scale.true_distance / session.scale.px_distance
    new_column = f"Distance_{session.scale.true_scale}"
    df[new_column] = df[column] * session.scale.factor
    order = [col for col in df if col not in [column, new_column]] + [column, new_column]
    return df[order]
