"""Regions of interest: drawing tools, per-frame membership, labels, transitions, masks."""

from __future__ import annotations

import functools as fct
import os

import cv2
import holoviews as hv
import numpy as np
import pandas as pd
from holoviews import streams

from .config import Mask, Session
from .io import _preprocess_frame, crop_frame

hv.extension("bokeh")

__all__ = [
    "roi_plot",
    "roi_location",
    "roi_linearize",
    "roi_transitions",
    "mask_select",
]


def roi_plot(session: Session) -> hv.Element:
    """Interactive tool to draw the regions named in ``session.region_names``.

    Mutates ``session.roi_stream`` (a PolyDraw stream, or ``None`` when there
    are no regions) and returns the element to display.
    """
    region_names = session.region_names
    nobjects = len(region_names) if region_names else 0

    reference = session.reference
    image = hv.Image((np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference))
    image.opts(
        width=int(reference.shape[1] * session.stretch.width),
        height=int(reference.shape[0] * session.stretch.height),
        invert_yaxis=True,
        cmap="gray",
        colorbar=True,
        toolbar="below",
        title="No Regions to Draw" if nobjects == 0 else "Draw Regions: " + ", ".join(region_names),
    )

    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=["poly_draw"])

    def centers(data):
        try:
            x_ls, y_ls = data["xs"], data["ys"]
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        rois = region_names[: len(xs)]
        return hv.Labels((xs, ys, rois))

    if nobjects > 0:
        session.roi_stream = poly_stream
        dmap = hv.DynamicMap(centers, streams=[poly_stream])
        return image * poly * dmap
    session.roi_stream = None
    return image


def roi_location(session: Session, location: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean column per region indicating whether the animal is inside it.

    No-op when ``session.region_names`` is ``None``. Also records the ROI vertex
    coordinates in an ``ROI_coordinates`` column.
    """
    if session.region_names is None:
        return location

    roi_masks = {}
    for poly in range(len(session.roi_stream.data["xs"])):
        x = np.array(session.roi_stream.data["xs"][poly])
        y = np.array(session.roi_stream.data["ys"][poly])
        xy = np.column_stack((x, y)).astype(np.int32)  # cv2 needs int32 contours
        mask = np.zeros(session.reference.shape)
        cv2.fillPoly(mask, pts=[xy], color=255)
        roi_masks[session.region_names[poly]] = mask == 255

    # Positional (numpy) indexing rather than by the 'Frame' label so it does
    # not depend on the dataframe index matching frame numbers under pandas 3.0.
    yi = location["Y"].to_numpy().astype(np.intp)
    xi = location["X"].to_numpy().astype(np.intp)
    for name, mask in roi_masks.items():
        location[name] = mask[yi, xi]

    location["ROI_coordinates"] = str(session.roi_stream.data)
    return location


def roi_linearize(rois: pd.DataFrame, null_name: str = "non_roi") -> pd.Series:
    """Collapse per-region boolean columns into one label per frame.

    Frames in multiple regions get the names joined in column order (e.g.
    ``left_top``); frames in none get ``null_name``.
    """
    cols = list(rois.columns)
    membership = rois.to_numpy(dtype=bool)
    labels = [
        "_".join(c for c, in_region in zip(cols, row, strict=True) if in_region) or null_name
        for row in membership
    ]
    return pd.Series(labels, index=rois.index, dtype=object)


def roi_transitions(regions: pd.Series, include_first: bool = False) -> pd.Series:
    """Boolean series flagging frames where the ROI label differs from the previous frame."""
    transitions = regions != regions.shift(1, fill_value=regions.iloc[0])
    if include_first:
        transitions.iloc[0] = True
    return transitions


def mask_select(session: Session, fstfile: bool = False) -> hv.Element:
    """Interactive tool to draw regions to exclude from analysis.

    Mutates ``session.mask`` (a :class:`~eztrack.config.Mask`) and returns the
    element to display. In batch mode (``fstfile=True``) the first file's first
    frame is loaded into ``session.f0``.
    """
    if fstfile:
        session.file = session.file_names[0]
        session.fpath = os.path.join(os.path.normpath(session.dpath), session.file)
        if not os.path.isfile(session.fpath):
            raise FileNotFoundError(
                f"{session.fpath} not found. Check that directory and file names are correct"
            )
        print(f"file: {session.fpath}")
        cap = cv2.VideoCapture(session.fpath)
        cap.set(cv2.CAP_PROP_POS_FRAMES, session.start)
        _ret, frame = cap.read()
        session.f0 = _preprocess_frame(frame, session, crop=False)

    f0 = crop_frame(session.f0, session.crop)
    image = hv.Image((np.arange(f0.shape[1]), np.arange(f0.shape[0]), f0))
    image.opts(
        width=int(f0.shape[1] * session.stretch.width),
        height=int(f0.shape[0] * session.stretch.height),
        invert_yaxis=True,
        cmap="gray",
        colorbar=True,
        toolbar="below",
        title="Draw Regions to be Exluded",
    )

    mask = Mask()
    poly = hv.Polygons([])
    mask.stream = streams.PolyDraw(source=poly, drag=True, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=["poly_draw"])
    session.mask = mask

    def make_mask(data, mask):
        try:
            x_ls, _y_ls = data["xs"], data["ys"]
        except TypeError:
            x_ls, _y_ls = [], []

        if len(x_ls) > 0:
            mask.array = np.zeros(f0.shape)
            for submask in range(len(x_ls)):
                x = np.array(mask.stream.data["xs"][submask])
                y = np.array(mask.stream.data["ys"][submask])
                xy = np.column_stack((x, y)).astype(np.int32)  # cv2 needs int32 contours
                cv2.fillPoly(mask.array, pts=[xy], color=1)
            mask.array = mask.array.astype("bool")
        return hv.Labels((0, 0, ""))

    make_mask_ptl = fct.partial(make_mask, mask=mask)
    dmap = hv.DynamicMap(make_mask_ptl, streams=[mask.stream])
    return image * poly * dmap
