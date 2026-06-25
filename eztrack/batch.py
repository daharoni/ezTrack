"""Batch processing: run the tracking pipeline across a folder of videos."""

from __future__ import annotations

import os

import cv2
import holoviews as hv
import pandas as pd

from .config import Session, TrackingParams
from .io import check_p_frames, make_reference
from .summary import summarize_location
from .tracking import track_location
from .viz import heatmap, show_trace

hv.extension("bokeh")

__all__ = ["batch_process"]


def batch_process(
    session: Session,
    params: TrackingParams,
    bin_dict: dict | None,
    accept_p_frames: bool = False,
) -> tuple[pd.DataFrame, hv.Layout]:
    """Track every file in ``session.file_names`` with one shared parameter set.

    Writes a per-file ``*_LocationOutput.csv`` and a combined ``BatchSummary.csv``
    in ``session.dpath``. Returns the combined summary dataframe and a HoloViews
    layout of each session's trace and heatmap. The reference frame is generated
    per video (median of 50 frames).
    """
    images = []
    summary_all = None
    for file in session.file_names:
        print(f"Processing File: {file}")
        session.file = file
        session.fpath = os.path.join(os.path.normpath(session.dpath), file)

        cap = cv2.VideoCapture(session.fpath)
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"total frames: {cap_max}")
        print(f"nominal fps: {cap.get(cv2.CAP_PROP_FPS)}")
        print(
            f"dimensions (h x w): {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))},"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
        )
        if accept_p_frames is False:
            check_p_frames(cap)
        cap.release()

        make_reference(session, num_frames=50)
        location = track_location(session, params)
        location.to_csv(os.path.splitext(session.fpath)[0] + "_LocationOutput.csv", index=False)
        file_summary = summarize_location(location, session, bin_dict=bin_dict)
        summary_all = (
            file_summary
            if summary_all is None
            else pd.concat([summary_all, file_summary], sort=False)
        )

        trace = show_trace(session, location)
        hmap = heatmap(session, location, sigma=None)
        images += [trace.opts(title=file), hmap.opts(title=file)]

    sum_pathout = os.path.join(os.path.normpath(session.dpath), "BatchSummary.csv")
    summary_all.to_csv(sum_pathout, index=False)

    return summary_all, hv.Layout(images)
