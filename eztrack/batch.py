"""Batch processing: run the tracking pipeline across a folder of videos.

One shared :class:`~eztrack.config.Selections` and :class:`~eztrack.config.TrackParams`
are applied to every file (the reference frame is rebuilt per video). Draw the
selections once on the first file, or load them from a saved config.
"""

from __future__ import annotations

import os

import holoviews as hv
import pandas as pd

from .analyze import summarize
from .config import Session, TrackParams
from .tracking import track
from .video import check_decodable, reference_frame
from .viz import heatmap, trace

hv.extension("bokeh")

__all__ = ["batch_process"]


def batch_process(
    session: Session,
    params: TrackParams,
    bins: dict[str, tuple[int, int]] | None = None,
    num_frames: int = 50,
    check: bool = True,
) -> tuple[pd.DataFrame, hv.Layout]:
    """Track every file in ``session.file_names`` with the one shared config.

    Writes ``<video>_tracking.csv`` per file and a combined ``BatchSummary.csv``
    in ``session.dpath``. Returns the combined summary and a HoloViews layout of
    each video's trace and heatmap.
    """
    summaries, panels = [], []
    for file in session.file_names:
        print(f"Processing: {file}")
        session.file = file
        if check:
            check_decodable(session)
        reference_frame(session, num_frames=num_frames)

        df = track(session, params)
        df.to_csv(os.path.splitext(session.fpath)[0] + "_tracking.csv", index=False)
        summaries.append(summarize(df, session, bins=bins))

        panels.extend([trace(session, df).opts(title=file), heatmap(session, df).opts(title=file)])

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(os.path.join(os.path.normpath(session.dpath), "BatchSummary.csv"), index=False)
    return summary, hv.Layout(panels)
