"""Backward-compatible façade for the LocationTracking pipeline.

The implementation now lives in focused submodules (:mod:`eztrack.io`,
:mod:`eztrack.tracking`, :mod:`eztrack.roi`, :mod:`eztrack.scale`,
:mod:`eztrack.summary`, :mod:`eztrack.viz`, :mod:`eztrack.playback`) with typed
config objects in :mod:`eztrack.config`. This module re-exports the public API
so ``import eztrack.location as lt`` keeps working in the notebooks.
"""

from __future__ import annotations

# batch lives in its own module to break the import cycle (it depends on most others)
from .batch import batch_process
from .config import DisplayParams, Mask, Scale, Session, Stretch, TrackingParams
from .io import (
    batch_load_files,
    check_p_frames,
    crop_frame,
    load_and_crop,
    make_reference,
)
from .playback import play_video, play_video_ext
from .roi import mask_select, roi_linearize, roi_location, roi_plot, roi_transitions
from .scale import distance_tool, scale_distance, set_scale
from .summary import summarize_location
from .tracking import locate, track_location
from .viz import heatmap, location_thresh_view, show_trace

__all__ = [
    # config
    "Session",
    "TrackingParams",
    "DisplayParams",
    "Scale",
    "Stretch",
    "Mask",
    # io
    "load_and_crop",
    "make_reference",
    "crop_frame",
    "batch_load_files",
    "check_p_frames",
    # tracking
    "locate",
    "track_location",
    # roi
    "roi_plot",
    "roi_location",
    "roi_linearize",
    "roi_transitions",
    "mask_select",
    # scale
    "distance_tool",
    "set_scale",
    "scale_distance",
    # summary
    "summarize_location",
    # viz
    "location_thresh_view",
    "show_trace",
    "heatmap",
    # playback
    "play_video",
    "play_video_ext",
    # batch
    "batch_process",
]
