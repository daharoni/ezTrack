"""ezTrack: open-source video analysis for animal behavior.

This package ships the **LocationTracking** module, which tracks a single
animal's location frame-by-frame (e.g. in an open field), quantifies distance
travelled and time spent in user-defined regions of interest, and renders
traces and heatmaps.

The analysis functions live in :mod:`eztrack.location`; the interactive
workflow is driven from the bundled Jupyter notebooks, which you can copy into a
working directory with the CLI::

    eztrack notebooks list
    eztrack notebooks copy individual
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("eztrack")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

# Public API re-exported lazily from eztrack.location so that lightweight uses
# (e.g. the `eztrack notebooks` CLI) don't pull in holoviews/opencv just to
# import the package. `from eztrack import Session` triggers the import on first
# access via __getattr__ below.
_LAZY = {
    "Session",
    "TrackingParams",
    "DisplayParams",
    "Scale",
    "Stretch",
    "Mask",
    "load_and_crop",
    "make_reference",
    "crop_frame",
    "batch_load_files",
    "check_p_frames",
    "locate",
    "track_location",
    "roi_plot",
    "roi_location",
    "roi_linearize",
    "roi_transitions",
    "mask_select",
    "distance_tool",
    "set_scale",
    "scale_distance",
    "summarize_location",
    "location_thresh_view",
    "show_trace",
    "heatmap",
    "play_video",
    "play_video_ext",
    "batch_process",
}

__all__ = ["__version__", *sorted(_LAZY)]

if TYPE_CHECKING:  # let type checkers/IDEs resolve the lazy names
    from .location import *  # noqa: F403


def __getattr__(name: str):
    if name in _LAZY:
        from . import location

        return getattr(location, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
