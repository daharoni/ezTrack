"""ezTrack: open-source video analysis for animal location tracking.

Tracks a single animal's location frame-by-frame in a behavior video (e.g. an
open field), quantifies distance travelled and time spent in user-defined
regions, and renders motion traces and occupancy heatmaps.

The pipeline is a handful of explicit steps over a :class:`Session`::

    import eztrack as ez

    session = ez.Session(dpath="videos", file="clip.mp4", region_names=["left", "right"])
    ez.crop_tool(session)                 # draw a crop box (or set session.selections.crop)
    ez.reference_frame(session)           # build the background frame
    ez.roi_tool(session)                  # draw the named regions
    df = ez.track(session, ez.TrackParams(method="dark"))
    summary = ez.summarize(df, session)

Selections (crop, mask, ROIs, scale) are plain serializable objects: draw them
once, ``session.selections.save("sel.json")``, and replay headlessly with
``session.selections = ez.Selections.load("sel.json")``.

Config objects (:class:`Session`, :class:`Selections`, :class:`TrackParams`, ...)
import without OpenCV/HoloViews; the heavier pipeline functions are imported
lazily so the ``eztrack notebooks`` CLI stays light.
"""

from importlib.metadata import PackageNotFoundError, version

# Lightweight, dependency-free config objects -- safe to import eagerly.
from .config import (
    Crop,
    Mask,
    PlayParams,
    ROIs,
    Scale,
    Selections,
    Session,
    Stretch,
    TrackParams,
    Window,
)

try:
    __version__ = version("eztrack")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0"

# Heavier functions (pull in OpenCV/HoloViews) are resolved lazily on first access.
_LAZY = {
    # video
    "reference_frame": "video",
    "discover_files": "video",
    "check_decodable": "video",
    "preprocess": "video",
    # tracking
    "track": "tracking",
    "locate": "tracking",
    "hampel_filter": "tracking",
    # analyze
    "summarize": "analyze",
    "apply_scale": "analyze",
    # regions
    "roi_membership": "regions",
    "mask_array": "regions",
    # interactive
    "file_chooser": "interactive",
    "crop_tool": "interactive",
    "mask_tool": "interactive",
    "roi_tool": "interactive",
    "name_rois": "interactive",
    "distance_tool": "interactive",
    "set_scale": "interactive",
    # viz
    "image": "viz",
    "trace": "viz",
    "heatmap": "viz",
    "threshold_preview": "viz",
    "outlier_plot": "viz",
    # playback
    "play_inline": "playback",
    "play_window": "playback",
    # batch
    "batch_process": "batch",
}

__all__ = [
    "__version__",
    "Session",
    "Selections",
    "Crop",
    "Mask",
    "ROIs",
    "Scale",
    "Window",
    "Stretch",
    "TrackParams",
    "PlayParams",
    *sorted(_LAZY),
]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)
