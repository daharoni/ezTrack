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

try:
    __version__ = version("eztrack")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    __version__ = "0.0.0"

__all__ = ["__version__"]
