"""Typed configuration objects for the LocationTracking pipeline.

These dataclasses replace the stringly-typed dictionaries (``video_dict``,
``tracking_params``, ``display_dict``, ``scale``) that the original ezTrack
threaded through every function.

:class:`Session` is the central object. It holds the user's inputs *and* the
state the pipeline builds up as it runs (reference frame, crop/ROI/mask
selections, scale). Pipeline functions take a ``Session`` and **mutate it in
place** -- e.g. :func:`eztrack.io.load_and_crop` sets ``session.crop`` and
``session.f0`` -- then return only the HoloViews element to display.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

__all__ = ["Stretch", "Scale", "Mask", "TrackingParams", "DisplayParams", "Session"]


def _from_dict(cls, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys (migration aid)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Stretch:
    """Display-only aspect-ratio scaling for plotted frames (never affects analysis)."""

    width: float = 1.0
    height: float = 1.0


@dataclass
class Scale:
    """Pixel-to-real-world distance conversion (was the ``scale``/``distance`` dict)."""

    px_distance: float | None = None
    true_distance: float | None = None
    true_scale: str | None = None
    factor: float | None = None


@dataclass
class Mask:
    """Region to exclude from analysis (was the ``mask`` dict ``{'mask','stream'}``)."""

    array: np.ndarray | None = None
    stream: Any = None  # holoviews PolyDraw stream


@dataclass
class TrackingParams:
    """Parameters controlling location tracking (was ``tracking_params``).

    See the LocationTracking notebook for a full description of each field.
    """

    loc_thresh: float = 99.5
    use_window: bool = True
    window_size: int = 100
    window_weight: float = 0.9
    method: str = "abs"  # 'abs' | 'light' | 'dark'
    rmv_wire: bool = False
    wire_krn: int = 5

    from_dict = classmethod(_from_dict)


@dataclass
class DisplayParams:
    """Playback options for :func:`eztrack.playback.play_video` (was ``display_dict``)."""

    start: int = 0
    stop: int = 200
    fps: int = 30
    resize: tuple[int, int] | None = None
    save_video: bool = False

    from_dict = classmethod(_from_dict)


@dataclass
class Session:
    """A tracking session: user inputs plus pipeline-built state (was ``video_dict``).

    User inputs
        dpath, file, start, end, region_names, dsmpl, stretch, ftype, altfile.
    Built up by the pipeline (start as ``None``)
        fpath, f0 (first frame), reference, crop (BoxEdit stream),
        roi_stream (PolyDraw stream), mask, scale, file_names (batch).
    """

    dpath: str
    file: str | None = None
    start: int = 0
    end: int | None = None
    region_names: list[str] | None = None
    dsmpl: float = 1.0
    stretch: Stretch = field(default_factory=Stretch)
    ftype: str | None = None  # batch: video file extension to glob for
    altfile: str | None = None  # alternative video used to build the reference

    # state populated as the pipeline runs
    fpath: str | None = None
    f0: np.ndarray | None = None
    reference: np.ndarray | None = None
    crop: Any = None  # holoviews BoxEdit stream
    roi_stream: Any = None  # holoviews PolyDraw stream
    mask: Mask | None = None
    scale: Scale | None = None
    file_names: list[str] = field(default_factory=list)  # batch

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Build a Session from a legacy ``video_dict`` (nested dicts are converted)."""
        data = dict(data)
        if isinstance(data.get("stretch"), dict):
            data["stretch"] = _from_dict(Stretch, data["stretch"])
        if isinstance(data.get("scale"), dict):
            data["scale"] = _from_dict(Scale, data["scale"])
        if isinstance(data.get("mask"), dict):
            m = data["mask"]
            data["mask"] = Mask(array=m.get("mask"), stream=m.get("stream"))
        return _from_dict(cls, data)
