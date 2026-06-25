"""Plain, serializable configuration objects for the location-tracking pipeline.

Nothing in this module imports OpenCV or HoloViews -- these are pure data
classes describing *what* to analyse, never *how* to render it. Two ideas:

* **Selections** (:class:`Crop`, :class:`Mask`, :class:`ROIs`, :class:`Scale`)
  are the geometry the user picks. They are JSON-serializable so an analysis is
  fully reproducible: draw them once in the notebook, ``selections.save(...)``,
  and replay them headlessly or across a batch with ``Selections.load(...)``.
  The interactive HoloViews widgets in :mod:`eztrack.interactive` are just one
  way to *produce* these objects -- you can also construct them by hand.
* **Parameters** (:class:`TrackParams`, :class:`Window`) control the tracking
  maths, validated on construction.

:class:`Session` is a thin notebook-friendly container bundling the video
inputs, the :class:`Selections`, and the reference frame. The pure core in
:mod:`eztrack.tracking` can also be driven without it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Stretch",
    "Crop",
    "Mask",
    "ROIs",
    "Scale",
    "Window",
    "TrackParams",
    "PlayParams",
    "Selections",
    "Session",
]

METHODS = ("abs", "light", "dark")


@dataclass
class Stretch:
    """Display-only aspect-ratio scaling for plotted frames (never affects analysis)."""

    width: float = 1.0
    height: float = 1.0


@dataclass
class Crop:
    """A rectangular crop in (uncropped) frame pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def bounds(self) -> tuple[int, int, int, int]:
        """Integer ``(xmin, ymin, xmax, ymax)`` with corners ordered."""
        xmin, xmax = sorted((self.x0, self.x1))
        ymin, ymax = sorted((self.y0, self.y1))
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return the cropped view of ``frame``."""
        xmin, ymin, xmax, ymax = self.bounds()
        return frame[ymin:ymax, xmin:xmax]

    @classmethod
    def from_boxedit(cls, data: dict | None) -> Crop | None:
        """Build a Crop from a HoloViews ``BoxEdit`` stream's ``.data`` dict.

        Returns ``None`` when no box has been drawn.
        """
        if not data or not data.get("x0"):
            return None
        return cls(x0=data["x0"][0], y0=data["y0"][0], x1=data["x1"][0], y1=data["y1"][0])


@dataclass
class Mask:
    """Polygonal regions (in cropped-frame coordinates) to exclude from analysis."""

    polygons: list[list[list[float]]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.polygons)

    @classmethod
    def from_polydraw(cls, data: dict | None) -> Mask:
        """Build a Mask from a HoloViews ``PolyDraw`` stream's ``.data`` dict."""
        return cls(polygons=_polydraw_polygons(data))


@dataclass
class ROIs:
    """Named polygonal regions of interest (in cropped-frame coordinates)."""

    names: list[str] = field(default_factory=list)
    polygons: list[list[list[float]]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.polygons)

    @classmethod
    def from_polydraw(cls, names: list[str] | None, data: dict | None) -> ROIs:
        """Build ROIs from declared ``names`` and a ``PolyDraw`` stream's ``.data`` dict.

        Names are matched positionally to drawn polygons.
        """
        polygons = _polydraw_polygons(data)
        names = list(names or [])[: len(polygons)]
        return cls(names=names, polygons=polygons)


@dataclass
class Scale:
    """Pixel-to-real-world distance conversion."""

    px_distance: float | None = None
    real_distance: float | None = None
    unit: str | None = None

    @property
    def factor(self) -> float | None:
        """Real-world units per pixel, or ``None`` if not fully defined."""
        if self.px_distance and self.real_distance:
            return self.real_distance / self.px_distance
        return None


@dataclass
class Window:
    """Search window that biases tracking toward the animal's previous location."""

    size: int = 100
    weight: float = 0.9


@dataclass
class TrackParams:
    """Parameters controlling the per-frame location estimate (validated)."""

    threshold_pct: float = 99.5
    method: str = "abs"  # 'abs' | 'light' | 'dark'
    window: Window | None = field(default_factory=Window)
    remove_wire: bool = False
    wire_kernel: int = 5

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not 0 <= self.threshold_pct <= 100:
            raise ValueError(f"threshold_pct must be in [0, 100], got {self.threshold_pct}")
        if self.window is not None and not 0 <= self.window.weight <= 1:
            raise ValueError(f"window.weight must be in [0, 1], got {self.window.weight}")


@dataclass
class PlayParams:
    """Options for inline/external playback of a tracked segment."""

    start: int = 0
    stop: int = 200
    fps: int = 30
    resize: tuple[int, int] | None = None
    save: bool = False


@dataclass
class Selections:
    """The geometry a user picks for a video: crop, exclusion mask, ROIs, scale.

    Fully JSON-serializable so an analysis can be saved and replayed.
    """

    crop: Crop | None = None
    mask: Mask | None = None
    rois: ROIs | None = None
    scale: Scale | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form suitable for ``json.dump``."""
        return {
            "crop": asdict(self.crop) if self.crop else None,
            "mask": asdict(self.mask) if self.mask else None,
            "rois": asdict(self.rois) if self.rois else None,
            "scale": asdict(self.scale) if self.scale else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Selections:
        """Inverse of :meth:`to_dict`."""
        return cls(
            crop=Crop(**data["crop"]) if data.get("crop") else None,
            mask=Mask(**data["mask"]) if data.get("mask") else None,
            rois=ROIs(**data["rois"]) if data.get("rois") else None,
            scale=Scale(**data["scale"]) if data.get("scale") else None,
        )

    def save(self, path: str) -> None:
        """Write the selections to ``path`` as JSON."""
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> Selections:
        """Read selections previously written by :meth:`save`."""
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


@dataclass
class Session:
    """Notebook-friendly container: video inputs + selections + reference frame.

    The interactive widgets populate ``self.selections`` in place; the pure
    tracking core reads from it. ``reference`` and ``file_names`` are built up
    as the pipeline runs.
    """

    dpath: str
    file: str | None = None
    start: int = 0
    end: int | None = None
    downsample: float = 1.0
    region_names: list[str] | None = None
    stretch: Stretch = field(default_factory=Stretch)
    altfile: str | None = None
    ftype: str | None = None  # batch: video file extension to glob for
    selections: Selections = field(default_factory=Selections)
    reference: np.ndarray | None = field(default=None, repr=False)
    file_names: list[str] = field(default_factory=list)

    @property
    def fpath(self) -> str:
        """Absolute path to the current video file."""
        if self.file is None:
            raise ValueError("session.file is not set")
        return os.path.join(os.path.normpath(self.dpath), self.file)


def _polydraw_polygons(data: dict | None) -> list[list[list[float]]]:
    """Convert a ``PolyDraw`` stream's ``{'xs': [...], 'ys': [...]}`` to vertex lists."""
    if not data:
        return []
    xs, ys = data.get("xs") or [], data.get("ys") or []
    return [
        [[float(x), float(y)] for x, y in zip(xpoly, ypoly, strict=False)]
        for xpoly, ypoly in zip(xs, ys, strict=False)
    ]
