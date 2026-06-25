"""The tracking core: estimate the animal's location per frame, build the track.

Algorithm (unchanged in spirit from the original ezTrack): difference each frame
from the reference, optionally bias toward the previous location, threshold to the
brightest ``threshold_pct`` of differences, and take the center of mass of what
remains. Pure with respect to HoloViews -- it only touches OpenCV/NumPy/pandas and
the plain config objects, so the whole pipeline runs and is tested headlessly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm.auto import tqdm

from .analyze import apply_scale
from .config import Session, TrackParams
from .regions import linearize, mask_array, roi_membership, transitions
from .video import open_capture, preprocess

__all__ = ["Located", "locate", "track"]


@dataclass
class Located:
    """Result of locating the animal in a single frame."""

    x: float
    y: float
    detected: bool
    dif: np.ndarray  # thresholded difference image (for previews)


def locate(
    frame: np.ndarray,
    reference: np.ndarray,
    params: TrackParams,
    prior: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> Located:
    """Locate the animal in a single preprocessed frame.

    ``frame`` and ``reference`` must already be grayscale/cropped to the same
    shape. ``prior`` is the previous ``(x, y)`` used for windowed weighting;
    ``mask`` is a boolean exclusion array. When nothing rises above threshold the
    animal is "not detected": the prior location is reused (or the frame center
    on the first frame) and ``detected`` is False.
    """
    if params.method == "abs":
        dif = np.abs(frame.astype(np.int16) - reference.astype(np.int16))
    elif params.method == "light":
        dif = frame.astype(np.int16) - reference.astype(np.int16)
    else:  # "dark" -- validated in TrackParams
        dif = reference.astype(np.int16) - frame.astype(np.int16)

    if mask is not None:
        dif[mask] = 0

    if prior is not None and params.window is not None:
        dif = _apply_window(dif, prior, params.window)

    dif = np.where(dif < np.percentile(dif, params.threshold_pct), 0, dif)

    if params.remove_wire:
        dif = _remove_wire(dif, params.wire_kernel)

    if dif.max() <= 0:  # nothing detected this frame
        h, w = frame.shape
        x, y = prior if prior is not None else (w / 2, h / 2)
        return Located(x=float(x), y=float(y), detected=False, dif=dif)

    cy, cx = ndimage.center_of_mass(dif)
    return Located(x=float(cx), y=float(cy), detected=True, dif=dif)


def _apply_window(dif: np.ndarray, prior: tuple[float, float], window) -> np.ndarray:
    """Down-weight differences far from ``prior`` so distractors don't grab the COM."""
    half = window.size // 2
    px, py = int(round(prior[0])), int(round(prior[1]))
    dif = dif + (-dif.min())  # shift so the lowest value is 0 before scaling
    weights = np.full(dif.shape, 1 - window.weight)
    weights[max(py - half, 0) : py + half, max(px - half, 0) : px + half] = 1
    return dif * weights


def _remove_wire(dif: np.ndarray, kernel: int) -> np.ndarray:
    """Morphological open to drop a thin (e.g. headstage-wire) signal; revert if it nukes all."""
    krn = np.ones((kernel, kernel), np.uint8)
    opened = cv2.morphologyEx(dif.astype(np.int16), cv2.MORPH_OPEN, krn)
    return dif if opened.sum() == 0 else opened


def track(session: Session, params: TrackParams, progress: bool = True) -> pd.DataFrame:
    """Track the animal across the session and return a per-frame dataframe.

    Columns: ``frame`` (absolute video frame number), ``x``, ``y``, ``detected``,
    ``distance_px`` (and a scaled ``distance_<unit>`` when a scale is set), plus
    one boolean column per ROI, an ``roi`` label, and an ``roi_transition`` flag
    when ROIs are present. Run parameters live in ``df.attrs``, not in every row.
    """
    if session.reference is None:
        raise ValueError("No reference frame. Call eztrack.reference_frame(session) first.")

    cap = open_capture(session.fpath)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, session.start)
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = int(session.end) if session.end is not None else cap_max
        n = end - session.start

        mask = mask_array(session.selections.mask, session.reference.shape)

        xs, ys, detected = np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool)
        prior: tuple[float, float] | None = None
        processed = 0
        for i in tqdm(range(n), disable=not progress, desc="tracking"):
            ret, frame = cap.read()
            if not ret:
                break
            loc = locate(preprocess(frame, session), session.reference, params, prior, mask)
            xs[i], ys[i], detected[i] = loc.x, loc.y, loc.detected
            prior = (loc.x, loc.y)
            processed += 1
    finally:
        cap.release()

    xs, ys, detected = xs[:processed], ys[:processed], detected[:processed]
    dist = np.zeros(processed)
    if processed > 1:
        dist[1:] = np.hypot(np.diff(xs), np.diff(ys))

    df = pd.DataFrame(
        {
            "frame": np.arange(session.start, session.start + processed),
            "x": xs,
            "y": ys,
            "detected": detected,
            "distance_px": dist,
        }
    )
    df.attrs.update(
        file=session.file,
        start=session.start,
        threshold_pct=params.threshold_pct,
        method=params.method,
        window=None
        if params.window is None
        else {"size": params.window.size, "weight": params.window.weight},
    )

    df = _add_rois(df, session)
    return apply_scale(df, session.selections.scale, "distance_px")


def _add_rois(df: pd.DataFrame, session: Session) -> pd.DataFrame:
    """Attach per-ROI membership columns, the linearized label, and transitions."""
    rois = session.selections.rois
    membership = roi_membership(
        rois, df["x"].to_numpy(), df["y"].to_numpy(), session.reference.shape
    )
    if not membership:
        return df
    for name, inside in membership.items():
        df[name] = inside
    df["roi"] = linearize(df[list(membership)])
    df["roi_transition"] = transitions(df["roi"])
    df.attrs["roi_coordinates"] = {"names": rois.names, "polygons": rois.polygons}
    return df
