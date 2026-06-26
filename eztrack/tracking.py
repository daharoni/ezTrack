"""The tracking core: estimate the animal's location per frame, build the track.

Algorithm (unchanged in spirit from the original ezTrack): difference each frame
from the reference, optionally bias toward the previous location, threshold to the
brightest ``threshold_pct`` of differences, and take the center of mass of what
remains. Pure with respect to HoloViews -- it only touches OpenCV/NumPy/pandas and
the plain config objects, so the whole pipeline runs and is tested headlessly.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm.auto import tqdm

from .analyze import apply_scale, detection_rate
from .config import Session, TrackParams
from .regions import linearize, mask_array, roi_membership, transitions
from .video import downscale, open_capture, preprocess

__all__ = ["Located", "locate", "track", "hampel_filter", "save_tracking", "load_tracking"]

# track() warns when fewer than this percent of frames actually detected the
# animal -- a quiet low-detection track silently under-measures distance.
LOW_DETECTION_WARN_PCT = 90.0


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
    ref16, frame16 = reference.astype(np.int16), frame.astype(np.int16)
    raw_threshold = params.threshold_abs is not None and params.threshold_on == "raw"
    if raw_threshold:
        # Threshold the raw pixel value directly, ignoring the baseline. The clip
        # both selects (pixels past the cutoff) and weights (by how far past), so
        # no further cutoff is applied below.
        t = params.threshold_abs
        if params.method == "dark":
            dif = np.clip(t - frame16, 0, None)  # pixels darker than t, weighted by darkness
        else:  # "light" -- "abs" rejected in TrackParams for raw mode
            dif = np.clip(frame16 - t, 0, None)  # pixels brighter than t
    elif params.method == "abs":
        dif = np.abs(frame16 - ref16)
    elif params.method == "light":
        dif = frame16 - ref16
    else:  # "dark" -- validated in TrackParams
        dif = ref16 - frame16

    if mask is not None:
        dif[mask] = 0

    # An absolute difference cutoff is in raw intensity units, so apply it *before*
    # windowing rescales the difference. The signed ``dif`` already encodes the
    # method's direction, so one cutoff covers dark (darker by >= cutoff), light
    # (brighter) and abs (changed either way). (Raw mode is already thresholded by
    # its clip above.)
    if params.threshold_abs is not None and not raw_threshold:
        dif = np.where(dif < params.threshold_abs, 0, dif)

    if prior is not None and params.window is not None:
        dif = _apply_window(dif, prior, params.window)

    # The percentile cutoff is applied after windowing, so the weighting shapes
    # which pixels survive (down-weighted far pixels fall below the percentile).
    if params.threshold_abs is None:
        dif = np.where(dif < np.percentile(dif, params.threshold_pct), 0, dif)

    if params.denoise:
        dif = _denoise(dif, params.denoise_kernel)

    com = _center_of_mass(dif)
    if com is None:  # nothing rose above threshold this frame
        h, w = frame.shape
        x, y = prior if prior is not None else (w / 2, h / 2)
        return Located(x=float(x), y=float(y), detected=False, dif=dif)

    return Located(x=com[0], y=com[1], detected=True, dif=dif)


def _apply_window(dif: np.ndarray, prior: tuple[float, float], window) -> np.ndarray:
    """Down-weight differences far from ``prior`` so distractors don't grab the COM.

    Returns float32 (rather than float64): for an estimate that only feeds a
    thresholded center-of-mass, single precision is more than enough and roughly
    halves the per-frame cost of this and the percentile that follows.
    """
    half = window.size // 2
    px, py = int(round(prior[0])), int(round(prior[1]))
    shifted = (dif - dif.min()).astype(np.float32)  # shift so the lowest value is 0
    weights = np.full(dif.shape, np.float32(1 - window.weight), dtype=np.float32)
    weights[max(py - half, 0) : py + half, max(px - half, 0) : px + half] = 1
    return shifted * weights


def _center_of_mass(dif: np.ndarray) -> tuple[float, float] | None:
    """Intensity-weighted ``(x, y)`` of the surviving pixels, or ``None`` if none.

    Equivalent to :func:`scipy.ndimage.center_of_mass` but computed only over the
    nonzero pixels -- after thresholding that is a tiny fraction of the frame, so
    this is several times faster than scanning the whole array.
    """
    ys, xs = np.nonzero(dif)
    if xs.size == 0:
        return None
    weights = dif[ys, xs].astype(np.float64)
    total = weights.sum()
    if total == 0:
        return None
    return (xs * weights).sum() / total, (ys * weights).sum() / total


def _denoise(dif: np.ndarray, kernel: int) -> np.ndarray:
    """Morphologically open (erode then dilate) the thresholded mask.

    Erosion removes any feature thinner than ``kernel`` px (specks, a thin
    headstage wire); the following dilation grows the survivors back to roughly
    their original size, so the animal's bulk is preserved. If the kernel is
    larger than the animal the opening erases everything; the mask is left empty
    so the frame is reported *not detected* (the prior position is reused). We do
    not fall back to the un-opened mask -- that would resurrect the very specks
    the filter removed and let them grab the center of mass, so a larger kernel
    would paradoxically find *more* noise.
    """
    krn = np.ones((kernel, kernel), np.uint8)
    return cv2.morphologyEx(dif.astype(np.int16), cv2.MORPH_OPEN, krn)


def track(session: Session, params: TrackParams, progress: bool = True) -> pd.DataFrame:
    """Track the animal across the session and return a per-frame dataframe.

    Columns: ``frame`` (absolute video frame number), ``x``, ``y``, ``detected``,
    ``distance_px`` (and a scaled ``distance_<unit>`` when a scale is set), plus
    one boolean column per ROI, an ``roi`` label, and an ``roi_transition`` flag
    when ROIs are present. Run parameters live in ``df.attrs``, not in every row.

    ``distance_px`` is counted only between consecutive *detected* frames (see
    :func:`_step_distance`); a low overall detection rate therefore under-counts
    distance and triggers a warning. Timing is intentionally left to downstream:
    the ``frame`` column is the join key against a per-frame timestamp file
    (e.g. ``timestamps.csv``) -- ezTrack never infers time from the video's fps.

    ``session.spatial_downsample`` and ``session.temporal_downsample`` (both
    reduction factors, 1 = none) only speed things up; outputs stay in the video's
    **original** space. Spatial tracking happens on a shrunken frame with positions
    mapped back to full-res pixels. Temporal tracking runs only every Nth frame:
    those frames are the rows of the output (no positions are invented for the
    skipped frames) and the ``frame`` column carries the real frame numbers, so
    downstream can resample if it wants to.
    """
    if session.reference is None:
        raise ValueError("No reference frame. Call eztrack.reference_frame(session) first.")

    ds = session.spatial_downsample
    stride = session.temporal_downsample
    reference = downscale(session.reference, ds)
    full_mask = mask_array(session.selections.mask, session.reference.shape)
    mask = None if full_mask is None else downscale(full_mask.astype(np.uint8), ds).astype(bool)

    cap = open_capture(session.fpath)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, session.start)
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = int(session.end) if session.end is not None else cap_max
        n = max(end - session.start, 0)

        frames, axs, ays, adet = [], [], [], []
        prior: tuple[float, float] | None = None
        for i in tqdm(range(n), disable=not progress, desc="tracking"):
            # Track every stride-th frame; skip-decode the rest with grab() (no
            # full decode), so skipped frames cost almost nothing.
            if i % stride != 0:
                if not cap.grab():
                    break
                continue
            ret, frame = cap.read()
            if not ret:
                break
            small = downscale(preprocess(frame, session), ds)
            # locate works in downsampled space; map the prior in and the result
            # back out by the spatial factor so the caller sees full-res pixels.
            small_prior = None if prior is None else (prior[0] / ds, prior[1] / ds)
            loc = locate(small, reference, params, small_prior, mask)
            x, y = loc.x * ds, loc.y * ds
            frames.append(session.start + i)
            axs.append(x)
            ays.append(y)
            adet.append(loc.detected)
            prior = (x, y)
    finally:
        cap.release()

    xs, ys, detected = np.asarray(axs), np.asarray(ays), np.asarray(adet, dtype=bool)
    df = pd.DataFrame(
        {
            "frame": np.asarray(frames, dtype=int),
            "x": xs,
            "y": ys,
            "detected": detected,
            "distance_px": _step_distance(xs, ys, detected),
        }
    )
    _warn_if_low_detection(df)
    df.attrs.update(
        file=session.file,
        start=session.start,
        threshold_pct=params.threshold_pct,
        method=params.method,
        window=None
        if params.window is None
        else {"size": params.window.size, "weight": params.window.weight},
        threshold_abs=params.threshold_abs,
        threshold_on=params.threshold_on,
        denoise=params.denoise_kernel if params.denoise else None,
        spatial_downsample=ds,
        temporal_downsample=stride,
    )

    df = _add_rois(df, session)
    return apply_scale(df, session.selections.scale, "distance_px")


def _step_distance(xs: np.ndarray, ys: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Per-frame travelled distance, counted only between two *detected* frames.

    A step is the move from the previous row to this one. When either end is a
    failed frame (``detected`` False) the position there was carried forward, not
    measured, so the step is unknown -- we record 0 rather than the spurious
    freeze-then-jump it would otherwise produce. Distance is therefore an honest
    *measured* path length; pair it with the detection rate to see how much of the
    track it covers. (Filling the gaps is deliberately left to downstream code.)
    """
    dist = np.zeros(xs.size)
    if xs.size > 1:
        step = np.hypot(np.diff(xs), np.diff(ys))
        both_detected = detected[1:] & detected[:-1]
        dist[1:] = np.where(both_detected, step, 0.0)
    return dist


def _warn_if_low_detection(df: pd.DataFrame) -> None:
    """Emit a warning when the animal was detected in too few frames to trust."""
    pct = detection_rate(df)["pct_detected"]
    if pct < LOW_DETECTION_WARN_PCT:
        warnings.warn(
            f"Animal detected in only {pct}% of frames "
            f"(< {LOW_DETECTION_WARN_PCT}%). Distance is measured between detected "
            "frames only, so it under-counts here; review threshold_preview / params.",
            stacklevel=3,
        )


def _json_default(obj):
    """Make the numpy values that live in ``df.attrs`` JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def save_tracking(df: pd.DataFrame, csv_path: str) -> str:
    """Write a track to ``csv_path`` plus a ``.json`` sidecar of its run metadata.

    ``DataFrame.to_csv`` keeps only the per-frame columns, so the provenance
    :func:`track` records in ``df.attrs`` (the parameters, the ROI polygons, the
    down-sampling factors) would be silently dropped -- leaving a CSV you can no
    longer reproduce or even interpret. This writes that metadata alongside the
    CSV as ``<name>.json`` so a saved track is self-describing. Returns the
    sidecar path; read both back together with :func:`load_tracking`.
    """
    df.to_csv(csv_path, index=False)
    meta_path = os.path.splitext(csv_path)[0] + ".json"
    with open(meta_path, "w") as fh:
        json.dump(dict(df.attrs), fh, indent=2, default=_json_default)
    return meta_path


def load_tracking(csv_path: str) -> pd.DataFrame:
    """Read a track written by :func:`save_tracking`, restoring ``df.attrs``.

    Reads the per-frame CSV and merges its ``.json`` sidecar (when present) back
    into ``df.attrs``, so the loaded frame carries the same run metadata it had in
    memory. Returns the dataframe unchanged if no sidecar is found.
    """
    df = pd.read_csv(csv_path)
    meta_path = os.path.splitext(csv_path)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            df.attrs.update(json.load(fh))
    return df


def hampel_filter(
    location: pd.DataFrame, session: Session, window: int = 7, sigma: float = 3.0
) -> pd.DataFrame:
    """Remove position outliers from a completed track (run *after* :func:`track`).

    Slides a window over the tracked ``x``/``y`` and replaces any point lying more
    than ``sigma`` scaled-MADs from the local median position with that median --
    killing the occasional bad-frame jump without smoothing genuine movement.
    ``distance_px``, the ROI columns, and the scaled distance are all recomputed
    from the cleaned positions. Returns a new dataframe; the input is unchanged.

    ``window`` is counted in **tracked rows**, not original frames: with
    ``temporal_downsample=N`` each row is N frames apart, so the window spans
    ``(2*window+1) * N`` original frames of real time. Spatial downsampling does
    not affect it (still one row per frame).
    """
    if window < 1 or sigma <= 0:
        raise ValueError(f"window must be >= 1 and sigma > 0, got window={window}, sigma={sigma}")

    df = location.copy()
    x, y, _ = _hampel_xy(df["x"].to_numpy(), df["y"].to_numpy(), window, sigma)
    df["x"], df["y"] = x, y

    detected = df["detected"].to_numpy() if "detected" in df.columns else np.ones(len(df), bool)
    df["distance_px"] = _step_distance(x, y, detected)

    df = _add_rois(df, session)
    return apply_scale(df, session.selections.scale, "distance_px")


def _hampel_xy(
    x: np.ndarray, y: np.ndarray, window: int, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint 2-D Hampel filter on a position track.

    A point is an outlier when its ``(x, y)`` lies more than ``sigma`` scaled-MADs
    from the **median position** of the ``2*window+1`` frames centered on it, where
    the spread is the median Euclidean distance of those frames from that median
    (so it scales with how much the animal is actually moving). Outliers have
    *both* coordinates replaced by the local median position -- x and y are treated
    as one point, never filtered independently. Returns ``(x, y, outlier_mask)``.
    """
    win = 2 * window + 1
    pad = np.full(window, np.nan)
    xw = sliding_window_view(np.concatenate([pad, x, pad]), win)  # (n, win) per-frame windows
    yw = sliding_window_view(np.concatenate([pad, y, pad]), win)

    mx = np.nanmedian(xw, axis=1)  # local median position
    my = np.nanmedian(yw, axis=1)
    spread = np.nanmedian(np.hypot(xw - mx[:, None], yw - my[:, None]), axis=1)
    mad = 1.4826 * spread

    deviation = np.hypot(x - mx, y - my)  # how far this frame is from its local median
    outlier = (deviation > sigma * mad) & (mad > 0)
    return np.where(outlier, mx, x), np.where(outlier, my, y), outlier


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
