"""Post-tracking analysis: real-world scaling and binned summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Scale, Session

__all__ = ["apply_scale", "summarize", "detection_rate"]


def apply_scale(df: pd.DataFrame, scale: Scale | None, column: str) -> pd.DataFrame:
    """Append a real-world-scaled copy of ``column`` (e.g. ``distance_cm``).

    Returns ``df`` unchanged when no usable scale is defined.
    """
    if scale is None or scale.factor is None:
        return df
    df[f"distance_{scale.unit}"] = df[column] * scale.factor
    return df


def summarize(
    df: pd.DataFrame, session: Session, bins: dict[str, tuple[int, int]] | None = None
) -> pd.DataFrame:
    """Summarize distance travelled and proportional time per ROI, per time bin.

    ``bins`` maps a bin name to an inclusive ``(start_frame, end_frame)`` range;
    when ``None`` a single whole-session ``"all"`` bin is used. Frame numbers are
    matched against the ``frame`` column.
    """
    if bins is None:
        bins = {"all": (int(df["frame"].min()), int(df["frame"].max()))}

    # The actual ROI columns come from the drawn regions' names; fall back to the
    # declared region_names for the (replayed/headless) case where rois aren't set.
    rois = session.selections.rois
    roi_names = (rois.names if rois else None) or session.region_names or []
    region_cols = [c for c in roi_names if c in df.columns]
    scale = session.selections.scale
    rows = []
    for name, (lo, hi) in bins.items():
        window = df[df["frame"].between(lo, hi)]
        row: dict[str, object] = {
            "file": session.file,
            "bin": name,
            "frame_range": (lo, hi),
            "distance_px": window["distance_px"].sum(),
        }
        if scale is not None and scale.factor is not None:
            row[f"distance_{scale.unit}"] = window["distance_px"].sum() * scale.factor
        row.update(detection_rate(window))
        for col in region_cols:
            row[f"prop_{col}"] = window[col].mean() if len(window) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def detection_rate(df: pd.DataFrame) -> dict[str, float]:
    """Frame counts and the percentage of frames the animal was actually detected.

    A failed frame (``detected`` is False) holds the previous position rather than
    a fresh estimate, so a low detection rate means much of the track is carried-
    forward rather than measured. Returns ``n_frames``, ``n_detected``,
    ``n_failed`` and ``pct_detected`` -- handy to print per video in the notebook
    and folded into every :func:`summarize` bin.
    """
    n = len(df)
    detected = int(df["detected"].sum()) if "detected" in df.columns else n
    return {
        "n_frames": n,
        "n_detected": detected,
        "n_failed": n - detected,
        "pct_detected": round(100 * detected / n, 1) if n else float("nan"),
    }
