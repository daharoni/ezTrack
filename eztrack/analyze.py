"""Post-tracking analysis: real-world scaling and binned summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Scale, Session

__all__ = ["apply_scale", "summarize"]


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

    region_cols = [c for c in (session.region_names or []) if c in df.columns]
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
        for col in region_cols:
            row[f"prop_{col}"] = window[col].mean() if len(window) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)
