"""Binned / whole-session summaries of distance travelled and time in each region."""

from __future__ import annotations

import pandas as pd

from .config import Session
from .scale import scale_distance

__all__ = ["summarize_location"]


def summarize_location(
    location: pd.DataFrame, session: Session, bin_dict: dict | None = None
) -> pd.DataFrame:
    """Summarize distance travelled and proportional time in each ROI per time bin.

    ``bin_dict`` maps a bin name to a ``(start_frame, end_frame)`` tuple; when
    ``None`` a single whole-session ("all") bin is used.
    """
    avg_dict = {"all": (location["Frame"].min(), location["Frame"].max())}
    bin_dict = bin_dict if bin_dict is not None else avg_dict

    bins = pd.Series(bin_dict).rename("range(f)").reset_index().rename(columns={"index": "bin"})
    bins["Distance_px"] = bins["range(f)"].apply(
        lambda r: location[location["Frame"].between(*r)]["Distance_px"].sum()
    )
    if session.region_names is not None:
        bins_reg = bins["range(f)"].apply(
            lambda r: location[location["Frame"].between(*r)][session.region_names].mean()
        )
        bins = bins.join(bins_reg)
        drp_cols = ["Distance_px", "Frame", "X", "Y"] + session.region_names
    else:
        drp_cols = ["Distance_px", "Frame", "X", "Y"]
    bins = pd.merge(
        location.drop(drp_cols, axis="columns"), bins, left_index=True, right_index=True
    )

    return scale_distance(session, df=bins, column="Distance_px")
