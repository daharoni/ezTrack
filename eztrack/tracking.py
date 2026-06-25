"""Frame-by-frame location tracking: per-frame center-of-mass and the full session loop."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from .config import Session, TrackingParams
from .io import _preprocess_frame
from .roi import roi_linearize, roi_location, roi_transitions
from .scale import scale_distance

__all__ = ["locate", "track_location"]


def locate(cap, params: TrackingParams, session: Session, prior=None):
    """Locate the animal in the next frame of ``cap`` as a center-of-mass (y, x).

    Returns ``(ret, dif, com, frame)`` where ``dif`` is the thresholded
    difference image and ``com`` the center of mass. ``prior`` is the previous
    ``[y, x]`` used for windowed weighting when ``params.use_window``.
    """
    ret, frame = cap.read()

    if prior is not None and params.use_window:
        window_size = params.window_size // 2
        ymin, ymax = prior[0] - window_size, prior[0] + window_size
        xmin, xmax = prior[1] - window_size, prior[1] + window_size

    if not ret:
        return ret, None, None, frame

    frame = _preprocess_frame(frame, session)

    # find difference from reference
    if params.method == "abs":
        dif = np.absolute(frame - session.reference)
    elif params.method == "light":
        dif = frame - session.reference
    elif params.method == "dark":
        dif = session.reference - frame
    dif = dif.astype("int16")
    if session.mask is not None and session.mask.array is not None:
        dif[session.mask.array] = 0

    # apply window
    weight = 1 - params.window_weight
    if prior is not None and params.use_window:
        dif = dif + (dif.min() * -1)  # scale so lowest value is 0
        dif_weights = np.ones(dif.shape) * weight
        dif_weights[slice(ymin if ymin > 0 else 0, ymax), slice(xmin if xmin > 0 else 0, xmax)] = 1
        dif = dif * dif_weights

    # threshold differences and find center of mass for remaining values
    dif[dif < np.percentile(dif, params.loc_thresh)] = 0

    # remove influence of wire
    if params.rmv_wire:
        ksize = params.wire_krn
        kernel = np.ones((ksize, ksize), np.uint8)
        dif_wirermv = cv2.morphologyEx(dif, cv2.MORPH_OPEN, kernel)
        krn_violation = dif_wirermv.sum() == 0
        dif = dif if krn_violation else dif_wirermv
        if krn_violation:
            frm = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1 - session.start)
            print(f"WARNING: wire_krn too large. Reverting to rmv_wire=False for frame {frm}")

    com = ndimage.center_of_mass(dif)
    return ret, dif, com, frame


def track_location(session: Session, params: TrackingParams) -> pd.DataFrame:
    """Track the animal across the session and return a per-frame dataframe.

    Columns: video/parameter metadata, ``Frame``, ``X``, ``Y``,
    ``Distance_px`` (plus ROI membership/labels/transitions and a scaled
    distance column when those are configured on ``session``).
    """
    cap = cv2.VideoCapture(session.fpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, session.start)
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_max = int(session.end) if session.end is not None else cap_max

    x = np.zeros(cap_max - session.start)
    y = np.zeros(cap_max - session.start)
    d = np.zeros(cap_max - session.start)

    time.sleep(0.2)  # allow printing
    for f in tqdm(range(len(d))):
        if f > 0:
            yprior = np.around(y[f - 1]).astype(int)
            xprior = np.around(x[f - 1]).astype(int)
            ret, _dif, com, _frame = locate(cap, params, session, prior=[yprior, xprior])
        else:
            ret, _dif, com, _frame = locate(cap, params, session)

        if ret:
            y[f] = com[0]
            x[f] = com[1]
            if f > 0:
                d[f] = np.sqrt((y[f] - y[f - 1]) ** 2 + (x[f] - x[f - 1]) ** 2)
        else:
            # no frame detected: truncate to the frames we actually processed
            f = f - 1
            x = x[:f]
            y = y[:f]
            d = d[:f]
            break

    cap.release()
    time.sleep(0.2)  # allow printing
    print(f"total frames processed: {len(d)}\n")

    df = pd.DataFrame(
        {
            "File": session.file,
            "Location_Thresh": np.ones(len(d)) * params.loc_thresh,
            "Use_Window": str(params.use_window),
            "Window_Weight": np.ones(len(d)) * params.window_weight,
            "Window_Size": np.ones(len(d)) * params.window_size,
            "Start_Frame": np.ones(len(d)) * session.start,
            "Frame": np.arange(len(d)),
            "X": x,
            "Y": y,
            "Distance_px": d,
        }
    )

    df = roi_location(session, df)
    if session.region_names is not None:
        print("Defining transitions...")
        df["ROI_location"] = roi_linearize(df[session.region_names])
        df["ROI_transition"] = roi_transitions(df["ROI_location"])

    return scale_distance(session, df=df, column="Distance_px")
