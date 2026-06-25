"""Play back a tracked segment with the estimated location marked."""

from __future__ import annotations

import os
import time
from io import BytesIO

import cv2
import pandas as pd
import PIL.Image
from IPython.display import Image, clear_output, display

from .config import PlayParams, Session
from .video import open_capture, preprocess

__all__ = ["play_inline", "play_window", "opencv_is_headless"]


def opencv_is_headless() -> bool:
    """True if the installed OpenCV was built without GUI (HighGUI) support.

    ``opencv-python-headless`` still exports ``cv2.imshow`` (so ``hasattr`` can't
    tell the builds apart) but raises when it is called. The build report is the
    reliable signal: headless builds print ``GUI: NONE``.
    """
    for line in cv2.getBuildInformation().splitlines():
        head, _, val = line.partition(":")
        if head.strip() == "GUI" and val.strip().upper() == "NONE":
            return True
    return False


def _writer(session: Session, shape: tuple[int, int]) -> cv2.VideoWriter:
    height, width = shape
    return cv2.VideoWriter(
        os.path.join(os.path.normpath(session.dpath), "video_output.avi"),
        0,  # fourcc 0: uncompressed
        20.0,
        (width, height),
        isColor=False,
    )


def play_inline(session: Session, params: PlayParams, df: pd.DataFrame) -> None:
    """Play a segment inline in the notebook with the tracked position marked."""
    cap = open_capture(session.fpath)
    writer = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, session.start + params.start)
        for f in range(params.start, params.stop):
            ret, frame = cap.read()
            if not ret:
                print("warning: failed to get video frame")
                continue
            frame = preprocess(frame, session)
            if writer is None and params.save:
                writer = _writer(session, frame.shape)
            cv2.drawMarker(frame, (int(df["x"].iloc[f]), int(df["y"].iloc[f])), color=255)
            _show(frame, params.fps, params.resize)
            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print("Done playing segment")


def play_window(session: Session, params: PlayParams, df: pd.DataFrame) -> None:
    """Play a segment in an external OpenCV window (requires a GUI OpenCV build)."""
    if opencv_is_headless():
        raise RuntimeError(
            "play_window needs a GUI build of OpenCV, but ezTrack installs "
            "opencv-python-headless. Use play_inline() for the notebook player, "
            "or `pip install opencv-python` for an external window."
        )
    cap = open_capture(session.fpath)
    writer = None
    rate = int(1000 / params.fps)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, session.start + params.start)
        for f in range(params.start, params.stop):
            ret, frame = cap.read()
            if not ret:
                print("warning: failed to get video frame")
                continue
            frame = preprocess(frame, session)
            if writer is None and params.save:
                writer = _writer(session, frame.shape)
            cv2.drawMarker(frame, (int(df["x"].iloc[f]), int(df["y"].iloc[f])), color=255)
            cv2.imshow("preview", frame)
            cv2.waitKey(rate)
            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if writer is not None:
            writer.release()


def _show(frame, fps: int, resize) -> None:
    img = PIL.Image.fromarray(frame, "L")
    if resize:
        img = img.resize(size=resize)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    display(Image(data=buffer.getvalue()))
    time.sleep(1 / fps)
    clear_output(wait=True)
