"""Play back tracked video with the animal's estimated location marked."""

from __future__ import annotations

import os
import time
from io import BytesIO

import cv2
import pandas as pd
import PIL.Image
from IPython.display import Image, clear_output, display

from .config import DisplayParams, Session
from .io import _preprocess_frame, crop_frame

__all__ = ["play_video", "play_video_ext"]


def _opencv_is_headless() -> bool:
    """Return True if the installed OpenCV was built without GUI (HighGUI) support.

    opencv-python-headless still exports ``cv2.imshow`` (so ``hasattr`` can't
    tell the builds apart) but raises ``cv2.error`` when it is called. The build
    report is the reliable signal: headless builds print ``GUI: NONE``.
    """
    for line in cv2.getBuildInformation().splitlines():
        head, _, val = line.partition(":")
        if head.strip() == "GUI" and val.strip().upper() == "NONE":
            return True
    return False


def _display_image(frame, fps: int, resize) -> None:
    """Render a single grayscale frame inline in the notebook, then pace to ``fps``."""
    img = PIL.Image.fromarray(frame, "L")
    img = img.resize(size=resize) if resize else img
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    display(Image(data=buffer.getvalue()))
    time.sleep(1 / fps)
    clear_output(wait=True)


def _open_writer(session: Session) -> cv2.VideoWriter:
    """Open a grayscale AVI writer (``video_output.avi``) sized to the cropped frame."""
    cap = cv2.VideoCapture(session.fpath)
    _ret, frame = cap.read()
    frame = _preprocess_frame(frame, session)
    cap.release()
    height, width = int(frame.shape[0]), int(frame.shape[1])
    return cv2.VideoWriter(
        os.path.join(os.path.normpath(session.dpath), "video_output.avi"),
        0,  # fourcc 0: uncompressed; writes at up to 20 fps
        20.0,
        (width, height),
        isColor=False,
    )


def play_video(session: Session, display_params: DisplayParams, location: pd.DataFrame) -> None:
    """Play a segment back inline in the notebook with the tracked position marked."""
    cap = cv2.VideoCapture(session.fpath)
    writer = _open_writer(session) if display_params.save_video else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, session.start + display_params.start)
    for f in range(display_params.start, display_params.stop):
        ret, frame = cap.read()
        if ret:
            frame = _preprocess_frame(frame, session)
            markposition = (int(location["X"][f]), int(location["Y"][f]))
            cv2.drawMarker(img=frame, position=markposition, color=255)
            _display_image(frame, display_params.fps, display_params.resize)
            if writer is not None:
                writer.write(frame)
        else:
            print("warning. failed to get video frame")

    print("Done playing segment")
    if writer is not None:
        writer.release()


def play_video_ext(
    session: Session, display_params: DisplayParams, location: pd.DataFrame, crop=None
) -> None:
    """Play a segment in an external OpenCV window (needs a GUI build of OpenCV).

    ezTrack pins ``opencv-python-headless`` (matching minian), so this fails fast
    with a friendly error unless a GUI OpenCV is installed; use :func:`play_video`
    for the inline notebook player. ``crop`` overrides ``session.crop`` if given.
    """
    if _opencv_is_headless():
        raise RuntimeError(
            "play_video_ext needs a GUI build of OpenCV, but ezTrack installs "
            "opencv-python-headless. Use play_video() to view frames inline in the "
            "notebook, or `pip install opencv-python` for an external window."
        )

    cap = cv2.VideoCapture(session.fpath)
    writer = _open_writer(session) if display_params.save_video else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, session.start + display_params.start)
    rate = int(1000 / display_params.fps)
    for f in range(display_params.start, display_params.stop):
        ret, frame = cap.read()
        if ret:
            frame = _preprocess_frame(frame, session, crop=False)
            frame = crop_frame(frame, crop)
            markposition = (int(location["X"][f]), int(location["Y"][f]))
            cv2.drawMarker(img=frame, position=markposition, color=255)
            cv2.imshow("preview", frame)
            cv2.waitKey(rate)
            if writer is not None:
                writer.write(frame)
        else:
            print("warning. failed to get video frame")

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if writer is not None:
        writer.release()
