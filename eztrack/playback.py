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


def _marker_track(df: pd.DataFrame, seg: range) -> pd.DataFrame:
    """Marker ``x``/``y`` for each absolute frame in ``seg``, keyed by frame number.

    Looking up by frame (not row position) keeps the marker aligned with the video
    when the track is sparse (temporal downsampling) or filtered; skipped frames
    are forward-filled to the last known position, and frames before the first
    tracked one are left NaN (no marker drawn).
    """
    return df.set_index("frame")[["x", "y"]].sort_index().reindex(seg, method="ffill")


def _play(session: Session, params: PlayParams, df: pd.DataFrame, sink) -> None:
    """Drive the shared playback loop, sending each marked frame to ``sink``.

    ``sink(frame)`` displays one frame; ``sink.close()`` tears the display down.
    The only difference between the inline and windowed players is the sink.

    The marker is looked up by **frame number**, not row position, so the dot
    stays aligned with the video even when the track is sparse (temporal
    downsampling) or has been filtered. Frames with no tracked row (e.g. those
    skipped by ``temporal_downsample``) reuse the last known position.
    """
    seg = range(session.start + params.start, session.start + params.stop)
    pos = _marker_track(df, seg)

    cap = open_capture(session.fpath)
    writer = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start)
        for abs_frame in seg:
            ret, frame = cap.read()
            if not ret:
                print("warning: failed to get video frame")
                continue
            frame = preprocess(frame, session)
            if writer is None and params.save:
                writer = _writer(session, frame.shape)
            x, y = pos.loc[abs_frame, "x"], pos.loc[abs_frame, "y"]
            if pd.notna(x):
                cv2.drawMarker(frame, (int(x), int(y)), color=255)
            sink(frame)
            if writer is not None:
                writer.write(frame)
    finally:
        cap.release()
        sink.close()
        if writer is not None:
            writer.release()


class _InlineSink:
    """Render frames inline in the notebook, paced to ``fps``."""

    def __init__(self, fps: int, resize):
        self.fps = fps
        self.resize = resize

    def __call__(self, frame) -> None:
        img = PIL.Image.fromarray(frame, "L")
        if self.resize:
            img = img.resize(size=self.resize)
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        display(Image(data=buffer.getvalue()))
        time.sleep(1 / self.fps)
        clear_output(wait=True)

    def close(self) -> None:
        print("Done playing segment")


class _WindowSink:
    """Render frames in an external OpenCV window, paced to ``fps``."""

    def __init__(self, fps: int):
        self.rate = int(1000 / fps)

    def __call__(self, frame) -> None:
        cv2.imshow("preview", frame)
        cv2.waitKey(self.rate)

    def close(self) -> None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def play_inline(session: Session, params: PlayParams, df: pd.DataFrame) -> None:
    """Play a segment inline in the notebook with the tracked position marked."""
    _play(session, params, df, _InlineSink(params.fps, params.resize))


def play_window(session: Session, params: PlayParams, df: pd.DataFrame) -> None:
    """Play a segment in an external OpenCV window (requires a GUI OpenCV build)."""
    if opencv_is_headless():
        raise RuntimeError(
            "play_window needs a GUI build of OpenCV, but ezTrack installs "
            "opencv-python-headless. Use play_inline() for the notebook player, "
            "or `pip install opencv-python` for an external window."
        )
    _play(session, params, df, _WindowSink(params.fps))
