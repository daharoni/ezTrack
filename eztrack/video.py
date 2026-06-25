"""Video I/O: open captures, preprocess frames, build the reference frame.

The reference (background) frame is the per-pixel median of frames sampled
across the session; because the animal moves, the median of enough frames is
the empty arena, which every frame is then differenced against.
"""

from __future__ import annotations

import fnmatch
import os

import cv2
import numpy as np

from .config import Session

__all__ = [
    "preprocess",
    "open_capture",
    "first_frame",
    "reference_frame",
    "discover_files",
    "check_decodable",
    "VideoError",
]


class VideoError(RuntimeError):
    """Raised when a video cannot be opened or decoded."""


def open_capture(path: str) -> cv2.VideoCapture:
    """Open ``path`` for reading, raising :class:`VideoError` if it cannot be opened."""
    if not os.path.isfile(path):
        raise VideoError(f"{path} not found. Check the directory and file name are correct.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"Could not open {path}. Is it a supported video format?")
    return cap


def preprocess(frame: np.ndarray, session: Session, *, crop: bool = True) -> np.ndarray:
    """Grayscale -> optional downsample -> optional crop.

    Every read path (reference, tracking, playback) funnels through here so the
    transform is defined exactly once.
    """
    out = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if session.downsample < 1:
        out = cv2.resize(
            out,
            (int(out.shape[1] * session.downsample), int(out.shape[0] * session.downsample)),
            interpolation=cv2.INTER_NEAREST,
        )
    if crop and session.selections.crop is not None:
        out = session.selections.crop.apply(out)
    return out


def first_frame(session: Session, *, crop: bool = False) -> np.ndarray:
    """Read and preprocess the frame at ``session.start``."""
    cap = open_capture(session.fpath)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, session.start)
        ret, frame = cap.read()
        if not ret:
            raise VideoError(f"Could not read frame {session.start} of {session.fpath}.")
        return preprocess(frame, session, crop=crop)
    finally:
        cap.release()


def video_info(session: Session) -> dict[str, float]:
    """Return basic capture properties (frame count, fps, height, width)."""
    cap = open_capture(session.fpath)
    try:
        return {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        }
    finally:
        cap.release()


def reference_frame(
    session: Session,
    num_frames: int = 100,
    altfile: bool = False,
    frames: list[int] | None = None,
) -> np.ndarray:
    """Build the reference frame as the per-pixel median of sampled frames.

    Mutates and returns ``session.reference``. Pass ``altfile=True`` to sample
    an animal-free companion video (``session.altfile``), or ``frames`` to pick
    specific frame numbers.
    """
    path = os.path.join(os.path.normpath(session.dpath), session.altfile) if altfile else None
    cap = open_capture(path or session.fpath)
    try:
        cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = int(session.end) if session.end is not None else cap_max

        if frames is None:
            # evenly spaced (monotonic) sample -> sequential-ish seeks
            frames = [int(f) for f in np.linspace(session.start, end - 1, num=num_frames)]

        sample = None
        for idx, framenum in enumerate(frames):
            grabbed, frame = _read_at(cap, framenum)
            while not grabbed:  # skip undecodable frames by re-sampling
                framenum = int(np.random.randint(session.start, end))
                grabbed, frame = _read_at(cap, framenum)
            processed = preprocess(frame, session)
            if sample is None:
                sample = np.zeros((len(frames), *processed.shape), dtype=processed.dtype)
            sample[idx] = processed
    finally:
        cap.release()

    session.reference = np.median(sample, axis=0)
    return session.reference


def _read_at(cap: cv2.VideoCapture, framenum: int) -> tuple[bool, np.ndarray | None]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
    ret, frame = cap.read()
    return ret, frame


def discover_files(session: Session) -> list[str]:
    """Populate and return ``session.file_names`` (files of type ``session.ftype``)."""
    if not os.path.isdir(session.dpath):
        raise VideoError(f"{session.dpath} not found. Check that the directory is correct.")
    names = sorted(os.listdir(session.dpath))
    session.file_names = fnmatch.filter(names, "*." + (session.ftype or "*"))
    return session.file_names


def check_decodable(
    session: Session, allowed_fraction: float = 0.01, frames_checked: int = 300
) -> None:
    """Raise :class:`VideoError` if too many of the first frames fail to decode."""
    cap = open_capture(session.fpath)
    try:
        frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        allowed = int(frames_checked * allowed_fraction)
        failed = sum(not cap.read()[0] for _ in range(frames_checked))
    finally:
        cap.release()
    if failed > allowed:
        pct = (failed / frames_checked) * 100 if frames_checked else 0
        raise VideoError(
            f"Video compression not supported: ~{pct:.0f}% of frames are undecodable "
            "(p-frames or blank). Consider converting the video."
        )
