"""Video loading, frame preprocessing, reference-frame generation, and file discovery."""

from __future__ import annotations

import fnmatch
import os

import cv2
import holoviews as hv
import numpy as np
from holoviews import streams

from .config import Session

hv.extension("bokeh")

__all__ = [
    "crop_frame",
    "load_and_crop",
    "make_reference",
    "batch_load_files",
    "check_p_frames",
]


def crop_frame(frame: np.ndarray, crop=None) -> np.ndarray:
    """Crop ``frame`` to the box held in the HoloViews ``crop`` BoxEdit stream.

    Returns the frame unchanged when no (valid) crop selection is present.
    """
    try:
        xs = [crop.data["x0"][0], crop.data["x1"][0]]
        ys = [crop.data["y0"][0], crop.data["y1"][0]]
        fxmin, fxmax = int(min(xs)), int(max(xs))
        fymin, fymax = int(min(ys)), int(max(ys))
        return frame[fymin:fymax, fxmin:fxmax]
    except (AttributeError, KeyError, IndexError, TypeError):
        return frame


def _preprocess_frame(frame: np.ndarray, session: Session, *, crop: bool = True) -> np.ndarray:
    """Convert a raw BGR frame to grayscale, optionally downsample, optionally crop.

    Shared by every read path (reference, tracking, playback) so the
    grayscale -> ``dsmpl`` resize -> crop sequence lives in one place.
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if session.dsmpl < 1:
        frame = cv2.resize(
            frame,
            (int(frame.shape[1] * session.dsmpl), int(frame.shape[0] * session.dsmpl)),
            interpolation=cv2.INTER_NEAREST,
        )
    if crop:
        frame = crop_frame(frame, session.crop)
    return frame


def _image(arr: np.ndarray, session: Session, title: str, **opts) -> hv.Image:
    """Build a stretched, gray, y-inverted HoloViews Image of ``arr``."""
    img = hv.Image((np.arange(arr.shape[1]), np.arange(arr.shape[0]), arr))
    img.opts(
        width=int(arr.shape[1] * session.stretch.width),
        height=int(arr.shape[0] * session.stretch.height),
        invert_yaxis=True,
        cmap="gray",
        toolbar="below",
        title=title,
        **opts,
    )
    return img


def load_and_crop(
    session: Session,
    cropmethod: str | None = None,
    fstfile: bool = False,
    accept_p_frames: bool = False,
) -> hv.Element:
    """Load the first frame and (optionally) expose a box crop tool.

    Mutates ``session``: sets ``fpath``, ``f0`` (uncropped first frame), and
    ``crop`` (a BoxEdit stream when ``cropmethod='Box'``, else ``None``).
    Returns the HoloViews element to display. In batch mode (``fstfile=True``)
    the first file in ``session.file_names`` is used.
    """
    if fstfile:
        session.file = session.file_names[0]

    session.fpath = os.path.join(os.path.normpath(session.dpath), session.file)
    if not os.path.isfile(session.fpath):
        raise FileNotFoundError(
            f"{session.fpath} not found. Check that directory and file names are correct"
        )
    print(f"file: {session.fpath}")
    cap = cv2.VideoCapture(session.fpath)

    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total frames: {cap_max}")
    print(f"nominal fps: {cap.get(cv2.CAP_PROP_FPS)}")
    print(
        f"dimensions (h x w): {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))},"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
    )

    if accept_p_frames is False:
        check_p_frames(cap)

    cap.set(cv2.CAP_PROP_POS_FRAMES, session.start)
    _ret, frame = cap.read()
    frame = _preprocess_frame(frame, session, crop=False)
    session.f0 = frame
    cap.release()

    image = _image(frame, session, "First Frame.  Crop if Desired", colorbar=True)

    if cropmethod is None:
        image.opts(title="First Frame")
        session.crop = None
        return image

    if cropmethod == "Box":
        box = hv.Polygons([])
        box.opts(alpha=0.5)
        session.crop = streams.BoxEdit(source=box, num_objects=1)
        return image * box

    return image


def make_reference(
    session: Session,
    num_frames: int = 100,
    altfile: bool = False,
    fstfile: bool = False,
    frames=None,
) -> hv.Image:
    """Build a reference frame as the per-pixel median of a subset of frames.

    Removes the (moving) animal from the scene. Mutates ``session.reference``
    and returns its HoloViews Image. Pass ``altfile=True`` to use
    ``session.altfile`` instead of the analysed video, or ``frames`` to choose
    specific frame numbers.
    """
    if fstfile:
        session.file = session.file_names[0]
    vname = (session.altfile or "") if altfile else session.file
    fpath = os.path.join(os.path.normpath(session.dpath), vname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError("File not found. Check that directory and file names are correct.")
    cap = cv2.VideoCapture(fpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _ret, frame = cap.read()
    frame = _preprocess_frame(frame, session)
    h, w = frame.shape[0], frame.shape[1]
    cap_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_max = int(session.end) if session.end is not None else cap_max

    if frames is None:
        frames = np.linspace(start=session.start, stop=cap_max, num=num_frames)
    else:
        num_frames = len(frames)  # keep num_frames in sync with a passed list

    collection = np.zeros((num_frames, h, w))
    for idx, framenum in enumerate(frames):
        grabbed = False
        while not grabbed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
            ret, frame = cap.read()
            if ret:
                collection[idx, :, :] = _preprocess_frame(frame, session)
                grabbed = True
            else:
                framenum = np.random.randint(session.start, cap_max, 1)[0]
    cap.release()

    session.reference = np.median(collection, axis=0)
    return _image(session.reference, session, "Reference Frame", colorbar=True)


def batch_load_files(session: Session) -> None:
    """Populate ``session.file_names`` with files of type ``session.ftype`` in ``dpath``."""
    if not os.path.isdir(session.dpath):
        raise FileNotFoundError(f"{session.dpath} not found. Check that directory is correct")
    names = sorted(os.listdir(session.dpath))
    session.file_names = fnmatch.filter(names, "*." + session.ftype)


def check_p_frames(cap, p_prop_allowed: float = 0.01, frames_checked: int = 300) -> None:
    """Raise if too many of the first frames fail to decode (p-frame/blank check)."""
    frames_checked = min(frames_checked, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    p_allowed = int(frames_checked * p_prop_allowed)

    p_frms = 0
    for _i in range(frames_checked):
        ret, _frame = cap.read()
        p_frms = p_frms + 1 if not ret else p_frms
    if p_frms > p_allowed:
        raise RuntimeError(
            "Video compression method not supported. "
            f"Approximately {(p_frms / frames_checked) * 100}% frames are p frames or blank. "
            "Consider video conversion."
        )
