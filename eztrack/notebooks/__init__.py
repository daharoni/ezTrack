"""Jupyter notebooks shipped with ezTrack.

The notebooks live inside the installed package so that ``pip install eztrack``
gives you everything needed to run them offline. Copy one into your working
directory with the ``eztrack notebooks`` CLI::

    eztrack notebooks list
    eztrack notebooks copy individual

Each entry in :data:`NOTEBOOKS` is a single ``.ipynb`` file bundled alongside
this module.
"""

import importlib.resources as ir
import shutil
from pathlib import Path

__all__ = ["NOTEBOOKS", "notebook_root", "copy"]

# Notebook name -> (filename, one-line description).
NOTEBOOKS: dict[str, tuple[str, str]] = {
    "individual": (
        "LocationTracking_Individual.ipynb",
        "Track one animal in a single video: crop, reference, tune, track, summarize.",
    ),
    "batch": (
        "LocationTracking_BatchProcess.ipynb",
        "Apply one set of tracking parameters across a folder of videos.",
    ),
}


def notebook_root() -> Path:
    """Filesystem directory holding the bundled notebooks."""
    return Path(ir.files(__name__))


def copy(name: str, dest: Path) -> list[Path]:
    """Copy the notebook registered under ``name`` into ``dest``.

    Returns the list of paths written (one per notebook).
    """
    if name not in NOTEBOOKS:
        raise KeyError(f"No notebook matching {name!r}. Available: {', '.join(NOTEBOOKS)}")
    filename = NOTEBOOKS[name][0]
    src = notebook_root() / filename
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / filename
    shutil.copy2(src, target)
    return [target]
