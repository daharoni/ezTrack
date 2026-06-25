"""Shared fixtures: a synthetic ground-truth video and the bundled practice clip.

The synthetic video is a white square moving left-to-right on a black background.
Because the true position of the animal is known for every frame, tests can
assert that tracking recovers it -- a real correctness check, not just a smoke
test -- and it runs entirely headlessly in CI without the practice video.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

PRACTICE_VIDEO = (
    Path(__file__).resolve().parent.parent / "PracticeVideos" / "LocationTracking_Clip.mp4"
)

# Synthetic video geometry (kept in one place so tests can assert against it).
SYN_W, SYN_H, SYN_N = 120, 80, 40
SYN_SIDE = 12
SYN_Y = SYN_H // 2
SYN_X0, SYN_X1 = 20, 100  # square center travels from x=20 to x=100


def syn_truth_x() -> np.ndarray:
    """Ground-truth x of the square's center for each of the SYN_N frames."""
    return np.linspace(SYN_X0, SYN_X1, SYN_N)


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the moving-square video to a temp AVI and return its path."""
    path = tmp_path_factory.mktemp("video") / "moving_square.avi"
    writer = cv2.VideoWriter(str(path), 0, 20.0, (SYN_W, SYN_H), isColor=True)
    half = SYN_SIDE // 2
    for cx in syn_truth_x().astype(int):
        frame = np.zeros((SYN_H, SYN_W, 3), dtype=np.uint8)
        frame[SYN_Y - half : SYN_Y + half, cx - half : cx + half] = 255
        writer.write(frame)
    writer.release()
    assert path.exists()
    return path
