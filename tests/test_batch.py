"""End-to-end batch processing over a folder of (synthetic) videos."""

from __future__ import annotations

import shutil

import eztrack as ez


def test_batch_process_writes_outputs(synthetic_video, tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    for name in ("a.avi", "b.avi"):
        shutil.copy(synthetic_video, folder / name)

    session = ez.Session(dpath=str(folder), ftype="avi", region_names=["left"])
    session.selections.rois = ez.ROIs(
        names=["left"], polygons=[[[0, 0], [60, 0], [60, 80], [0, 80]]]
    )
    assert ez.discover_files(session) == ["a.avi", "b.avi"]

    summary, panels = ez.batch_process(
        session,
        ez.TrackParams(threshold_pct=99, window=None),
        num_frames=20,
        check=False,
    )

    assert len(summary) == 2  # one summary row per video
    assert set(summary["file"]) == {"a.avi", "b.avi"}
    assert (folder / "a_tracking.csv").exists()
    assert (folder / "b_tracking.csv").exists()
    assert (folder / "BatchSummary.csv").exists()
    assert type(panels).__name__ == "Layout"  # trace + heatmap per video
