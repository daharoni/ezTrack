"""Smoke tests for the interactive widget builders.

The live drawing can't be driven headlessly, but the builders must at least
construct a HoloViews element without error (and the coordinate-conversion logic
they rely on is unit-tested in test_config.py). This guards against the widget
wiring breaking when the core API changes.
"""

from __future__ import annotations

import pytest

import eztrack as ez
from eztrack.config import ROIs


def _is_hv_element(obj) -> bool:
    return hasattr(obj, "opts")  # all HoloViews elements/overlays expose .opts


def _session(synthetic_video, **kw):
    return ez.Session(dpath=str(synthetic_video.parent), file=synthetic_video.name, **kw)


def test_crop_and_mask_tools_build(synthetic_video):
    session = _session(synthetic_video)
    assert _is_hv_element(ez.crop_tool(session))
    assert _is_hv_element(ez.mask_tool(session))


def test_roi_tool_with_and_without_regions(synthetic_video):
    with_regions = _session(synthetic_video, region_names=["left", "right"])
    ez.reference_frame(with_regions, num_frames=20)
    assert _is_hv_element(ez.roi_tool(with_regions))

    no_regions = _session(synthetic_video)
    ez.reference_frame(no_regions, num_frames=20)
    assert _is_hv_element(ez.roi_tool(no_regions))


def _session_with_drawn_rois(synthetic_video, n):
    session = _session(synthetic_video)
    ez.reference_frame(session, num_frames=20)
    data = {"xs": [[0, 10, 10, 0]] * n, "ys": [[0, 0, 10, 10]] * n}
    session.selections.rois = ROIs.from_polydraw(None, data)  # n polygons, auto-named
    return session


def test_name_rois_assigns_and_mirrors_region_names(synthetic_video):
    session = _session_with_drawn_rois(synthetic_video, 2)
    out = ez.name_rois(session, ["left", "right"])
    assert _is_hv_element(out)
    assert session.selections.rois.names == ["left", "right"]
    assert session.region_names == ["left", "right"]  # mirrored for replay/attrs


def test_name_rois_rejects_count_mismatch(synthetic_video):
    session = _session_with_drawn_rois(synthetic_video, 2)
    with pytest.raises(ValueError, match="drew 2 region"):
        ez.name_rois(session, ["only_one"])


def test_name_rois_errors_when_nothing_drawn(synthetic_video):
    session = _session(synthetic_video)
    ez.reference_frame(session, num_frames=20)
    with pytest.raises(ValueError, match="No regions drawn"):
        ez.name_rois(session, ["left"])


def test_distance_tool_and_set_scale(synthetic_video):
    session = _session(synthetic_video)
    ez.reference_frame(session, num_frames=20)
    assert _is_hv_element(ez.distance_tool(session))

    ez.set_scale(session, real_distance=50, unit="cm")
    assert session.selections.scale.real_distance == 50
    assert session.selections.scale.unit == "cm"
