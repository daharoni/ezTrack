<p align="center">
  <img width="150" src="./Images/KathleenWang_for_ezTrack.png">
</p>

# Behavior Tracking with ezTrack

[![pytest](https://github.com/daharoni/ezTrack/actions/workflows/testandcov.yml/badge.svg)](https://github.com/daharoni/ezTrack/actions/workflows/testandcov.yml)
[![lint](https://github.com/daharoni/ezTrack/actions/workflows/lint.yml/badge.svg)](https://github.com/daharoni/ezTrack/actions/workflows/lint.yml)

ezTrack tracks the location of a single animal across a behavior video on a frame-by-frame
basis (for example, in an open field), quantifies distance travelled and time spent in
user-defined regions of interest, and renders motion traces and occupancy heatmaps. The
workflow is driven from Jupyter notebooks.

This is a from-scratch rewrite of the original ezTrack **Location Tracking** module around a
small, pure, fully-tested core, packaged for current scientific Python (Python ≥ 3.10,
NumPy/SciPy/pandas/OpenCV/HoloViews) so it can run in a shared virtual environment alongside
[minian](https://github.com/miniscope/minian).

> **Note on reproducibility:** the tracking algorithm is the same in spirit as the original
> (median background → per-frame difference → threshold → center of mass), but this is a
> clean reimplementation and output is *not* guaranteed bit-identical to pre-2.0 ezTrack or
> the 2019 paper. Pin a version if you need stable numbers across an analysis.

> **Scope:** this package ships the **Location Tracking** module. The legacy **Freeze
> Analysis** module remains in `FreezeAnalysis/` as unmodernized source and is not part of
> the installable package.

![Examples](../master/Images/Examples.gif)

# Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -e .          # editable, for teaching/development
# or: pip install .
```

OpenCV is installed as `opencv-python-headless` (no GUI windows), matching minian. Video
plays back inline in the notebook via `play_inline`; the external-window `play_window`
requires a GUI build of OpenCV and is disabled under the headless install.

# Running the notebooks

The analysis notebooks are bundled inside the package. Copy one into a working directory
with the CLI:

```bash
eztrack notebooks list                 # show available notebooks
eztrack notebooks copy individual      # single-video tracking notebook
eztrack notebooks copy batch           # batch-processing notebook
```

Then launch Jupyter and open the copied notebook (`jupyter notebook`). If you cloned the
repo you can also open them in place under `eztrack/notebooks/`. The file picker opens in
the working directory — browse to your video (or the bundled `PracticeVideos/` clip if you
cloned the repo).

# The pipeline

The analysis is a handful of explicit steps over a `Session`. Selections (crop, exclusion
mask, ROIs, scale) are plain serializable objects: the interactive notebook widgets produce
them, but you can also build them by hand or load them from a JSON file.

```python
import eztrack as ez

session = ez.Session(
    dpath="videos", file="clip.mp4",
    region_names=["left", "right"],
)

# 1. selections — from notebook widgets, or constructed/loaded directly
ez.crop_tool(session)                 # draw a box  -> session.selections.crop
ez.mask_tool(session)                 # exclude regions -> session.selections.mask
ez.reference_frame(session, num_frames=50)
ez.roi_tool(session)                  # draw named regions -> session.selections.rois
ez.distance_tool(session); ez.set_scale(session, 100, "cm")

# replay the exact same selections later / across videos:
session.selections.save("selections.json")
session.selections = ez.Selections.load("selections.json")

# 2. tune + track
params = ez.TrackParams(threshold_pct=99.5, method="abs", window=ez.Window(100, 0.9))
ez.threshold_preview(session, params)              # sanity-check on random frames
location = ez.track(session, params)               # -> per-frame DataFrame

# 3. visualize, save, summarize
ez.trace(session, location); ez.heatmap(session, location)
ez.save_tracking(location, "clip_tracking.csv")   # CSV + a .json sidecar of df.attrs
summary = ez.summarize(location, session, bins=None)
```

Output is a tidy per-frame `DataFrame` (`frame, x, y, detected, distance_px`, a scaled
`distance_<unit>` when a scale is set, plus one boolean column per ROI, an `roi` label and
an `roi_transition` flag). Run parameters and ROI geometry live in `df.attrs`, not repeated
in every row — so save with `ez.save_tracking` (which writes a JSON sidecar of `df.attrs`)
rather than a bare `df.to_csv`, which would drop that provenance.

`distance_px` is the path length counted **only between consecutive detected frames**: a
step into or out of a failed frame (`detected == False`, where the previous position was
carried forward) contributes 0 rather than a spurious freeze-then-jump. Distance is thus an
honest *measured* path length, and a low detection rate under-counts it — `track()` warns
when the animal is detected in under 90% of frames, and `summarize` / `ez.detection_rate`
report `pct_detected`.

**Timing is intentionally left to downstream.** ezTrack never infers time from the video's
frame rate. The `frame` column is the join key: merge the output against your per-frame
timestamp file (e.g. `timestamps.csv` / `timestamps.dat` in the recording directory) to put
the track on a real-time axis and derive speeds.

## Architecture

A HoloViews-free **pure core** that is fully unit-tested headlessly, with thin interactive
and rendering layers on top:

| Module | Responsibility |
| --- | --- |
| `eztrack.config` | plain, serializable selections + validated parameters (no OpenCV/HoloViews) |
| `eztrack.video` | frame reading, preprocessing, reference-frame generation |
| `eztrack.regions` | polygon rasterization, ROI membership, transitions (pure geometry) |
| `eztrack.tracking` | the per-frame `locate` and the full-session `track` |
| `eztrack.analyze` | real-world scaling and binned summaries |
| `eztrack.interactive` | HoloViews widgets that populate `Selections` |
| `eztrack.viz` | trace / heatmap / threshold-preview plots |
| `eztrack.playback`, `eztrack.batch` | inline/window playback and folder batch runs |

# Tests

```bash
pip install -e ".[test]"      # or: pdm install --with test
pdm run test                  # fast suite (bare `pytest` works too)
pdm run test-all              # include the slow full-clip run
pdm run lint                  # ruff check + ruff format --check
```

The suite asserts correctness against a synthetic ground-truth video (a moving square whose
true position is known each frame), unit-tests the geometry and config layers, exercises the
batch and interactive-widget builders, and runs a golden end-to-end pass on the practice clip
with programmatically-set selections.

CI mirrors minian's setup: `testandcov.yml` runs the full suite with coverage across an
OS × Python matrix (Linux 3.10–3.13; macOS/Windows on the endpoints) and uploads to Codecov;
`lint.yml` runs ruff and a `pre-commit` job. Install the local hooks with `pre-commit install`
— `nbstripout` keeps committed notebooks output-free so diffs stay reviewable.

**Interactive widgets** (the HoloViews crop/ROI/mask/distance tools) can't be exercised
headlessly — they only *produce* `Selections`, which the tests cover directly — so run
`LocationTracking_Individual.ipynb` top-to-bottom to confirm the widgets render.

# Citation
Please cite ezTrack if you use it in your research:

> Pennington ZT, Dong Z, Feng Y, Vetere LM, Page-Harley L, Shuman T, Cai DJ (2019). ezTrack:
> An open-source video analysis pipeline for the investigation of animal behavior.
> *Scientific Reports* 9(1): 19979.

For background, see the [ezTrack wiki](https://github.com/denisecailab/ezTrack/wiki).

# License
This project is licensed under GNU GPLv3.
