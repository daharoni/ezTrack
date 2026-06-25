<p align="center">
  <img width="150" src="./Images/KathleenWang_for_ezTrack.png">
</p>

# Behavior Tracking with ezTrack
ezTrack tracks the location of a single animal across a behavior video on a frame-by-frame
basis (for example, in an open field), quantifies distance travelled and time spent in
user-defined regions of interest, and renders motion traces and occupancy heatmaps. The
workflow is driven from Jupyter notebooks.

This repository has been modernized to install and run as a standard Python package on
current scientific Python (Python ≥ 3.10, NumPy/SciPy/pandas/OpenCV/HoloViews), so it can be
used in a shared virtual environment alongside [minian](https://github.com/miniscope/minian).

> **Scope note:** This package currently ships the **Location Tracking** module. The legacy
> **Freeze Analysis** module remains in `FreezeAnalysis/` as unmodernized source and is not
> part of the installable package.

![Examples](../master/Images/Examples.gif)

# Installation

Use a virtual environment (recommended), then install:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# from a clone of this repo (editable, for teaching/development):
pip install -e .

# or build/install the package:
pip install .
```

OpenCV is installed as `opencv-python-headless` (no GUI windows), matching minian. The
notebook plays tracking video inline via `PlayVideo`; the external-window `PlayVideo_ext`
requires a GUI build of OpenCV and is disabled under the headless install.

# Running the notebooks

The analysis notebooks are bundled inside the package. Copy one into a working directory
with the CLI:

```bash
eztrack notebooks list                 # show available notebooks
eztrack notebooks copy individual      # single-video tracking notebook
eztrack notebooks copy batch           # batch-processing notebook
```

Then launch Jupyter and open the copied notebook:

```bash
jupyter notebook
```

If you cloned the repo, you can also open the notebooks in place under
`eztrack/notebooks/`. The `individual` notebook defaults to the sample clip in
`PracticeVideos/`; for your own data, edit the `video_dict` `dpath`/`file` entries.

# Location Tracking Module
The location tracking module analyzes a single animal's location frame by frame. It lets
you crop the video frame, exclude regions from analysis, specify regions of interest (e.g.
left/right), quantify time spent in each region and distance travelled, and define a
real-world scale.

![schematic_lt](../master/Images/LocationTracking_Schematic.png)

## API at a glance
Configuration uses typed dataclasses (`Session`, `TrackingParams`, `DisplayParams`,
`Scale`, `Stretch`, `Mask`) instead of plain dictionaries. Pipeline functions take a
`Session` and mutate it in place as the analysis progresses:

```python
import eztrack.location as lt

session = lt.Session(dpath="videos", file="clip.mp4", region_names=["left", "right"])
lt.load_and_crop(session, cropmethod="Box")   # sets session.crop
lt.make_reference(session, num_frames=50)      # sets session.reference
lt.roi_plot(session)                           # sets session.roi_stream
location = lt.track_location(session, lt.TrackingParams(method="dark"))
```

Implementation is split into focused modules (`eztrack.io`, `eztrack.tracking`,
`eztrack.roi`, `eztrack.scale`, `eztrack.summary`, `eztrack.viz`, `eztrack.playback`,
`eztrack.batch`); `eztrack.location` re-exports the full public API. Legacy dictionaries
can be migrated with `Session.from_dict(video_dict)`.

# What was modernized
- Added packaging (`pyproject.toml`, PDM backend), a console entry point (`eztrack`), and a
  bundled-notebook CLI — mirroring minian's conventions.
- Fixed deprecated/broken APIs for the modern stack: HoloViews `hv.extension`, SciPy
  `ndimage.center_of_mass`, pandas Copy-on-Write-safe ROI labelling/transition logic,
  OpenCV `int32` polygon points and keyword `interpolation=` on `cv2.resize`.
- Removed the blanket `warnings.filterwarnings("ignore")`.

The package version is a static `1.0.0` for this first PyPI-ready cut; SCM-based versioning
(as minian uses) can be adopted later.

# Citation
Please cite ezTrack if you use it in your research:

> Pennington ZT, Dong Z, Feng Y, Vetere LM, Page-Harley L, Shuman T, Cai DJ (2019). ezTrack:
> An open-source video analysis pipeline for the investigation of animal behavior.
> *Scientific Reports* 9(1): 19979.

For the original instructions and background, see the
[ezTrack wiki](https://github.com/denisecailab/ezTrack/wiki).

# License
This project is licensed under GNU GPLv3.
