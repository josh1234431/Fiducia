# Fiducia

**Aerial and satellite photogrammetry.** An independent, open-source photogrammetry
workstation for classical aerial survey.

The full processing chain — interior orientation, ground control, bundle
adjustment, orthorectification, mosaicking, terrain extraction — on one
canvas, running in parallel, saving continuously.

> Fiducia is an independent implementation built from published
> photogrammetric literature. See [NOTICE](NOTICE).

---

## Why it exists

Classical aerial photogrammetry software is expensive, and the incumbent tools
carry decades of accumulated friction: no autosave, absolute paths that break
when a drive remounts, one job spread across a dozen modal dialogs,
single-threaded processing. Fiducia targets each of those directly.

| The usual behaviour | Fiducia |
|---|---|
| No autosave. A crash loses the session. | Every edit is journalled to disk with `fsync` before it is acknowledged, then snapshotted. A hard crash replays the journal on next open. |
| Absolute paths break when a USB drive remounts as `F:` | Paths are stored relative to the project bundle; a broken link is re-resolved by searching remembered roots. |
| A `Processing step` dropdown showing one step at a time | A step rail with a live status light on every step, so the state of the whole block is visible at once. |
| A new viewer window per task | One persistent canvas. The image stays put; the step changes what is overlaid on it. |
| Single-threaded, blocking processing | Tiled work farmed across every core, in a background queue, with progress and cancellation. |
| "Bundle adjustment failed." | A readiness check that names the missing piece before you run, and automatic blunder detection that names the suspect point after. |

## Structure

| Part | Where | What it is |
|---|---|---|
| Processing engine | `engine/` | The Python service that does all the work |
| Job queue | `jobqueue.py` | Background work, progress, cancellation |
| Image viewer | `ImageViewer.jsx` | Tiled pan/zoom canvas |
| Command palette | `CommandPalette.jsx` | <kbd>Ctrl</kbd>+<kbd>K</kbd> to reach anything |
| Fiducial detection | `fiducial_detection.py` | Measure one photo, the rest follow |
| Automatic ground control | `auto_control.py` | Matching against a reference mosaic |
| Certificate reading | `certificate_reader.py` | Camera calibration from the certificate |
| Import and export | `exchange.py` | What crosses the project boundary |

Project bundles are `.fidu` directories.

---

## Running it

The Python environment lives **outside** OneDrive on purpose — a virtualenv
inside a synced folder is unusably slow.

```bash
python -m venv %LOCALAPPDATA%\Fiducia\venv
%LOCALAPPDATA%\Fiducia\venv\Scripts\python.exe -m pip install -r engine/requirements.txt
npm install
```

Desktop app:

```bash
npm run dev
```

Engine on its own, for batch or headless work:

```bash
python engine/server.py --port 8731
```

Every operation the interface performs is a plain HTTP call, so a batch of
blocks can be scripted without opening a window.

---

## Building the installer

```bash
npm run dist
```

This produces `%LOCALAPPDATA%\Fiducia\build\release\Fiducia-Setup-<version>.exe`
elease\Fiducia-Setup-<version>.exe`
(about 185 MB). It builds the interface, freezes the engine with PyInstaller
(`engine/fiducia-engine.spec`, entry point `engine/packaged_main.py`) and packages both
with electron-builder (`electron-builder.config.cjs`). The recipe is in
`packaging/build.ps1`. Build output is kept outside OneDrive.

The installer needs no Python and no administrator rights. It installs per
user, with Start menu and desktop shortcuts and an uninstaller. It is not yet
code-signed, so Windows SmartScreen warns on first run: choose "More info",
then "Run anyway".

To check a packaged engine against the workflow suite:
`FIDUCIA_ENGINE_EXE=<path to fiducia-engine.exe> python scratch/test_api.py`

## Architecture

```
Electron shell  ──spawns──>  Engine (Python/FastAPI)
      │                             │
   React UI  ──HTTP + WebSocket──>  owns ALL project state
                                    │
                              .fidu bundle on disk
                              ├── project.json      canonical state
                              ├── journal/          write-ahead log
                              ├── snapshots/        restore points
                              ├── cache/            overviews
                              └── outputs/          orthos, mosaics, DEMs
```

The interface holds no authoritative state. Every mutation goes to the engine,
is journalled, and the returned state is adopted — so what is on screen is
always what is on disk.

### Engine modules

| Module | Responsibility |
|---|---|
| `collinearity.py` | Collinearity condition, rotation conventions, analytic Jacobians |
| `camera.py` | Interior orientation: fiducial fit, radial distortion, chip geometry |
| `resection.py` | Single-photo exterior orientation |
| `bundle.py` | Sparse block adjustment, robust weighting, blunder detection |
| `geodesy.py` | CRS handling, South African Lo zones |
| `raster.py` | Tiled access, overviews, display rendering |
| `project.py` | Autosave, journalling, snapshots, portable links |
| `jobqueue.py` | Background job queue |
| `ortho.py` | Tiled backward-projection resampling |
| `mosaic.py` | Cutlines, colour balancing, feathered blending |
| `tiepoints.py` | NCC and feature-based matching with RANSAC filtering |
| `stereo_dem.py` | Epipolar resampling and dense stereo matching |
| `lidar.py` | LAS/LAZ filtering and rasterising |
| `satellite_rpc.py` | RPC evaluation, inversion and GCP refinement |
| `fiducial_detection.py` | Automatic fiducial detection |
| `auto_control.py` | Automatic ground control against a reference |
| `certificate_reader.py` | Reading calibration certificates |
| `terrain.py` | DEM merge, repair, hillshade, contours, bare earth, volume |
| `exchange.py` | Survey files, flight logs, camera libraries, GIS layers |
| `reports.py` | Project and residual reports |
| `classic_report.py` | The classic text report layout |
| `plausibility.py` | Is the setup, and the solution, physically possible? |
| `corrections.py` | Atmospheric refraction and earth curvature |
| `memory_budget.py` | How much of the machine a job may use |
| `storage.py` | What happened to the disk, in plain words |

---

## South African coordinate systems

The Lo (Gauss Conform) zones are first-class. Legacy packages ship them still
carrying the **Cape datum**, which has been wrong for every project since the
1999 switch to Hartebeesthoek94, so the earth model has to be corrected by hand
for every project.

Here `ZALO19` **is** Hartebeesthoek94. `ZALO19_WGS84` and `ZALO19_CAPE` are
separate, clearly-labelled entries. All zones from Lo15 to Lo33 are included,
south-oriented (`+axis=wsu`) so coordinates are not mirrored.

---

## Correctness

Fifteen suites. Most build a scene whose answer is known exactly; three check
Fiducia against independent references -- an established commercial
package's solution of a real block, a ray tracer through the standard atmosphere, and a spherical
earth. Parts that need real project data skip themselves unless pointed at it
(`FIDUCIA_CLASSIC_REPORTS`, `FIDUCIA_REAL_PROJECT`).

```bash
python scratch/test_engine.py        # photogrammetric maths
python scratch/test_pipeline.py      # raster pipeline, autosave, recovery
python scratch/test_auto.py          # fiducial detection, GCP matching, certificate reading
python scratch/test_terrain.py       # terrain operations, data exchange, mixed projections
python scratch/test_api.py           # full workflow over HTTP
python scratch/test_storage.py       # losing the project drive mid-session
python scratch/test_classic_report.py     # classic report layout, byte for byte
python scratch/test_plausibility.py  # camera sanity, solution sanity, live residuals
python scratch/test_memory_budget.py      # memory-safe parallel orthorectification
python scratch/test_precision.py     # reported precision against Monte Carlo
python scratch/test_reference_block.py   # a real block against a reference solution
python scratch/test_crs.py           # control, output and DEM in different projections
python scratch/test_corrections.py   # refraction and earth curvature against physics
```

### Against a reference solution, on a real block

Given exactly the inputs of a real film block (calibration, fiducials,
control, tie points, weights), Fiducia reproduces the reference solution
**within the solution's own uncertainty**: every orientation parameter of every
photo within 1.2 standard deviations, every tie point within 3 (median 0.12),
and the control corrections with 0.994 correlation and identical signs. Its
own objective is 5-6% lower at its own solution than at the reference, so the
optimiser reaches the minimum. Of the conventions, the rotation order
Rz(kappa) Ry(phi) Rx(omega), the angle signs and the principal-point sign were
all confirmed from the reference output.

The comparison also showed one photo of that block to be weakly determined --
its corners uncertain by +-12 m -- which is now reported for every solution.

### Precision is reported, and true

Every adjustment reports standard deviations of each photo's position and
angles, of every tie point, and how far each photo's corners could land on
the ground. Over 150 Monte Carlo solves the actual scatter matches the
reported precision to within 0.89-1.14 (sampling error), sigma-0 averages
0.99, and the corner uncertainty matches to within 3%. Ignoring the
position/tilt correlation would have doubled it.

Each test builds a scene whose answer is known exactly and checks the solver
recovers it. Selected results:

- **Analytic Jacobians** match finite differences to 7×10⁻⁸
- **Refraction** constant matches ray tracing through the International
  Standard Atmosphere to 0.02%; corrected, a resection through refracted rays
  recovers the camera exactly (0.25 m out uncorrected)
- **Earth curvature**, on a true sphere, is recovered to 0.4 mm at sea level
  (0.66 m out uncorrected)
- **Control, output and DEM in three different projections** give orthophotos
  correlating 0.9998 with the true ground in all nine combinations
- **Automatic control on hilly ground** lands within 6 mm median of the true
  position, from a model biased by 11.5 m
- **Space resection** recovers a known orientation to 10⁻¹¹ degrees at every
  heading: a sweep of 144 headings and 288 solves is exact to 3×10⁻¹⁰ px. It is
  started from all four quarter-turns and keeps the best physically valid
  answer, because collinearity has false minima with plausible residuals
- **A camera that contradicts its images** (a sensor described landscape for a
  portrait file, chip dimensions typed as offsets) is refused before solving,
  with a one-click fix — residuals alone cannot catch it, since the
  orientation absorbs much of the error
- **Bundle adjustment** on a 3-photo strip returns a 7.99 µm image residual
  against 8 µm of injected measurement noise, with perspective centres within
  2.4 m and tie points triangulating to 0.45 m mean
- **Orthorectification** correlates 0.9946 with the true ground pattern, median
  radiometric error 1 DN
- **Mosaicking** correlates 0.9925 with no hard seam artefacts
- **Crash recovery** replays unsaved edits from the journal
- **Pulling the project drive** mid-session (a real drive letter removed with
  `subst`) refuses the next edit with a message naming the drive, rolls it
  back, and resumes saving by itself when the drive returns
- **Relinking** recovers 3/3 images after they move on disk
- **Volume** against an analytic cone is exact to 0.00%
- **Hillshade** matches the analytic illumination of a known slope to 0.3%
- **Bare-earth filtering** removes a 12 m building and 8 m trees while leaving
  open ground within 0.003 m
- **Contour tracing** of a cone returns circles of the correct radius to 1.5%

### Blunder detection

A control point displaced by 31 m is reported at 29.35 m while good control
stays under 1 m — 61× the spread of the rest, named explicitly:

> Ground residual of 29.35 m is 61.7 times the spread of the other control
> points. Check the surveyed coordinate and the measured position.

This is what robust (soft-L1) weighting buys. Under a plain least-squares loss
the same blunder reads 18 m against 10 m for good points — detectable in
principle, invisible in practice.

---

## Design

**Manrope** carries everything proportional, from a 10px label to the hero. **JetBrains
Mono** takes the data role: rounded terminals, far less rigid, and still
perfectly tabular where columns must align. Tracking is optical — tight at
display sizes, open at small ones.

**Surfaces have depth.** Flat fills separated by hairlines read as a
wireframe. Every surface now carries a faint top-down gradient and a
one-pixel top highlight, which is how light actually falls on a raised panel,
and elevation is carried by shadow rather than by a drawn border. Hairlines
survive only where they encode something — a table row, a column edge.

**The rail has no lines at all.** Steps are soft pills that light up;
separation comes from surface, shadow and space. A progress spine runs down
its edge showing how much of the chain is done.

**Completion is unmistakable.** Four states, each readable without its label:
an empty ring is locked, a dashed teal ring is ready, an amber ring needs
attention, and done is a filled green disc with a checkmark drawn into it.

**Motion answers actions.** Buttons give under the press. Rows lift on
approach. A measured point lands on the photograph with one pulse. A finished
job washes green once and gets out of the way. A number that changed while you
were looking elsewhere flashes to say so. The only unprompted animation is a
step crossing into done — the disc overshoots, the tick draws itself, a halo
expands away — and that one is earned, because it marks real progress through
real work. `prefers-reduced-motion` stops all of it.

The workspace defaults to **dark**: a bright surround measurably distorts how
an operator judges tone in a photograph, which is why every serious
image-analysis tool ships dark. The light theme is complete, not an
afterthought — signal colours are defined per theme, because a hue legible on
a dark panel is not legible on white. Every foreground and background pair, on
every surface, in both themes, clears WCAG AA.

## Licence

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Apache 2.0 rather than MIT for the explicit patent grant and contribution
terms, which is the sensible posture for anything in a field with active
patents.

---

## Status

The core chain is complete and validated: project → camera → images →
fiducials → control → bundle adjustment → orthos → mosaic → reports.

Stereo DEM, LiDAR and satellite RPC are implemented end-to-end with their
interfaces in place, but have had less real-data exercise than the core chain.

## What it does

| Step | What happens |
|---|---|
| Project | Math model, projection, output grid |
| Camera | Interior orientation — read a certificate, or load one from the library |
| Images | Add photographs, measure fiducials (or detect them automatically), clip |
| Control | Import a survey file, measure by hand, or match against a reference |
| Model | Bundle adjustment with robust weighting and blunder detection |
| Ortho | Tiled backward projection across every core |
| Mosaic | Cutlines, colour balance, live preview |
| Stereo DEM | Epipolar resampling and dense matching |
| Terrain | Merge, fill, smooth, hillshade, contour, bare earth, volume, profile |
| LiDAR | LAS/LAZ to terrain models, validated against a reference |
| Reports | Project and residual reports, control and orientation exports |

### Known gaps

- **Fonts are CDN-loaded.** Manrope and Inter come from Google Fonts, with
  system fallbacks. Vendor the woff2 files into `resources/fonts` for a
  guaranteed-offline build.
- **No packaged installer yet.** Runs from source; `electron-builder` is the
  next step.
- **`.pix` output is lightly tested.** GDAL's PCIDSK driver reads `.pix` files,
  so existing projects open directly, and PCIDSK is offered as an export format.
- **Refraction and earth curvature** are applied in the adjustment and the
  orthophoto, but not yet in stereo DEM extraction or automatic control.
- **The height scale factor** of map coordinates (points h above the datum are
  h/R short of their true separation; 24 ppm at 150 m) is not modelled.
- **Tested on synthetic scenes, real film blocks and a real drone flight.**
  More real blocks, especially satellite ones, are the next step.

---

## Author

Created by **Joshua Metcalf** ·
[LinkedIn](https://www.linkedin.com/in/joshua-metcalf-1b2184263)
