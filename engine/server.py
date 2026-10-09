"""The Fiducia processing engine.

A local HTTP + WebSocket service that owns all project state and does all the
photogrammetry. The Electron interface is a client; it holds no authoritative
state of its own, which is what makes autosave and crash recovery actually
work -- there is exactly one copy of the truth and it is on disk.

Run standalone for headless/batch work:

    python engine/server.py --port 8731

Everything the interface can do is available here as a plain HTTP call, so the
same operations can be scripted for a batch of blocks without opening a window.
"""

from __future__ import annotations

import os

# Before numpy is imported anywhere: its maths library otherwise sets aside a
# buffer per core in this process (1.3 GB on a 20-core machine). The engine's
# own linear algebra is small, so four threads are plenty. Processes it
# starts get one each (memory_budget.single_threaded_children).
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import argparse
import asyncio
import json
import math
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from fiducia import memory_budget as _memory_budget
# Before any raster is opened: GDAL's default cache is 5% of RAM,
# which the engine does not need and the operator's other work does.
_memory_budget.limit_raster_cache(_memory_budget.ENGINE_CACHE_MB)

from fiducia import camera as camera_mod
from fiducia import (
    auto_control, stereo_dem, exchange, geodesy, mosaic, fiducial_detection, satellite_rpc,
    ortho, certificate_reader, raster, reports, lidar, terrain, classic_report, tiepoints,
    placement,
)
from fiducia import bundle as bundle_module
from fiducia.bundle import BundleInput, Observation, adjust_block, corner_uncertainty
from fiducia.camera import CameraModel, FiducialFit, fit_fiducial_transform, fit_radial_from_table
from fiducia.jobqueue import JobQueue
from fiducia.project import Project, ProjectError, _blank_image
from fiducia.storage import StorageError, explain as explain_storage
from fiducia.resection import resect
from fiducia.collinearity import project_points
from fiducia import plausibility
from fiducia import corrections

VERSION = "0.2.1"

jobs = JobQueue(max_concurrent=3)
_current: Optional[Project] = None
_sockets: set[WebSocket] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None


# -- plumbing --------------------------------------------------------------


def broadcast(payload: dict) -> None:
    """Push an event to every connected client, from any thread."""
    if _loop is None:
        return
    message = json.dumps(payload, default=_json_safe)
    for socket in list(_sockets):
        asyncio.run_coroutine_threadsafe(_send(socket, message), _loop)


async def _send(socket: WebSocket, message: str) -> None:
    try:
        await socket.send_text(message)
    except Exception:
        _sockets.discard(socket)


def _json_safe(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    return str(value)


def clean(value: Any):
    """Recursively replace non-finite floats with None so JSON stays valid."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.floating, np.integer)):
        return clean(value.item())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    return value


def require_project() -> Project:
    if _current is None:
        raise HTTPException(status_code=409, detail="No project is open")
    return _current


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    jobs.subscribe(broadcast)
    yield
    if _current is not None:
        _current.close()
    raster.close_all()


app = FastAPI(title="Fiducia Engine", version=VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- errors ----------------------------------------------------------------
#
# Nothing reaches the interface as a bare "Internal Server Error". A disk
# problem is described in terms of the drive; anything else names the error
# so it can be reported, and the full traceback goes to the engine log.


@app.exception_handler(StorageError)
async def _storage_error(request: Request, exc: StorageError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc), "storage": exc.to_dict()})


@app.exception_handler(OSError)
async def _os_error(request: Request, exc: OSError) -> JSONResponse:
    problem = explain_storage(exc, action="read or write")
    if _current is not None:
        # The file may be anywhere; the project folder is what matters most.
        _current.check_storage(force=True)
    return JSONResponse(status_code=503, content={"detail": str(problem),
                                                  "storage": problem.to_dict()})


@app.exception_handler(ProjectError)
async def _project_error(request: Request, exc: ProjectError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    import traceback
    traceback.print_exception(exc)
    # This handler runs outside the CORS middleware, so the header is added
    # here, or the browser hides the message behind a network error.
    return JSONResponse(
        status_code=500,
        content={"detail": f"Something went wrong in the engine ({type(exc).__name__}: {exc}). "
                           "Your project is safe; the engine log has the details."},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "version": VERSION,
        "project": _current.summary() if _current else None,
        "activeJobs": len(jobs.list(active_only=True)),
    }


@app.websocket("/live")
async def live(socket: WebSocket) -> None:
    """Progress, autosave confirmations and project changes, pushed."""
    await socket.accept()
    _sockets.add(socket)
    try:
        await socket.send_text(json.dumps({"type": "hello", "version": VERSION}))
        while True:
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _sockets.discard(socket)


# -- project ---------------------------------------------------------------


@app.post("/project/new")
def project_new(payload: dict = Body(...)) -> dict:
    global _current
    directory = payload.get("directory")
    name = payload.get("name") or "Untitled project"
    if not directory:
        raise HTTPException(400, "A project directory is required")

    if _current is not None:
        _current.close()

    try:
        _current = Project.create(directory, name, payload.get("description", ""))
    except ProjectError as exc:
        raise HTTPException(400, str(exc)) from exc

    _record_author(_current, payload.get("author"))
    _current.subscribe(broadcast)
    broadcast({"type": "project.opened", "summary": _current.summary()})
    return clean({"state": _current.state, "summary": _current.summary()})


def _record_author(project, author) -> None:
    """Record who the project belongs to, if it does not say already.

    A project keeps its original author; opening someone else's bundle does
    not reassign it. Bundles made before authors were recorded take the name
    of whoever opens them first.
    """
    author = (author or "").strip()
    if author and not (project.state.get("author") or "").strip():
        project.mutate("set_author", lambda state: state.__setitem__("author", author))


@app.post("/project/open")
def project_open(payload: dict = Body(...)) -> dict:
    global _current
    directory = payload.get("directory")
    if not directory:
        raise HTTPException(400, "A project directory is required")

    if _current is not None:
        _current.close()

    try:
        _current = Project.open(directory)
    except ProjectError as exc:
        raise HTTPException(400, str(exc)) from exc

    _record_author(_current, payload.get("author"))
    _current.subscribe(broadcast)
    links = [vars(s) for s in _current.relink_all()]
    broadcast({"type": "project.opened", "summary": _current.summary()})
    return clean({"state": _current.state, "summary": _current.summary(), "links": links})


@app.get("/project")
def project_get() -> dict:
    project = require_project()
    return clean({"state": project.state, "summary": project.summary()})


@app.get("/project/storage")
def project_storage(force: bool = False) -> dict:
    """Whether the project folder can still be written, and if not, why."""
    return require_project().check_storage(force=force)


@app.post("/project/save-as")
def project_save_as(payload: dict = Body(...)) -> dict:
    """Continue the open project in a new folder -- the way off a lost drive."""
    global _current
    project = require_project()
    directory = payload.get("directory")
    if not directory:
        raise HTTPException(400, "A destination folder is required")
    moved = project.save_as(directory)
    _current = moved
    moved.subscribe(broadcast)
    broadcast({"type": "project.opened", "summary": moved.summary()})
    return clean({"state": moved.state, "summary": moved.summary()})


@app.post("/project/close")
def project_close() -> dict:
    global _current
    if _current is not None:
        _current.close()
        _current = None
    raster.close_all()
    return {"ok": True}


@app.patch("/project")
def project_patch(payload: dict = Body(...)) -> dict:
    """Shallow-merge top-level keys. Journalled and autosaved like any edit."""
    project = require_project()

    def apply(state: dict) -> None:
        for key, value in payload.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and isinstance(state.get(key), dict):
                state[key].update(value)
            else:
                state[key] = value

    project.mutate("patch", apply)
    return clean({"state": project.state, "summary": project.summary()})


@app.post("/project/snapshot")
def project_snapshot(payload: dict = Body(default={})) -> dict:
    project = require_project()
    return project.snapshot(payload.get("reason", "manual"))


@app.get("/project/snapshots")
def project_snapshots() -> dict:
    return {"snapshots": require_project().list_snapshots()}


@app.post("/project/restore")
def project_restore(payload: dict = Body(...)) -> dict:
    project = require_project()
    state = project.restore_snapshot(payload["path"])
    return clean({"state": state, "summary": project.summary()})


@app.post("/project/archive")
def project_archive(payload: dict = Body(...)) -> dict:
    project = require_project()
    path = project.archive(payload["destination"], payload.get("includeOutputs", False))
    return {"path": path}


@app.post("/project/relink")
def project_relink(payload: dict = Body(default={})) -> dict:
    project = require_project()
    if payload.get("imageId") and payload.get("path"):
        image = project.relink_image(payload["imageId"], payload["path"])
        _refresh_image_metadata(project, image)
        return clean({"image": image, "state": project.state})
    links = [vars(s) for s in project.relink_all()]
    return clean({"links": links, "state": project.state})


# -- reference data --------------------------------------------------------


@app.get("/projections")
def projections() -> dict:
    return {"presets": geodesy.list_presets()}


@app.get("/projections/describe")
def projection_describe(identifier: str = Query(...)) -> dict:
    try:
        return geodesy.describe_crs(identifier)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/reference")
def reference() -> dict:
    """Static vocabularies, so the interface never hardcodes engine options."""
    return {
        "fiducialSlots": list(camera_mod.FIDUCIAL_SLOTS),
        "enhancements": list(raster.ENHANCEMENTS),
        "resampling": list(ortho.RESAMPLING_ORDER.keys()),
        "colorBalance": list(mosaic.COLOR_BALANCE_METHODS),
        "cutlines": list(mosaic.CUTLINE_METHODS),
        "normalization": list(mosaic.NORMALIZATION_METHODS),
        "matchMethods": list(tiepoints.MATCH_METHODS),
        "demDetail": list(stereo_dem.DETAIL_LEVELS),
        "demMethods": list(stereo_dem.EXTRACTION_METHODS),
        "terrainTypes": list(stereo_dem.TERRAIN_TYPES),
        "lidarClasses": lidar.ASPRS_CLASSES,
        "cellAssignment": list(lidar.CELL_ASSIGNMENT),
        "voidFill": list(lidar.VOID_FILL),
        "heightMethods": list(lidar.HEIGHT_METHODS),
    }


# -- camera ----------------------------------------------------------------


@app.post("/camera/distortion-table")
def camera_distortion_table(payload: dict = Body(...)) -> dict:
    """Fit K0..K3 from a calibration certificate's distortion table."""
    try:
        return clean(
            fit_radial_from_table(
                payload.get("radii", []),
                payload.get("distortions", []),
                payload.get("distanceUnits", "mm"),
                payload.get("distortionUnits", "um"),
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/camera")
def camera_set(payload: dict = Body(...)) -> dict:
    project = require_project()
    model = CameraModel.from_dict(payload)
    # A self-calibration is a correction to one particular camera
    # description; once the description changes it no longer applies.
    current = CameraModel.from_dict(project.state.get("camera") or {})
    if model.without_adjustment().to_dict() != current.without_adjustment().to_dict():
        model = model.without_adjustment()
    refit = (model.kind == "film"
             and (model.fiducial_transform != current.fiducial_transform
                  or model.fiducials_mm != current.fiducials_mm))

    def apply(state: dict) -> None:
        state["camera"] = model.to_dict()
        if not refit:
            return
        # Every photo's interior orientation follows from these, so each is
        # fitted again rather than left describing the old camera.
        for image in state.get("images", []):
            measured = image.get("fiducials") or {}
            common = {k: v for k, v in measured.items() if k in model.fiducials_mm}
            if len(common) < 3:
                continue
            try:
                image["fiducialFit"] = fit_fiducial_transform(
                    {k: tuple(v) for k, v in common.items()},
                    {k: tuple(model.fiducials_mm[k]) for k in common},
                    model.fiducial_transform,
                ).to_dict()
                image["fiducialError"] = None
            except ValueError as exc:
                image["fiducialFit"] = None
                image["fiducialError"] = str(exc)

    project.mutate("camera", apply)
    return clean({"camera": model.to_dict()})


# -- images ----------------------------------------------------------------


def _refresh_image_metadata(project: Project, image: dict) -> None:
    try:
        info = raster.raster_info(image["path"])
        image["width"] = info.width
        image["height"] = info.height
        image["bandCount"] = info.band_count
        image["driver"] = info.driver
        image["dtype"] = info.dtype
        image["hasGeoreferencing"] = info.has_georeferencing
        image["maxZoom"] = info.to_dict()["maxZoom"]
        image["online"] = True
    except Exception as exc:
        image["online"] = False
        image["error"] = str(exc)


@app.post("/images/add")
def images_add(payload: dict = Body(...)) -> dict:
    project = require_project()
    paths = payload.get("paths") or []
    if not paths:
        raise HTTPException(400, "No image paths supplied")

    added: list[dict] = []

    def apply(state: dict) -> None:
        existing = {Path(i["path"]).resolve() for i in state["images"] if i.get("path")}
        for raw in paths:
            resolved = Path(raw).resolve()
            if resolved in existing:
                continue
            image = _blank_image(str(resolved))
            image["storedPath"] = project.store_path(resolved)
            _refresh_image_metadata(project, image)
            state["images"].append(image)
            added.append(image)
            project.remember_root(resolved)

    project.mutate("images_add", apply)

    # Overviews make the viewer instant; build them in the background so the
    # operator can start work on the first image immediately.
    for image in added:
        jobs.submit(
            "overviews",
            f"Building overviews — {image['name']}",
            lambda progress, cancel, path=image["path"]: raster.build_overviews(path),
            meta={"imageId": image["id"]},
        )

    return clean({"added": added, "state": project.state})


@app.delete("/images/{image_id}")
def images_remove(image_id: str) -> dict:
    project = require_project()

    def apply(state: dict) -> None:
        state["images"] = [i for i in state["images"] if i["id"] != image_id]
        state["observations"] = [o for o in state["observations"] if o.get("imageId") != image_id]

    project.mutate("images_remove", apply)
    return clean({"state": project.state})


@app.patch("/images/{image_id}")
def images_patch(image_id: str, payload: dict = Body(...)) -> dict:
    project = require_project()

    def apply(state: dict) -> dict:
        for image in state["images"]:
            if image["id"] == image_id:
                image.update({k: v for k, v in payload.items() if k != "id"})
                return image
        raise HTTPException(404, f"No image {image_id}")

    image = project.mutate("images_patch", apply)
    return clean({"image": image, "state": project.state})


@app.get("/images/{image_id}/info")
def images_info(image_id: str) -> dict:
    project = require_project()
    image = project.image(image_id)
    if not image.get("online"):
        raise HTTPException(409, f"{image['name']} is offline")
    return clean(raster.raster_info(image["path"]).to_dict())


@app.get("/images/{image_id}/tile/{z}/{x}/{y}.png")
def images_tile(
    image_id: str,
    z: int,
    x: int,
    y: int,
    enhancement: str = Query("linear2pct"),
    bands: str = Query(""),
    gamma: float = Query(1.0),
    colormap: str = Query(""),
) -> Response:
    """One display tile. This endpoint is hit constantly -- keep it lean."""
    project = require_project()
    image = project.image(image_id)
    if not image.get("online"):
        raise HTTPException(409, "Image is offline")

    band_list = [int(b) for b in bands.split(",") if b.strip().isdigit()] or None
    try:
        data = raster.render_tile(
            image["path"], z, x, y, band_list, enhancement, gamma, colormap or None
        )
    except Exception as exc:
        raise HTTPException(500, f"Tile render failed: {exc}") from exc

    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/raster/tile/{z}/{x}/{y}.png")
def raster_tile(
    z: int, x: int, y: int,
    path: str = Query(...),
    enhancement: str = Query("linear2pct"),
    bands: str = Query(""),
    gamma: float = Query(1.0),
    colormap: str = Query(""),
) -> Response:
    """Tiles for any raster on disk -- reference mosaics, DEMs, outputs."""
    band_list = [int(b) for b in bands.split(",") if b.strip().isdigit()] or None
    try:
        data = raster.render_tile(path, z, x, y, band_list, enhancement, gamma, colormap or None)
    except Exception as exc:
        raise HTTPException(500, f"Tile render failed: {exc}") from exc
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/raster/info")
def raster_info_endpoint(path: str = Query(...)) -> dict:
    try:
        return clean(raster.raster_info(path).to_dict())
    except Exception as exc:
        raise HTTPException(400, f"Could not open {path}: {exc}") from exc


@app.get("/raster/sample")
def raster_sample(
    path: str = Query(...), x: float = Query(...), y: float = Query(...),
    crs: Optional[str] = Query(None),
) -> dict:
    """Elevation (or pixel value) at a map coordinate — used by Extract Elevation.

    When the point misses the DEM, the swapped pair is tried too: control
    typed the wrong way round is the commonest reason a coordinate falls
    outside, and saying so is more use than saying only that it missed.
    """
    if not (math.isfinite(x) and math.isfinite(y)):
        raise HTTPException(400, "The coordinates must be two plain numbers. Check "
                                 "neither box holds text or both values at once.")
    # Coordinates are in the control's system (the open project's, unless
    # given); the DEM may be in another. Read it where the point really is.
    if crs is None and _current is not None:
        crs = (_current.state.get("projection") or {}).get("gcpSource") \
            or (_current.state.get("projection") or {}).get("output")
    from fiducia.ortho import _converter, _dem_crs
    to_dem = _converter(crs, _dem_crs(path))

    def sample(px: float, py: float) -> float:
        if to_dem:
            (px,), (py,) = to_dem([px], [py])
        return raster.sample_elevation(path, [px], [py])[0]

    try:
        value = sample(x, y)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc

    if math.isfinite(value):
        return {"value": float(value)}

    swapped = None
    try:
        other = sample(y, x)
        if math.isfinite(other):
            swapped = float(other)
    except Exception:  # noqa: BLE001
        pass
    return {"value": None, "outside": True, "swappedValue": swapped}


# -- fiducials / interior orientation --------------------------------------


@app.post("/images/{image_id}/fiducials")
def fiducials_set(image_id: str, payload: dict = Body(...)) -> dict:
    """Store measured fiducials and refit interior orientation immediately.

    Refitting on every mark is what lets the operator see the residual fall as
    they work, instead of discovering after all eight that one is wrong.
    """
    project = require_project()
    camera = CameraModel.from_dict(project.state.get("camera") or {})

    def apply(state: dict) -> dict:
        for image in state["images"]:
            if image["id"] != image_id:
                continue
            measured = dict(image.get("fiducials") or {})
            if "fiducials" in payload:
                measured = {k: list(v) for k, v in payload["fiducials"].items()}
            if payload.get("slot"):
                if payload.get("clear"):
                    measured.pop(payload["slot"], None)
                else:
                    measured[payload["slot"]] = [payload["col"], payload["row"]]
            image["fiducials"] = measured

            calibrated = camera.fiducials_mm
            common = {k: v for k, v in measured.items() if k in calibrated}
            if len(common) >= 3:
                try:
                    fit = fit_fiducial_transform(
                        {k: tuple(v) for k, v in common.items()},
                        {k: tuple(calibrated[k]) for k in common},
                        camera.fiducial_transform,
                    )
                    image["fiducialFit"] = fit.to_dict()
                    image["fiducialError"] = None
                except ValueError as exc:
                    image["fiducialFit"] = None
                    image["fiducialError"] = str(exc)
            else:
                image["fiducialFit"] = None
                image["fiducialError"] = (
                    f"{len(common)} of at least 3 fiducials measured"
                )
            return image
        raise HTTPException(404, f"No image {image_id}")

    image = project.mutate("fiducials", apply)
    return clean({"image": image})



# -- data exchange ---------------------------------------------------------


@app.post("/exchange/sniff")
def exchange_sniff(payload: dict = Body(...)) -> dict:
    """Inspect a text table so the operator can confirm the column mapping."""
    try:
        return clean(exchange.sniff_table(payload["path"]).to_dict())
    except exchange.ExchangeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/exchange/control/import")
def exchange_control_import(payload: dict = Body(...)) -> dict:
    """Read control points from a survey file, optionally applying them."""
    project = require_project()
    try:
        result = exchange.read_control_points(payload["path"], payload.get("mapping"))
    except exchange.ExchangeError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not payload.get("apply"):
        return clean(result)

    images = {i["name"]: i["id"] for i in project.state.get("images", [])}
    replace = bool(payload.get("replace"))

    def apply(state: dict) -> dict:
        if replace:
            keep = {g["id"] for g in state["gcps"] if g.get("source") != "import"}
            state["gcps"] = [g for g in state["gcps"] if g["id"] in keep]
            state["observations"] = [
                o for o in state["observations"]
                if o["pointId"] in keep or o.get("kind") == "tie"
            ]

        added, measured, unmatched = 0, 0, []
        for entry in result["points"]:
            existing = next((g for g in state["gcps"] if g["id"] == entry["id"]), None)
            record = existing or {"id": entry["id"], "note": ""}
            record.update({
                "x": entry["x"], "y": entry["y"], "z": entry["z"],
                "isCheckPoint": entry["isCheckPoint"],
                "source": "import",
            })
            if entry.get("sigma") is not None:
                record["sigma"] = entry["sigma"]
            if existing is None:
                state["gcps"].append(record)
                added += 1

            # Some survey files also carry the image measurement.
            if entry.get("image") and entry.get("col") is not None:
                image_id = images.get(entry["image"])
                if image_id is None:
                    stem = str(entry["image"]).rsplit(".", 1)[0]
                    image_id = images.get(stem)
                if image_id:
                    state["observations"] = [
                        o for o in state["observations"]
                        if not (o["pointId"] == entry["id"] and o["imageId"] == image_id)
                    ]
                    state["observations"].append({
                        "pointId": entry["id"], "imageId": image_id,
                        "col": entry["col"], "row": entry["row"],
                        "kind": "gcp", "weight": 1.0,
                    })
                    measured += 1
                else:
                    unmatched.append(entry["image"])

        return {"added": added, "measured": measured,
                "unmatchedImages": sorted(set(unmatched))[:10]}

    outcome = project.mutate("control_import", apply)
    return clean({**result, **outcome, "state": project.state})


@app.post("/exchange/control/export")
def exchange_control_export(payload: dict = Body(...)) -> dict:
    project = require_project()
    model = project.state.get("model") or {}
    residuals = model.get("controlResidualsM") if payload.get("withResiduals", True) else None

    if payload.get("format") == "geojson":
        return clean(exchange.write_points_geojson(
            payload["path"], project.state.get("gcps", []), residuals,
            (project.state.get("projection") or {}).get("output"),
        ))
    return clean(exchange.write_control_points(
        payload["path"], project.state.get("gcps", []), residuals,
    ))


@app.post("/exchange/exterior/import")
def exchange_exterior_import(payload: dict = Body(...)) -> dict:
    """Import a flight log, so a block can be rectified without ground control."""
    project = require_project()
    try:
        result = exchange.read_exterior_orientation(
            payload["path"], payload.get("mapping"), payload.get("angleUnits", "auto"),
        )
    except exchange.ExchangeError as exc:
        raise HTTPException(400, str(exc)) from exc

    if not payload.get("apply"):
        return clean(result)

    def apply(state: dict) -> dict:
        by_name = {}
        for image in state["images"]:
            by_name[image.get("name", "")] = image
            by_name[Path(image.get("path", "")).name] = image

        applied, unmatched = 0, []
        for record in result["records"]:
            image = by_name.get(record["image"]) or by_name.get(
                str(record["image"]).rsplit(".", 1)[0]
            )
            if image is None:
                unmatched.append(record["image"])
                continue
            image["exterior"] = record["exterior"]
            image["exteriorSource"] = "imported"
            # The measurement itself, kept apart from the working orientation
            # a solve replaces, so the adjustment can weigh it as an
            # observation (GNSS/IMU-assisted triangulation).
            image["exteriorObserved"] = list(record["exterior"])
            applied += 1
        return {"applied": applied, "unmatched": unmatched[:12]}

    outcome = project.mutate("exterior_import", apply)
    return clean({**result, **outcome, "state": project.state})


@app.post("/exchange/exterior/export")
def exchange_exterior_export(payload: dict = Body(...)) -> dict:
    project = require_project()
    return clean(exchange.write_exterior_orientation(
        payload["path"], project.state.get("images", []),
        payload.get("angleUnits", "degrees"),
    ))


@app.post("/exchange/footprints")
def exchange_footprints(payload: dict = Body(...)) -> dict:
    """Write the flight index: every photograph's ground footprint."""
    project = require_project()
    state = project.state
    return clean(exchange.write_footprints_geojson(
        payload["path"], state.get("images", []), state.get("camera") or {},
        (state.get("dem") or {}).get("referencePath"),
        (state.get("projection") or {}).get("output"),
    ))


@app.post("/exchange/camera/save")
def exchange_camera_save(payload: dict = Body(...)) -> dict:
    project = require_project()
    try:
        return clean(exchange.save_camera_library(
            payload["path"], project.state.get("camera") or {}, payload.get("label"),
        ))
    except exchange.ExchangeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/exchange/camera/load")
def exchange_camera_load(payload: dict = Body(...)) -> dict:
    try:
        library = exchange.load_camera_library(payload["path"])
    except exchange.ExchangeError as exc:
        raise HTTPException(400, str(exc)) from exc

    label = payload.get("label")
    if label:
        entry = next((c for c in library["cameras"] if c["label"] == label), None)
        if entry is None:
            raise HTTPException(404, "No camera by that name in the library")
        project = require_project()
        project.mutate("camera_load", lambda s: s.update({"camera": entry["camera"]}))
        return clean({"camera": entry["camera"], "state": project.state})

    return clean(library)


# -- terrain ---------------------------------------------------------------


@app.post("/terrain/statistics")
def terrain_statistics(payload: dict = Body(...)) -> dict:
    try:
        return clean(terrain.statistics(payload["path"]))
    except terrain.TerrainError as exc:
        raise HTTPException(400, str(exc)) from exc


def _terrain_job(kind: str, label: str, work, meta=None) -> dict:
    job = jobs.submit(kind, label, work, meta=meta or {})
    return {"job": job.to_dict()}


def _record_surface(project, entry: dict, role: str) -> None:
    """Keep every derived surface in the project, so it can be used again."""
    path = entry.get("outputPath", entry.get("path"))

    def apply(state: dict) -> None:
        state.setdefault("surfaces", [])
        state["surfaces"] = [s for s in state["surfaces"] if s.get("path") != path]
        state["surfaces"].append({
            "path": path,
            "role": role,
            "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {k: entry.get(k) for k in
                        ("cellSize", "coverage", "min", "max", "mean") if k in entry},
        })

    project.mutate("surface_recorded", apply)
    broadcast({"type": "project.changed", "operation": "surface_recorded"})


def _record_product(project, entry: dict, kind: str, source: str) -> None:
    """Keep hillshades and contours in the project, so they can be found again.

    Kept apart from surfaces: they are made from an elevation model but are not
    one, so they must not be offered for merging or volumes.
    """
    path = entry.get("outputPath", entry.get("path"))

    def apply(state: dict) -> None:
        products = [p for p in state.get("terrainProducts", []) if p.get("path") != path]
        products.append({
            "path": path,
            "kind": kind,
            "source": source,
            "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["terrainProducts"] = products

    project.mutate("terrain_product_recorded", apply)
    broadcast({"type": "project.changed", "operation": "terrain_product_recorded"})


@app.post("/terrain/merge")
def terrain_merge(payload: dict = Body(...)) -> dict:
    project = require_project()
    inputs = payload.get("inputs") or []
    if len(inputs) < 2:
        raise HTTPException(400, "Merging needs at least two elevation models")

    output = payload.get("outputPath") or str(project.outputs_dir / "merged_dem.tif")

    def work(progress, should_cancel):
        result = terrain.merge(inputs, output, payload.get("method", "feather"),
                               payload.get("cellSize"), progress)
        _record_surface(project, result, "merged")
        return result

    return _terrain_job("terrain", "Merging " + str(len(inputs)) + " elevation models", work)


@app.post("/terrain/fill")
def terrain_fill(payload: dict = Body(...)) -> dict:
    project = require_project()
    source = payload["path"]
    output = payload.get("outputPath") or str(
        project.outputs_dir / (Path(source).stem + "_filled.tif"))

    def work(progress, should_cancel):
        result = terrain.fill_voids(source, output,
                                    int(payload.get("maxRadiusCells", 40)),
                                    payload.get("method", "linear"), progress)
        _record_surface(project, result, "filled")
        return result

    return _terrain_job("terrain", "Filling voids in " + Path(source).name, work)


@app.post("/terrain/smooth")
def terrain_smooth(payload: dict = Body(...)) -> dict:
    project = require_project()
    source = payload["path"]
    output = payload.get("outputPath") or str(
        project.outputs_dir / (Path(source).stem + "_smoothed.tif"))

    def work(progress, should_cancel):
        result = terrain.smooth(source, output, payload.get("method", "median"),
                                int(payload.get("size", 3)), progress)
        _record_surface(project, result, "smoothed")
        return result

    return _terrain_job("terrain", "Smoothing " + Path(source).name, work)


@app.post("/terrain/hillshade")
def terrain_hillshade(payload: dict = Body(...)) -> dict:
    project = require_project()
    source = payload["path"]
    output = payload.get("outputPath") or str(
        project.outputs_dir / (Path(source).stem + "_hillshade.tif"))

    def work(progress, should_cancel):
        raster.close_raster(output)   # may be open in the viewer
        result = terrain.hillshade(
            source, output,
            float(payload.get("azimuth", 315.0)),
            float(payload.get("altitude", 45.0)),
            float(payload.get("zFactor", 1.0)),
            progress,
        )
        _record_product(project, {**result, "outputPath": result.get("outputPath", output)},
                        "hillshade", source)
        return result

    return _terrain_job("terrain", "Shading " + Path(source).name, work)


@app.post("/terrain/contours")
def terrain_contours(payload: dict = Body(...)) -> dict:
    project = require_project()
    source = payload["path"]
    output = payload.get("outputPath") or str(
        project.outputs_dir / (Path(source).stem + "_contours.geojson"))

    def work(progress, should_cancel):
        result = terrain.contours(
            source, output,
            float(payload.get("interval", 5.0)),
            float(payload.get("base", 0.0)),
            int(payload.get("indexEvery", 5)),
            float(payload.get("simplifyPx", 0.7)),
            int(payload.get("smoothCells", 2)),
            progress,
        )
        _record_product(project, {**result, "outputPath": result.get("outputPath", output)},
                        "contours", source)
        return result

    return _terrain_job("terrain", "Contouring " + Path(source).name, work)


@app.post("/terrain/bare-earth")
def terrain_bare_earth(payload: dict = Body(...)) -> dict:
    project = require_project()
    source = payload["path"]
    output = payload.get("outputPath") or str(
        project.outputs_dir / (Path(source).stem + "_dtm.tif"))

    def work(progress, should_cancel):
        result = terrain.surface_to_terrain(
            source, output,
            int(payload.get("maxWindowCells", 33)),
            float(payload.get("slopeTolerance", 0.4)),
            float(payload.get("initialTolerance", 0.5)),
            float(payload.get("maxTolerance", 8.0)),
            progress,
        )
        _record_surface(project, result, "terrain")
        return result

    return _terrain_job("terrain", "Filtering " + Path(source).name + " to bare earth", work)


@app.post("/terrain/volume")
def terrain_volume(payload: dict = Body(...)) -> dict:
    try:
        return clean(terrain.volume_between(
            payload["surface"], payload["base"], payload.get("bounds")))
    except terrain.TerrainError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/terrain/profile")
def terrain_profile(payload: dict = Body(...)) -> dict:
    try:
        return clean(terrain.profile(payload["path"], payload["line"],
                                     int(payload.get("samples", 400))))
    except terrain.TerrainError as exc:
        raise HTTPException(400, str(exc)) from exc


# -- automatic measurement -------------------------------------------------


@app.post("/auto/fiducials")
def auto_fiducials(payload: dict = Body(...)) -> dict:
    """Find fiducial marks on one or more photographs.

    With ``templateImageId`` set, chips are cut from a photo the operator has
    already measured -- the accurate path. Without it, a generated cross is
    used, which is best-effort for the first photo of a block.
    """
    project = require_project()
    camera = CameraModel.from_dict(project.state.get("camera") or {})

    target_ids = payload.get("imageIds") or []
    if not target_ids:
        raise HTTPException(400, "No images selected")

    templates = None
    template_id = payload.get("templateImageId")
    if template_id:
        source = project.image(template_id)
        measured = source.get("fiducials") or {}
        if len(measured) < 3:
            raise HTTPException(
                400,
                f"{source.get('name')} has only {len(measured)} measured marks. "
                "Measure at least three on it first, or run without a template.",
            )
        templates = fiducial_detection.extract_templates(
            source["path"], measured, int(payload.get("chipPx", 96))
        )

    options = fiducial_detection.DetectionOptions(
        chip_px=int(payload.get("chipPx", 96)),
        min_score=float(payload.get("minScore", 0.45)),
        max_residual_px=float(payload.get("maxResidualPx", 4.0)),
    )

    def work(progress, should_cancel):
        outcomes = []
        for index, image_id in enumerate(target_ids):
            if should_cancel():
                raise InterruptedError("Cancelled")
            image = project.image(image_id)
            if not image.get("online"):
                outcomes.append({"imageId": image_id, "name": image.get("name"),
                                 "error": "Image is offline"})
                continue

            def sub(fraction, message, index=index, name=image.get("name")):
                progress((index + fraction) / len(target_ids), f"{name} — {message}")

            try:
                result = fiducial_detection.detect_fiducials(
                    image["path"], camera, templates, options, sub
                )
            except Exception as exc:
                outcomes.append({"imageId": image_id, "name": image.get("name"),
                                 "error": str(exc)})
                continue

            entry = result.to_dict()
            entry["imageId"] = image_id
            entry["name"] = image.get("name")
            outcomes.append(entry)

        return {"results": outcomes}

    label = f"Detecting fiducials on {len(target_ids)} photo" + ("s" if len(target_ids) != 1 else "")
    job = jobs.submit("fiducials", label, work, meta={"imageIds": target_ids})
    return {"job": job.to_dict()}


@app.post("/auto/fiducials/apply")
def auto_fiducials_apply(payload: dict = Body(...)) -> dict:
    """Write accepted fiducial proposals onto their images."""
    project = require_project()
    camera = CameraModel.from_dict(project.state.get("camera") or {})

    def apply(state: dict) -> list:
        applied = []
        for entry in payload.get("results", []):
            image = next(
                (i for i in state["images"] if i["id"] == entry.get("imageId")), None
            )
            if image is None:
                continue
            measured = dict(image.get("fiducials") or {})
            for candidate in entry.get("candidates", []):
                measured[candidate["slot"]] = [candidate["col"], candidate["row"]]
            image["fiducials"] = measured

            common = {k: v for k, v in measured.items() if k in camera.fiducials_mm}
            if len(common) >= 3:
                try:
                    fit = fit_fiducial_transform(
                        {k: tuple(v) for k, v in common.items()},
                        {k: tuple(camera.fiducials_mm[k]) for k in common},
                        camera.fiducial_transform,
                    )
                    image["fiducialFit"] = fit.to_dict()
                    image["fiducialError"] = None
                except ValueError as exc:
                    image["fiducialError"] = str(exc)
            applied.append(image["id"])
        return applied

    applied = project.mutate("fiducials_apply", apply)
    return clean({"applied": applied, "state": project.state})


@app.post("/auto/gcps")
def auto_gcps(payload: dict = Body(...)) -> dict:
    """Propose ground control by correlating the photo against a reference."""
    project = require_project()
    projection = project.state.get("projection") or {}
    control_crs = projection.get("gcpSource") or projection.get("output")
    state = project.state

    reference = payload.get("referencePath") or (state.get("dem") or {}).get("referenceImage")
    if not reference:
        raise HTTPException(
            400,
            "No reference image set. Choose a geocoded orthomosaic to match against.",
        )

    dem_path = payload.get("demPath") or (state.get("dem") or {}).get("referencePath")
    image_ids = payload.get("imageIds") or [i["id"] for i in project.images_online()]

    options = auto_control.AutoControlOptions(
        target_count=int(payload.get("targetCount", 20)),
        search_m=float(payload.get("searchM", 90.0)),
        min_score=float(payload.get("minScore", 0.55)),
        min_separation_m=float(payload.get("minSeparationM", 120.0)),
        # Where no DEM covers a point: the height of the control already
        # measured, which is far closer to the ground than sea level.
        background_elevation=float(payload["backgroundElevation"])
        if payload.get("backgroundElevation") is not None
        else float(np.mean([g["z"] for g in state.get("gcps", []) if g.get("z") is not None]))
        if any(g.get("z") is not None for g in state.get("gcps", [])) else 0.0,
    )

    camera_dict = state.get("camera") or {}

    def seeds_for(image_id: str) -> list:
        seeds = []
        for record in state.get("observations", []):
            if record["imageId"] != image_id:
                continue
            gcp = next((g for g in state["gcps"] if g["id"] == record["pointId"]), None)
            if gcp and gcp.get("x") is not None and gcp.get("y") is not None:
                seeds.append((record["col"], record["row"], gcp["x"], gcp["y"]))
        return seeds

    def work(progress, should_cancel):
        # Photos with nothing to search from are placed on the reference
        # first, so control can be found without measuring any by hand.
        unplaced = [project.image(i) for i in image_ids
                    if not project.image(i).get("exterior") and len(seeds_for(i)) < 3
                    and not (project.image(i).get("placement") or {}).get("placed")]
        if unplaced:
            _place_and_store(project, unplaced, reference, control_crs,
                             lambda f, m: progress(0.25 * f, m), should_cancel)

        outcomes = []
        for index, image_id in enumerate(image_ids):
            if should_cancel():
                raise InterruptedError("Cancelled")
            image = project.image(image_id)

            # Points already measured on this photo seed the search when the
            # block has not been adjusted yet; failing that, its placement.
            seeds = seeds_for(image_id)
            seed_source = "manual control"
            if len(seeds) < 3 and (image.get("placement") or {}).get("placed"):
                seeds = [tuple(s) for s in image["placement"]["seeds"]]
                seed_source = "placement"

            def sub(fraction, message, index=index, name=image.get("name")):
                progress(0.25 * bool(unplaced) + (1 - 0.25 * bool(unplaced))
                         * (index + fraction) / len(image_ids), f"{name} — {message}")

            try:
                result = auto_control.match_ground_control(
                    image, camera_dict, reference, dem_path, seeds, options,
                    sub, should_cancel, control_crs=control_crs, seed_source=seed_source,
                )
            except InterruptedError:
                raise
            except Exception as exc:
                outcomes.append({"imageId": image_id, "name": image.get("name"),
                                 "error": str(exc), "proposals": []})
                continue

            entry = result.to_dict()
            entry["imageId"] = image_id
            entry["name"] = image.get("name")
            outcomes.append(entry)

        return {"results": outcomes}

    label = f"Matching control on {len(image_ids)} photo" + ("s" if len(image_ids) != 1 else "")
    job = jobs.submit("autogcp", label, work, meta={"imageIds": image_ids})
    return {"job": job.to_dict()}


def _photo_mapping(image: dict, state: dict, camera: CameraModel):
    """(pixel -> ground at a height, ground -> pixel) for a photo, or None.

    Rigorous through the solved orientation when there is one; otherwise the
    plane mapping of :func:`_ground_mapping`, which ignores height.
    """
    solve_failed = (state.get("model") or {}).get("quality") == "failed"
    if image.get("exterior") and not solve_failed:
        fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
        eo = np.asarray(image["exterior"], dtype=float)

        def to_ground(cols, rows, zs):
            ground = ortho._pixels_to_ground(np.column_stack([cols, rows]).astype(float),
                                             camera, fit, eo, np.asarray(zs, dtype=float))
            return ground[:, 0], ground[:, 1]

        def to_pixel(xs, ys, zs):
            pixels, ahead = ortho._ground_to_pixels(np.column_stack([xs, ys, zs]), camera, fit, eo)
            pixels[~ahead] = np.nan
            return pixels[:, 0], pixels[:, 1]

        return to_ground, to_pixel, True

    plane = _ground_mapping(image, state)
    if plane is None:
        return None
    return (lambda cols, rows, zs: plane[0](cols, rows),
            lambda xs, ys, zs: plane[1](xs, ys), False)


def _transfer_control(project: Project, point_ids, min_score: float = 0.7) -> int:
    """Measure control points on every photo that sees them.

    An automatic control point is found on one photo at a time. Seen on one
    photo only, its height cannot be checked and it ties nothing to the
    neighbours; seen on all of them it does both. Each point is predicted on
    the other photos through the orientation (or placement), then measured
    there by correlation and least-squares matching against the photo it was
    found on, warped through the ground at the point's height.
    """
    import rasterio

    state = project.state
    camera = CameraModel.from_dict(state.get("camera") or {})
    images = {i["id"]: i for i in project.images_online()}
    gcps = {g["id"]: g for g in state.get("gcps", [])}
    by_point: dict[str, list] = {}
    for o in state.get("observations", []):
        by_point.setdefault(o["pointId"], []).append(o)
    mappings = {iid: _photo_mapping(img, state, camera) for iid, img in images.items()}

    datasets: dict = {}

    def dataset(image_id):
        if image_id not in datasets:
            datasets[image_id] = rasterio.open(images[image_id]["path"])
        return datasets[image_id]

    found = []
    try:
        for pid in point_ids:
            gcp = gcps.get(pid)
            observed = by_point.get(pid) or []
            if not gcp or not observed or any(gcp.get(k) is None for k in ("x", "y", "z")):
                continue
            source = observed[0]
            map_a = mappings.get(source["imageId"])
            if map_a is None or source["imageId"] not in images:
                continue
            seen = {o["imageId"] for o in observed}
            x, y, z = float(gcp["x"]), float(gcp["y"]), float(gcp["z"])
            c, r = int(round(source["col"])), int(round(source["row"]))
            ga_x, ga_y = map_a[0](np.array([c, c + 1.0, c]), np.array([r, r, r + 1.0]), np.full(3, z))
            for target_id, target in images.items():
                mapping = mappings.get(target_id)
                if target_id in seen or mapping is None:
                    continue
                pc, pr = mapping[1](np.array([x]), np.array([y]), np.array([z]))
                margin = 60
                width, height = float(target.get("width") or 0), float(target.get("height") or 0)
                if not (np.isfinite(pc[0]) and margin < pc[0] < width - margin
                        and margin < pr[0] < height - margin):
                    continue
                bc, br = mapping[1](ga_x, ga_y, np.full(3, z))
                jac = np.array([[bc[1] - bc[0], bc[2] - bc[0]], [br[1] - br[0], br[2] - br[0]]])
                left_ds, right_ds = dataset(source["imageId"]), dataset(target_id)
                band = 1 if left_ds.count < 3 else 2
                result = tiepoints.refine_point(
                    left_ds, right_ds, band, c, r, (float(bc[0]), float(br[0])), jac,
                    half=20, search=12 if mapping[2] else 60, min_score=min_score)
                if result is None:
                    continue
                col, row, correlation, sigma = result
                # From the integer pixel matched to the point's own subpixel spot.
                offset = np.array([source["col"] - c, source["row"] - r])
                col += float(jac[0] @ offset)
                row += float(jac[1] @ offset)
                found.append({"pointId": pid, "imageId": target_id, "col": col, "row": row,
                              "kind": "gcp", "weight": float(source.get("weight", 1.0)),
                              "sigmaPx": sigma, "transferred": True,
                              "correlation": correlation})
    finally:
        for handle in datasets.values():
            handle.close()

    tried = set(point_ids)

    def apply(state: dict) -> None:
        existing = {(o["pointId"], o["imageId"]) for o in state["observations"]}
        state["observations"].extend(
            f for f in found if (f["pointId"], f["imageId"]) not in existing)
        # Remembered, so a point no other photo sees is not offered again.
        for gcp in state.get("gcps", []):
            if gcp["id"] in tried:
                gcp["transferTried"] = True
    project.mutate("control_transfer", apply)
    return len(found)


@app.post("/auto/gcps/transfer")
def auto_gcps_transfer(payload: dict = Body(default={})) -> dict:
    """Measure existing control points on every other photo that sees them."""
    project = require_project()
    ids = payload.get("pointIds") or [g["id"] for g in project.state.get("gcps", [])
                                       if g.get("source") == "automatic"]
    added = _transfer_control(project, ids)
    broadcast({"type": "project.changed", "operation": "control_transfer"})
    return clean({"added": added, "state": project.state})


@app.post("/auto/gcps/apply")
def auto_gcps_apply(payload: dict = Body(...)) -> dict:
    """Accept chosen proposals as real control points."""
    project = require_project()
    sigma = _automatic_control_sigma(project.state)

    def apply(state: dict) -> list:
        # A second pass after the model is solved finds the same features far
        # more precisely; its points replace the first pass's rather than
        # sitting beside them as near-duplicates.
        if payload.get("replaceAutomatic"):
            images = {entry["imageId"] for entry in payload.get("accepted", [])}
            stale = {o["pointId"] for o in state["observations"] if o["imageId"] in images}
            stale &= {g["id"] for g in state["gcps"] if g.get("source") == "automatic"}
            state["gcps"] = [g for g in state["gcps"] if g["id"] not in stale]
            state["observations"] = [o for o in state["observations"] if o["pointId"] not in stale]
        created = []
        index = len(state["gcps"]) + 1
        observed_on = {}
        for o in state["observations"]:
            observed_on.setdefault(o["pointId"], set()).add(o["imageId"])
        automatic = [g for g in state["gcps"] if g.get("source") == "automatic"]
        for entry in payload.get("accepted", []):
            # The same feature found from a neighbouring photo is the same
            # control point, measured once more, not a second point beside it.
            same = next((g for g in automatic
                         if entry["imageId"] not in observed_on.get(g["id"], set())
                         and math.hypot(g["x"] - float(entry["x"]), g["y"] - float(entry["y"]))
                         < MERGE_CONTROL_M), None)
            if same is not None:
                state["observations"].append({
                    "pointId": same["id"], "imageId": entry["imageId"],
                    "col": float(entry["col"]), "row": float(entry["row"]),
                    "kind": "gcp", "weight": float(entry.get("score", 1.0)),
                })
                observed_on.setdefault(same["id"], set()).add(entry["imageId"])
                continue
            while any(g["id"] == f"A{index:04d}" for g in state["gcps"]):
                index += 1
            point_id = f"A{index:04d}"
            record = {
                "id": point_id,
                "x": float(entry["x"]), "y": float(entry["y"]), "z": float(entry["z"]),
                "isCheckPoint": bool(entry.get("isCheckPoint", False)),
                "source": "automatic",
                "note": f"Matched at {float(entry.get('score', 0)):.2f} correlation",
            }
            if sigma:
                record["sigma"] = list(sigma)
            state["gcps"].append(record)
            automatic.append(record)
            state["observations"].append({
                "pointId": point_id,
                "imageId": entry["imageId"],
                "col": float(entry["col"]), "row": float(entry["row"]),
                "kind": "gcp",
                "weight": float(entry.get("score", 1.0)),
            })
            observed_on.setdefault(point_id, set()).add(entry["imageId"])
            created.append(point_id)
            index += 1
        return created

    created = project.mutate("autogcp_apply", apply)
    transferred = 0
    if payload.get("transfer", True) and created:
        transferred = _transfer_control(project, created)
    return clean({"created": created, "transferred": transferred, "state": project.state})


# Automatic control proposals closer than this from different photos are one
# feature: a few reference pixels.
MERGE_CONTROL_M = 1.5


def _automatic_control_sigma(state: dict):
    """Standard deviations (X, Y, Z) of control taken from reference data.

    Position: a reference pixel, which is what correlation against it can be
    trusted to (subpixel peaks notwithstanding, the reference's own accuracy
    is rarely better). Height: read off the DEM at a feature, where a cell
    averages whatever surrounds it -- roof and ground, tree and road -- so
    half a cell, and never under half a metre. None if neither is known.
    """
    dem = state.get("dem") or {}

    def cell(path):
        if not path:
            return None
        try:
            info = raster.raster_info(path)
            transform = info.to_dict().get("transform") or []
            size = abs(float(transform[0])) if transform else None
        except Exception:  # noqa: BLE001
            return None
        # A geographic raster's cell is in degrees.
        return size * 111_000.0 if size is not None and size < 0.01 else size

    reference = cell(dem.get("referenceImage"))
    height = cell(dem.get("referencePath"))
    if reference is None and height is None:
        return None
    horizontal = max(reference or 0.5, 0.1)
    vertical = max(0.5, 0.5 * height) if height else max(1.0, horizontal)
    return (horizontal, horizontal, vertical)


@app.post("/auto/reference-image")
def auto_reference_image(payload: dict = Body(...)) -> dict:
    """Record which geocoded image ground control is matched against.

    Several tiles of one orthomosaic (``paths``) are joined into one virtual
    image in the project's cache, so they are searched as one.
    """
    project = require_project()
    paths = [p for p in (payload.get("paths") or []) if p]
    if not paths and payload.get("path"):
        paths = [payload["path"]]
    for each in paths:
        try:
            raster.raster_info(each)
        except Exception as exc:
            raise HTTPException(400, f"Could not open {each}: {exc}") from exc

    path = paths[0] if len(paths) == 1 else None
    if len(paths) > 1:
        try:
            path = placement.build_reference_vrt(paths, str(project.cache_dir / "reference.vrt"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def apply(state: dict) -> None:
        dem = state.setdefault("dem", {})
        dem["referenceImage"] = path
        dem["referenceImages"] = paths
        # A new reference invalidates placements made against the old one.
        for image in state.get("images", []):
            image.pop("placement", None)

    project.mutate("reference_image", apply)
    return clean({"state": project.state})


@app.post("/auto/place")
def auto_place(payload: dict = Body(default={})) -> dict:
    """Find where each photo lies on the reference image, with no clicks."""
    project = require_project()
    state = project.state
    reference = (state.get("dem") or {}).get("referenceImage")
    if not reference:
        raise HTTPException(400, "No reference image set. Choose the geocoded reference first.")
    projection = state.get("projection") or {}
    control_crs = projection.get("gcpSource") or projection.get("output")
    image_ids = payload.get("imageIds") or [i["id"] for i in project.images_online()]
    images = [project.image(i) for i in image_ids]

    def work(progress, should_cancel):
        return {"placements": _place_and_store(project, images, reference, control_crs,
                                               progress, should_cancel)}

    job = jobs.submit("place", f"Placing {len(images)} photos on the reference", work)
    return {"job": job.to_dict()}


def _place_and_store(project, images, reference, control_crs, progress, should_cancel) -> list:
    """Place photos and keep the result on each image, for later searches."""
    results = placement.place_photos(images, reference, control_crs, progress, should_cancel)
    by_id = {r.image_id: r.to_dict() for r in results}

    def apply(state: dict) -> None:
        for image in state.get("images", []):
            if image["id"] in by_id:
                image["placement"] = by_id[image["id"]]

    project.mutate("placement", apply)
    return list(by_id.values())


# -- calibration certificate -----------------------------------------------


@app.get("/certificate/status")
def certificate_status() -> dict:
    """Whether certificate reading is usable, and what is missing if not."""
    return certificate_reader.available()


@app.post("/certificate/settings")
def certificate_settings(payload: dict = Body(...)) -> dict:
    """Choose the reading service and supply keys.

    Keys are held in memory for the life of the engine and never written by
    it. The response reports whether a key is present, never the key itself.
    """
    try:
        certificate_reader.configure(
            provider=payload.get("provider"),
            keys=payload.get("keys"),
            models=payload.get("models"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return certificate_reader.available()


def _match_sensor_to_images(result: dict, sizes: list[tuple[int, int]]) -> dict:
    """Swap a digital sensor's columns and rows when the photos are rotated.

    Certificates such as UltraCam's describe the sensor in its own orientation
    (20010 columns by 13080 rows), while the delivered photos are often turned
    a quarter (13080 by 20010). Applied as printed, the camera would contradict
    every image. Only the size is swapped; a non-zero principal point offset
    also depends on the direction of rotation, so that is left to the operator
    with a note.
    """
    camera = result.get("camera") or {}
    columns, rows = int(camera.get("columns") or 0), int(camera.get("rows") or 0)
    if camera.get("kind") != "digital" or not columns or not rows or not sizes:
        return result
    rotated = sum(1 for w, h in sizes if (w, h) == (rows, columns))
    upright = sum(1 for w, h in sizes if (w, h) == (columns, rows))
    if rotated and rotated > upright:
        fixes, detail = plausibility.turned_camera_fixes(camera, rows, columns)
        if len(fixes) == 1:
            camera.update(fixes[0]["fix"])
            note = (f"Your photos are {rows} by {columns} pixels, the certificate's "
                    f"{columns} by {rows} turned a quarter, so the camera was turned "
                    "to match them." + detail)
        else:
            # The direction of the turn changes the principal point; leave it
            # to the camera check, which offers each direction as a button.
            note = (f"Your photos are {rows} by {columns} pixels, the certificate's "
                    f"{columns} by {rows} turned a quarter. After applying, choose "
                    "the rotation they were delivered with in the camera check, "
                    "since the principal point differs between them.")
        result.setdefault("review", []).append(note)
    return result


@app.post("/certificate/read")
def certificate_read(payload: dict = Body(...)) -> dict:
    """Read a calibration certificate into camera parameters.

    Returns proposals only. Nothing is written to the project until the
    operator accepts them, which is the point -- the model is a transcriber,
    not an authority.
    """
    path = payload.get("path")
    if not path:
        raise HTTPException(400, "No certificate supplied")

    model = payload.get("model")
    provider = payload.get("provider")

    # The sizes of the open project's photos, to match a digital sensor to.
    sizes = []
    if _current is not None:
        sizes = [(int(i.get("width") or 0), int(i.get("height") or 0))
                 for i in _current.state.get("images", []) if i.get("width")]

    def work(progress, should_cancel):
        result = certificate_reader.read_certificate(path, model, progress, provider)
        return _match_sensor_to_images(result, sizes)

    job = jobs.submit(
        "certificate", f"Reading {Path(path).name}", work, meta={"path": path}
    )
    return {"job": job.to_dict()}


# -- control and tie points ------------------------------------------------


@app.post("/points/gcp")
def gcp_upsert(payload: dict = Body(...)) -> dict:
    """Create or update a ground control point and its image measurements."""
    project = require_project()

    def apply(state: dict) -> dict:
        point_id = payload.get("id")
        if not point_id:
            index = len([g for g in state["gcps"]]) + 1
            while any(g["id"] == f"G{index:04d}" for g in state["gcps"]):
                index += 1
            point_id = f"G{index:04d}"

        record = next((g for g in state["gcps"] if g["id"] == point_id), None)
        if record is None:
            record = {"id": point_id, "isCheckPoint": False, "note": ""}
            state["gcps"].append(record)

        for key in ("x", "y", "z", "isCheckPoint", "note", "source", "sigma"):
            if key in payload:
                record[key] = payload[key]

        for measurement in payload.get("measurements", []):
            existing = next(
                (
                    o for o in state["observations"]
                    if o["pointId"] == point_id and o["imageId"] == measurement["imageId"]
                ),
                None,
            )
            entry = {
                "pointId": point_id,
                "imageId": measurement["imageId"],
                "col": measurement["col"],
                "row": measurement["row"],
                "kind": "gcp",
                "weight": measurement.get("weight", 1.0),
            }
            if existing:
                existing.update(entry)
            else:
                state["observations"].append(entry)

        return record

    record = project.mutate("gcp_upsert", apply)
    return clean({"gcp": record, "state": project.state})


@app.delete("/points/gcp/{point_id}")
def gcp_delete(point_id: str) -> dict:
    project = require_project()

    def apply(state: dict) -> None:
        state["gcps"] = [g for g in state["gcps"] if g["id"] != point_id]
        state["observations"] = [o for o in state["observations"] if o["pointId"] != point_id]

    project.mutate("gcp_delete", apply)
    return clean({"state": project.state})


@app.delete("/points/observation")
def observation_delete(pointId: str = Query(...), imageId: str = Query(...)) -> dict:
    project = require_project()

    def apply(state: dict) -> None:
        state["observations"] = [
            o for o in state["observations"]
            if not (o["pointId"] == pointId and o["imageId"] == imageId)
        ]

    project.mutate("observation_delete", apply)
    return clean({"state": project.state})


def _ground_mapping(image: dict, state: dict):
    """(pixel -> ground, ground -> pixel) for a photo, or None if unplaced.

    From the solved orientation where there is one (at the control's mean
    height); otherwise a plane fitted to the photo's control points, or to its
    automatic placement. Good to tens of metres, which is all a search window
    needs.
    """
    import cv2

    heights = [float(g["z"]) for g in state.get("gcps", []) if isinstance(g.get("z"), (int, float))]
    mean_height = sum(heights) / len(heights) if heights else 0.0

    # An orientation from a solve that failed is worse than none: it would
    # steer the search to the wrong place.
    solve_failed = (state.get("model") or {}).get("quality") == "failed"
    if image.get("exterior") and not solve_failed:
        camera = CameraModel.from_dict(state.get("camera") or {})
        fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
        eo = np.asarray(image["exterior"], dtype=float)

        def to_ground(cols, rows):
            pixels = np.column_stack([cols, rows]).astype(float)
            ground = ortho._pixels_to_ground(pixels, camera, fit, eo, mean_height)
            return ground[:, 0], ground[:, 1]

        def to_pixel(xs, ys):
            ground = np.column_stack([xs, ys, np.full(len(xs), mean_height)])
            pixels, ahead = ortho._ground_to_pixels(ground, camera, fit, eo)
            pixels[~ahead] = np.nan
            return pixels[:, 0], pixels[:, 1]

        return to_ground, to_pixel

    # The placement's seeds cover the whole frame evenly; control points may
    # cluster, and a plane fitted to a cluster extrapolates badly to the
    # corners. So the placement first, control only without one.
    pairs = []
    if (image.get("placement") or {}).get("placed"):
        pairs = [((s[0], s[1]), (s[2], s[3])) for s in image["placement"]["seeds"]]
    if len(pairs) < 4:
        gcps = {g["id"]: g for g in state.get("gcps", [])}
        pairs = [((o["col"], o["row"]), (gcps[o["pointId"]]["x"], gcps[o["pointId"]]["y"]))
                 for o in state.get("observations", [])
                 if o["imageId"] == image["id"] and o["pointId"] in gcps
                 and gcps[o["pointId"]].get("x") is not None]
    if len(pairs) < 4:
        return None
    src = np.float64([p[0] for p in pairs])
    dst = np.float64([p[1] for p in pairs])
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC if len(pairs) > 8 else 0, 30.0)
    if homography is None:
        return None
    inverse = np.linalg.inv(homography)

    def apply(h, a, b):
        points = np.column_stack([a, b, np.ones(len(a))]) @ h.T
        return points[:, 0] / points[:, 2], points[:, 1] / points[:, 2]

    return (lambda cols, rows: apply(homography, cols, rows),
            lambda xs, ys: apply(inverse, xs, ys))


def _footprint_overlap(a: dict, b: dict, map_a, map_b):
    """Shared ground area as a share of the smaller photo, or None if unknown."""
    import cv2

    if map_a is None or map_b is None:
        return None

    def footprint(image, mapping):
        w, h = float(image.get("width") or 0), float(image.get("height") or 0)
        xs, ys = mapping[0](np.array([0, w, w, 0]), np.array([0, 0, h, h]))
        return np.float32(np.column_stack([xs, ys]))

    fa, fb = footprint(a, map_a), footprint(b, map_b)
    if not (np.isfinite(fa).all() and np.isfinite(fb).all()):
        return None
    area, _ = cv2.intersectConvexConvex(cv2.convexHull(fa), cv2.convexHull(fb))
    smaller = min(cv2.contourArea(cv2.convexHull(fa)), cv2.contourArea(cv2.convexHull(fb)))
    return float(area / smaller) if smaller > 0 else None


def _pair_predictor(map_left, map_right):
    """Left-photo pixel -> where the same ground falls on the right photo."""
    if map_left is None or map_right is None:
        return None

    def predict(col, row):
        xs, ys = map_left[0](np.array([col]), np.array([row]))
        cs, rs = map_right[1](xs, ys)
        return float(cs[0]), float(rs[0])

    return predict


@app.post("/points/tie/auto")
def tie_points_auto(payload: dict = Body(...)) -> dict:
    """Queue automatic tie point collection across image pairs."""
    project = require_project()
    refine = payload.get("refine", "lsm")
    if refine not in tiepoints.REFINE_METHODS:
        raise HTTPException(400, f"Unknown refinement {refine!r}")
    options = tiepoints.TiePointOptions(
        method=payload.get("method", "ncc"),
        target_count=int(payload.get("targetCount", 60)),
        min_score=float(payload.get("minScore", 0.75)),
        search_radius=int(payload.get("searchRadius", 220)),
        template_size=int(payload.get("templateSize", 31)),
        refine=refine,
        max_sigma_px=float(payload.get("maxSigmaPx", 0.5)),
    )
    # A second collection measures many of the same features again. Kept
    # alongside the first, each would count twice in the adjustment.
    replace = bool(payload.get("replaceAutomatic", True))

    images = [i for i in project.images_online()]
    pair_mode = payload.get("pairs", "overlapping")
    explicit = payload.get("imagePairs")

    # Where each photo lies on the ground, from the best available source, so
    # the search starts where the point actually is and only overlapping pairs
    # are tried. Without it the search starts at the same pixel in both photos,
    # which on a flight strip is a whole photo-spacing away from the truth.
    mappings = {i["id"]: _ground_mapping(i, project.state) for i in images}

    if explicit:
        pairs = [(p["left"], p["right"]) for p in explicit]
    else:
        pairs = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                a, b = images[i], images[j]
                overlap = _footprint_overlap(a, b, mappings[a["id"]], mappings[b["id"]])
                # Unknown overlap (neither photo placed yet): try the pair.
                if overlap is None or overlap >= 0.08:
                    pairs.append((a["id"], b["id"]))

    lookup = {i["id"]: i for i in images}

    # Each photo is the master for the pairs it leads, so a feature it shares
    # with two others is matched into both and becomes one three-ray point.
    masters: dict[str, list[str]] = {}
    for left_id, right_id in pairs:
        if left_id in lookup and right_id in lookup:
            masters.setdefault(left_id, []).append(right_id)

    def work(progress, should_cancel):
        collected: list = []
        done = 0
        for master_id, partner_ids in masters.items():
            if should_cancel():
                raise InterruptedError("Cancelled")
            master = lookup[master_id]
            partners = [{
                "id": pid, "path": lookup[pid]["path"],
                "predictor": _pair_predictor(mappings.get(master_id), mappings.get(pid)),
            } for pid in partner_ids]

            def sub_progress(fraction, message, done=done, count=len(partner_ids)):
                progress((done + fraction * count) / max(len(pairs), 1),
                         f"{master['name']} — {message}")

            collected.extend(tiepoints.collect_multi(
                {"id": master_id, "path": master["path"]}, partners, options, sub_progress))
            done += len(partner_ids)

        def apply(state: dict) -> None:
            if replace:
                old = {t["id"] for t in state["tiePoints"]
                       if t.get("source") in tiepoints.MATCH_METHODS}
                state["tiePoints"] = [t for t in state["tiePoints"] if t["id"] not in old]
                state["observations"] = [o for o in state["observations"]
                                         if o["pointId"] not in old]
            taken = {t["id"] for t in state["tiePoints"]}
            number = 0
            for point in collected:
                number += 1
                while f"T{number:04d}" in taken:
                    number += 1
                point_id = f"T{number:04d}"
                taken.add(point_id)
                state["tiePoints"].append({
                    "id": point_id, "score": point.score, "source": options.method,
                    "refine": options.refine, "rays": point.rays,
                })
                for image_id, col, row, sigma in point.observations:
                    entry = {"pointId": point_id, "imageId": image_id,
                             "col": float(col), "row": float(row), "kind": "tie"}
                    if sigma is not None:
                        entry["sigmaPx"] = float(sigma)
                    state["observations"].append(entry)

        project.mutate("tie_points_auto", apply)
        broadcast({"type": "project.changed", "operation": "tie_points_auto"})
        rays = [p.rays for p in collected]
        return {"collected": len(collected), "pairs": len(pairs),
                "multiRay": sum(1 for r in rays if r >= 3),
                "observations": int(sum(rays))}

    job = jobs.submit("tiepoints", f"Collecting tie points ({options.method.upper()})", work)
    return {"job": job.to_dict()}


# A drone block's adjustment: GNSS positions are metres-level, attitudes from
# the gimbal are too rough to weigh, SIFT points are ~1 px, and the consumer
# lens needs its distortion estimated. Its focal length is held: without
# control a nadir block cannot separate it from the depth of the scene.
DRONE_ADJUSTMENT = {"useExteriorObservations": True, "eoSigmaXY": 3.0, "eoSigmaZ": 5.0,
                    "eoSigmaAngleDeg": None, "tieSigmaPx": 0.8, "selfCalibration": "lens",
                    "lensPriors": "consumer"}


@app.post("/drone/align")
def drone_align(payload: dict = Body(default={})) -> dict:
    """Align a GPS-tagged drone block: camera, starting orientations and tie points."""
    from pyproj import Transformer
    from fiducia import drone

    project = require_project()
    images = [i for i in project.images_online()]
    if len(images) < 2:
        raise HTTPException(400, "Add at least two drone photos first")
    replace_camera = bool(payload.get("replaceCamera", False))

    def work(progress, should_cancel):
        progress(0.0, "Reading photo metadata")
        metas = {i["id"]: drone.read_metadata(i["path"]) for i in images}
        placed = [i for i in images if metas[i["id"]].has_position]
        missing = [i["name"] for i in images if not metas[i["id"]].has_position]
        if len(placed) < 2:
            raise ValueError("Fewer than two photos carry a GPS position")

        state = project.state
        camera = state.get("camera") or {}
        camera_note = None
        if replace_camera or not (camera.get("kind") == "digital" and camera.get("focalMm")
                                  and camera.get("pixelPitchMm")):
            found = drone.camera_from_metadata([metas[i["id"]] for i in placed])
            camera, camera_note = found["camera"], found["note"]
        focal, pitch = float(camera["focalMm"]), float(camera["pixelPitchMm"])

        projection = state.get("projection") or {}
        first = metas[placed[0]["id"]]
        crs = projection.get("output") or drone.suggest_crs(first.latitude, first.longitude)
        to_map = Transformer.from_crs("EPSG:4326", geodesy.resolve_crs(crs), always_xy=True)

        heights = [m.relative_altitude for m in metas.values() if m.relative_altitude]
        # Without a recorded height above ground, a generous guess only widens
        # the search for overlapping pairs; the tie points then measure it.
        height = float(np.median(heights)) if heights else 60.0
        footprint = max(int(camera["columns"]), int(camera["rows"])) * pitch * height / focal

        ground = {i["id"]: to_map.transform(metas[i["id"]].longitude, metas[i["id"]].latitude)
                  for i in placed}
        photos = [{"id": i["id"], "name": i["name"], "path": i["path"],
                   "position": ground[i["id"]]} for i in placed]
        points, summary = drone.collect_tracks(
            photos, focal, pitch, footprint,
            progress=lambda f, m: progress(0.05 + 0.9 * f, m), should_cancel=should_cancel)

        headings, measured = drone.heading_and_height(photos, points, focal / pitch)
        height_source = "photo metadata"
        if not heights and measured:
            height, height_source = measured, "tie points"
        gsd = pitch * height / focal
        footprint = max(int(camera["columns"]), int(camera["rows"])) * gsd
        ground_z = float(np.median([metas[i["id"]].altitude for i in placed])) - height

        exterior, heading_source = {}, set()
        for image in placed:
            meta = metas[image["id"]]
            yaw = meta.gimbal_yaw
            if yaw is not None:
                heading_source.add("gimbal")
            elif image["id"] in headings:
                yaw = headings[image["id"]]
                heading_source.add("tie points")
            else:
                yaw = meta.flight_yaw or 0.0
                heading_source.add("flight direction" if meta.flight_yaw is not None else "assumed north")
            omega, phi, kappa = drone.attitude_from_gimbal(
                yaw, meta.gimbal_pitch if meta.gimbal_pitch is not None else -90.0)
            x, y = ground[image["id"]]
            exterior[image["id"]] = [float(x), float(y), float(meta.altitude), omega, phi, kappa]

        def apply(state: dict) -> None:
            state["camera"] = camera
            proj = state.setdefault("projection", {})
            if not proj.get("output"):
                # A new project: its blank defaults are for scanned film.
                proj["output"] = crs
                proj["pixelSpacingX"] = proj["pixelSpacingY"] = round(2 * gsd, 3)
            for image in state["images"]:
                eo = exterior.get(image["id"])
                if eo:
                    image["exterior"] = list(eo)
                    image["exteriorObserved"] = list(eo)
                    image["exteriorSource"] = "gnss"
            old = {t["id"] for t in state["tiePoints"] if t.get("source") in tiepoints.MATCH_METHODS}
            state["tiePoints"] = [t for t in state["tiePoints"] if t["id"] not in old]
            state["observations"] = [o for o in state["observations"] if o["pointId"] not in old]
            taken = {t["id"] for t in state["tiePoints"]}
            number = 0
            for point in points:
                number += 1
                while f"T{number:05d}" in taken:
                    number += 1
                point_id = f"T{number:05d}"
                taken.add(point_id)
                state["tiePoints"].append({"id": point_id, "score": point.score,
                                           "source": "sift", "refine": "none", "rays": point.rays})
                for image_id, col, row, sigma in point.observations:
                    entry = {"pointId": point_id, "imageId": image_id,
                             "col": float(col), "row": float(row), "kind": "tie"}
                    if sigma is not None:
                        entry["sigmaPx"] = float(sigma)
                    state["observations"].append(entry)
            adjustment = state.setdefault("adjustment", {})
            for key, value in DRONE_ADJUSTMENT.items():
                adjustment.setdefault(key, value)
            state["mathModel"] = {**(state.get("mathModel") or {}), "kind": "aerial_digital"}
            # Kept so the Imagery step can show what the alignment found.
            state["droneBlock"] = {
                "flyingHeightM": round(height, 2), "heightSource": height_source,
                "groundZ": round(ground_z, 2), "crs": crs, "gsdM": round(gsd, 4),
                "headingSource": sorted(heading_source), "photos": len(placed),
                "withoutPosition": missing[:12], "tiePoints": summary["tiePoints"],
                "multiRay": summary["multiRay"], "pairsMatched": summary["pairsMatched"],
                "pairsTried": summary["pairsTried"], "alignedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        project.mutate("drone_align", apply)
        broadcast({"type": "project.changed", "operation": "drone_align"})
        return {**summary, "crs": crs, "flyingHeightM": round(height, 1),
                "footprintM": round(footprint, 1), "gsdM": round(gsd, 4),
                "heightSource": height_source, "headingSource": sorted(heading_source),
                "cameraNote": camera_note, "withoutPosition": missing[:12]}

    job = jobs.submit("drone", "Aligning drone photos", work)
    return {"job": job.to_dict()}


@app.post("/drone/dense")
def drone_dense(payload: dict = Body(default={})) -> dict:
    """Dense surface model of a solved drone block, from its own photos."""
    from fiducia import drone

    project = require_project()
    state = project.state
    block = state.get("droneBlock") or {}
    if not (block.get("surface") or {}).get("path"):
        raise HTTPException(409, "Solve the drone block on the Model step first")
    detail = payload.get("detail", "medium")
    if detail not in drone.DENSE_DETAIL_PX:
        raise HTTPException(400, f"Unknown detail {detail!r}")
    crs = (state.get("projection") or {}).get("output") or block.get("crs")
    images = [dict(i) for i in project.images_online()]
    camera = dict(state.get("camera") or {})
    observations = list(state.get("observations") or [])
    work_dir = str(Path(project.cache_dir) / "dense")
    output_path = str(Path(project.outputs_dir) / "drone_dsm.tif")

    def work(progress, should_cancel):
        raster.close_raster(output_path)   # may be open in the viewer
        info = drone.dense_surface(images, camera, observations, block, crs, work_dir, output_path,
                                   detail=detail, progress=progress, should_cancel=should_cancel)
        info["builtAt"] = time.strftime("%Y-%m-%d %H:%M:%S")

        def apply(state: dict) -> None:
            current = state.setdefault("droneBlock", {})
            previous = {(current.get("surface") or {}).get("path"),
                        (current.get("dense") or {}).get("path")}
            current["dense"] = info
            dem = state.setdefault("dem", {})
            # In place of our own tie-point surface, never over the operator's DEM.
            if not dem.get("referencePath") or dem.get("referencePath") in previous:
                dem["referencePath"] = info["path"]

        project.mutate("drone_dense", apply)
        broadcast({"type": "project.changed", "operation": "drone_dense"})
        return info

    job = jobs.submit("drone_dense", "Building the dense surface", work)
    return {"job": job.to_dict()}


@app.delete("/points/tie/{point_id}")
def tie_delete(point_id: str) -> dict:
    project = require_project()

    def apply(state: dict) -> None:
        state["tiePoints"] = [t for t in state["tiePoints"] if t["id"] != point_id]
        state["observations"] = [o for o in state["observations"] if o["pointId"] != point_id]

    project.mutate("tie_delete", apply)
    return clean({"state": project.state})


# -- model -----------------------------------------------------------------


def _film_observations(project: Project) -> tuple[list[Observation], dict, list[str]]:
    """Reduce pixel measurements to focal-plane mm, ready for adjustment.

    With the certificate camera only: a self-calibration from an earlier
    solve is what the adjustment estimates afresh, not something to correct
    the measurements by first.
    """
    camera = CameraModel.from_dict(project.state.get("camera") or {}).without_adjustment()
    observations: list[Observation] = []
    focal_by_image: dict[str, float] = {}
    problems: list[str] = []

    images = {i["id"]: i for i in project.state.get("images", [])}

    for record in project.state.get("observations", []):
        image = images.get(record["imageId"])
        if image is None:
            continue

        fit = None
        if camera.kind == "film":
            if not image.get("fiducialFit"):
                message = f"{image.get('name')} has no interior orientation yet"
                if message not in problems:
                    problems.append(message)
                continue
            fit = FiducialFit.from_dict(image["fiducialFit"])

        try:
            film = camera.pixel_to_film([[record["col"], record["row"]]], fit)[0]
        except ValueError as exc:
            message = str(exc)
            if message not in problems:
                problems.append(message)
            continue

        observations.append(
            Observation(
                image_id=record["imageId"],
                point_id=record["pointId"],
                x_mm=float(film[0]),
                y_mm=float(film[1]),
                weight=float(record.get("weight", 1.0)),
            )
        )
        focal_by_image[record["imageId"]] = camera.focal_mm

    return observations, focal_by_image, problems


@app.get("/model/readiness")
def model_readiness() -> dict:
    """Can the model be solved, and if not, precisely what is missing?

    The usual answer to an unsolvable block is "bundle adjustment failed".
    This endpoint exists so the interface can say which image, which point, and
    what to do about it — before the operator spends a minute finding out.
    """
    project = require_project()
    state = project.state
    camera = CameraModel.from_dict(state.get("camera") or {})
    blockers: list[dict] = []
    warnings: list[dict] = []

    if not camera.focal_mm:
        blockers.append({"code": "camera", "message": "Camera focal length is not set"})

    if not (state.get("projection") or {}).get("output"):
        blockers.append({"code": "projection", "message": "Output projection is not set"})

    images = state.get("images", [])
    online = [i for i in images if i.get("online")]
    if not online:
        blockers.append({"code": "images", "message": "No images are online"})

    offline = [i for i in images if not i.get("online")]
    if offline:
        warnings.append({
            "code": "offline",
            "message": f"{len(offline)} image(s) offline: "
                       + ", ".join(i.get("name", "?") for i in offline[:4]),
        })

    if camera.kind == "film":
        missing = [i for i in online if not i.get("fiducialFit")]
        if missing:
            blockers.append({
                "code": "fiducials",
                "message": f"Interior orientation missing on "
                           + ", ".join(i.get("name", "?") for i in missing[:4]),
            })
        poor = [
            i for i in online
            if i.get("fiducialFit") and i["fiducialFit"].get("rmsPx", 0) > 2.0
        ]
        for image in poor:
            warnings.append({
                "code": "fiducial_rms",
                "message": f"{image['name']} fiducial RMS is "
                           f"{image['fiducialFit']['rmsPx']:.2f} px (aim for under 2)",
            })

    for problem in plausibility.camera_problems(state.get("camera") or {}, online):
        (blockers if problem["severity"] == "blocker" else warnings).append(problem)

    control_crs = (state.get("projection") or {}).get("gcpSource") \
        or (state.get("projection") or {}).get("output")
    if control_crs:
        try:
            for problem in plausibility.control_crs_problems(
                    state.get("gcps", []), control_crs, (state.get("projection") or {}).get("output")):
                (blockers if problem.get("severity") == "blocker" else warnings).append(problem)
        except Exception:  # noqa: BLE001 -- a check must never stop the solve
            pass
    dem_path = (state.get("dem") or {}).get("referencePath")
    if control_crs and dem_path:
        try:
            dem_crs = raster.raster_info(dem_path).crs
        except Exception:  # noqa: BLE001
            dem_crs = None
        differences = plausibility.crs_differences(control_crs, dem_crs) if dem_crs else []
        if differences:
            warnings.append({
                "code": "crs_dem",
                "message": (
                    "The control projection does not match the reference DEM: "
                    + "; ".join(differences)
                    + ". Elevations are read from the DEM at the coordinates you type, "
                    "so pick the projection that matches the data."
                ),
            })

    gcps = state.get("gcps", [])
    active = [g for g in gcps if not g.get("isCheckPoint")]
    complete = [g for g in active if all(g.get(k) is not None for k in ("x", "y", "z"))]

    # A drone block (or imported GNSS/IMU) is held by its photo positions.
    gnss_held = (
        sum(1 for i in state.get("images", []) if i.get("exteriorObserved")) >= 3
        and bool(state.get("tiePoints"))
        and (state.get("adjustment") or {}).get("useExteriorObservations", True)
    )
    if len(complete) < 3 and gnss_held:
        if not complete:
            warnings.append({
                "code": "gcps_gnss_only",
                "message": "No ground control: the block is held by the photos' GPS positions "
                           "alone, so it is placed only as well as that GPS (metres for most "
                           "drones). Add control for survey accuracy.",
            })
    elif len(complete) < 3:
        blockers.append({
            "code": "gcps",
            "message": f"{len(complete)} of at least 3 ground control points have full XYZ",
        })
    elif len(complete) == 3:
        warnings.append({
            "code": "gcp_redundancy",
            "message": "3 control points gives an exact solution with no residuals. "
                       "Add a 4th to see your error.",
        })

    observations = state.get("observations", [])
    per_image: dict[str, int] = {}
    for record in observations:
        per_image[record["imageId"]] = per_image.get(record["imageId"], 0) + 1

    # Six unknowns per photo, two equations per measurement: under three
    # measurements an orientation simply cannot be determined, so this is a
    # blocker rather than a warning.
    for image in online:
        count = per_image.get(image["id"], 0)
        if count == 0:
            blockers.append({
                "code": "unmeasured",
                "message": f"{image['name']} has no measured points",
            })
        elif count < 3:
            blockers.append({
                "code": "underdetermined",
                "message": (
                    f"{image['name']} has only {count} measurement"
                    f"{'' if count == 1 else 's'} — 3 is the minimum to orient a "
                    "photo. Add control or tie points on it."
                ),
            })
        elif count == 3:
            warnings.append({
                "code": "thin",
                "message": (
                    f"{image['name']} has exactly 3 measurements — an exact "
                    "solution with no redundancy, so its error cannot be assessed."
                ),
            })

    return clean({
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "counts": {
            "images": len(images),
            "online": len(online),
            "gcps": len(active),
            "completeGcps": len(complete),
            "checkPoints": len([g for g in gcps if g.get("isCheckPoint")]),
            "tiePoints": len(state.get("tiePoints", [])),
            "observations": len(observations),
        },
    })


# Photos whose corners are uncertain by more than this many ground pixels are
# flagged: their orthophotos will be misplaced by more than a careful operator
# could see and correct by eye.
CORNER_WARN_PX = 5.0


def _precision_summary(camera: CameraModel, image: dict, eo, covariance,
                       ground_z: float) -> Optional[dict]:
    """Standard deviations and corner uncertainty of one photo's orientation."""
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
        return None
    fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
    if camera.kind == "digital":
        half_w = camera.columns * camera.pixel_pitch_mm / 2
        half_h = camera.rows * camera.pixel_pitch_mm / 2
    else:
        extent = [abs(v) for xy in (camera.fiducials_mm or {}).values() for v in xy]
        half_w = half_h = (max(extent) if extent else 115.0) * 0.95
    corner = corner_uncertainty(eo, cov, camera.focal_mm, half_w, half_h, ground_z)
    pixel_mm = camera.pixel_scale_mm(fit)
    gsd = pixel_mm * (float(eo[2]) - ground_z) / camera.focal_mm if pixel_mm else None
    sigma = [float(np.sqrt(max(cov[k, k], 0.0))) for k in range(6)]
    return {
        "sigma": sigma,                       # X0 Y0 Z0 in m, angles in radians
        "cornerM": corner,
        "cornerPx": corner / gsd if gsd else None,
        "gsdM": gsd,
    }


def _precision_problems(precision: dict, names: dict,
                        control_sigma_m: Optional[float] = None) -> list[str]:
    """Photos determined worse than their control can explain.

    Five photo pixels alone is too strict for fine photography: 0.15 m pixels
    controlled from a 0.5 m reference cannot do better than about a metre at
    the corners however much control there is. So a photo is weak only when
    its corners are also well beyond what the control's precision allows.
    """
    out = []
    floor = 2.5 * control_sigma_m if control_sigma_m else 0.0
    for image_id, entry in precision.items():
        if entry and entry.get("cornerPx") and entry["cornerPx"] > CORNER_WARN_PX \
                and entry.get("cornerM", 0) > floor:
            out.append(
                f"{names.get(image_id, image_id)} is weakly determined: its corners could be "
                f"off by {entry['cornerM']:.1f} m ({entry['cornerPx']:.0f} pixels) on the ground, "
                "1 sigma. Add control on it, spread across the frame, or tie it to its "
                "neighbours before trusting its orthophoto."
            )
    return out


def _refuse_impossible_control_crs(state: dict) -> None:
    """Stop before work whose answer the coordinate systems already ruin."""
    projection = state.get("projection") or {}
    control_crs = projection.get("gcpSource") or projection.get("output")
    if not control_crs:
        return
    try:
        problems = plausibility.control_crs_problems(state.get("gcps", []), control_crs,
                                                     projection.get("output"))
    except Exception:  # noqa: BLE001
        return
    blockers = [p["message"] for p in problems if p.get("severity") == "blocker"]
    if blockers:
        raise HTTPException(400, " ".join(blockers) + " (Project step, Control projection.)")


def _mean_control_height(state: dict) -> float:
    heights = [float(g["z"]) for g in state.get("gcps", []) if g.get("z") is not None]
    if heights:
        return float(np.mean(heights))
    # A drone block without control: the ground its tie points found.
    ground = (state.get("droneBlock") or {}).get("groundZ")
    return float(ground) if ground is not None else 0.0


def _default_image_sigma_mm(project, camera: CameraModel) -> float:
    """Half a pixel of the imagery actually in the project, in millimetres.

    Measurement precision belongs to pixels, not to millimetres: a fixed
    10 um is two pixels on a 5.2 um digital sensor and a quarter-pixel on a
    42 um film scan, which weights the same careful click four-fold
    differently depending on the camera.
    """
    scales = []
    for image in project.state.get("images", []):
        fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
        if camera.kind == "film" and fit is None:
            continue
        scale = camera.pixel_scale_mm(fit)
        if scale and scale != 1.0:
            scales.append(scale)
    return 0.5 * float(np.median(scales)) if scales else 0.010


# Adjustment settings and their defaults. Kept in the project (state
# "adjustment") so a block is always solved the same way, and overridable
# per request.
ADJUSTMENT_DEFAULTS = {
    "controlSigmaXY": 0.5,         # m, horizontal precision of control
    "controlSigmaZ": None,         # m; None = as horizontal
    "imageSigmaPx": 0.5,           # control and check point measurements
    "tieSigmaPx": 0.3,             # automatic tie points, least-squares matched
    "selfCalibration": "none",     # bundle.SELF_CALIBRATION_SETS
    "lensPriors": "metric",        # bundle.LENS_PRIORS
    "useExteriorObservations": True,   # GNSS/IMU, where imported
    "eoSigmaXY": 0.10,             # m
    "eoSigmaZ": 0.15,              # m
    "eoSigmaAngleDeg": None,       # None: attitudes not used
    "gnssShift": False,
    "robust": True,
    "autoRejectTies": True,
}


def _adjustment_settings(state: dict, payload: dict) -> dict:
    settings = {**ADJUSTMENT_DEFAULTS, **(state.get("adjustment") or {})}
    for key in ADJUSTMENT_DEFAULTS:
        if key in payload:
            settings[key] = payload[key]
    # Earlier names, still accepted.
    if payload.get("controlSigma") is not None:
        settings["controlSigmaXY"] = float(payload["controlSigma"])
    if settings.get("selfCalibration") not in bundle_module.SELF_CALIBRATION_SETS:
        raise HTTPException(400, f"Unknown self-calibration {settings.get('selfCalibration')!r}")
    if settings.get("lensPriors") not in bundle_module.LENS_PRIORS:
        raise HTTPException(400, f"Unknown lens priors {settings.get('lensPriors')!r}")
    return settings


def _resection_covariance(result, film: np.ndarray, ground: np.ndarray, focal: float):
    """sigma0^2 (J^T J)^-1 for a single-photo resection."""
    from fiducia.collinearity import collinearity_jacobian

    d_eo, _ = collinearity_jacobian(ground, result.eo, focal)
    jac = d_eo.reshape(2 * len(ground), 6)
    dof = 2 * len(ground) - 6
    if dof <= 0:
        return None
    residual = np.asarray(result.residuals_mm, dtype=float).ravel()
    sigma0_sq = float(residual @ residual) / dof
    try:
        return (sigma0_sq * np.linalg.inv(jac.T @ jac)).tolist()
    except np.linalg.LinAlgError:
        return None


@app.post("/model/preview")
def model_preview(payload: dict = Body(default={})) -> dict:
    """Residuals for one photo from its own control, without saving anything.

    Runs while control is being collected, so a bad point shows up as it is
    entered rather than after the block is finished. A single-photo resection
    takes milliseconds; nothing here touches the project.
    """
    project = require_project()
    state = project.state
    camera = CameraModel.from_dict(state.get("camera") or {})
    image_id = payload.get("imageId")
    image = next((i for i in state.get("images", []) if i["id"] == image_id), None)
    if image is None:
        raise HTTPException(404, "No such image")

    setup = [p for p in plausibility.camera_problems(state.get("camera") or {}, [image])
             if p["severity"] == "blocker"]

    gcps = {g["id"]: g for g in state.get("gcps", [])
            if not g.get("isCheckPoint") and all(g.get(k) is not None for k in ("x", "y", "z"))}
    measured = [o for o in state.get("observations", [])
                if o.get("imageId") == image_id and o.get("pointId") in gcps]

    base = {"imageId": image_id, "count": len(measured), "setupProblems": setup}
    if len(measured) < 4:
        return clean({**base, "ready": False,
                      "message": f"{len(measured)} of 4 control points measured on this photo. "
                                 "Residuals appear from the fourth."})
    if setup:
        return clean({**base, "ready": False,
                      "message": "Fix the camera description first; residuals from it "
                                 "would only measure that mistake."})

    fit = FiducialFit.from_dict(image["fiducialFit"]) if image.get("fiducialFit") else None
    try:
        film = np.array([camera.pixel_to_film([[o["col"], o["row"]]], fit)[0] for o in measured])
    except ValueError as exc:
        return clean({**base, "ready": False, "message": str(exc)})
    ground = np.array([[gcps[o["pointId"]][k] for k in ("x", "y", "z")] for o in measured],
                      dtype=float)

    try:
        result = resect(film, ground, camera.focal_mm, image_scale=camera.image_scale)
    except Exception as exc:  # noqa: BLE001
        return clean({**base, "ready": False, "message": f"Could not solve: {exc}"})

    if camera.apply_atmospheric or camera.apply_earth_curvature:
        film = corrections.to_flat(film, camera.focal_mm, result.eo[2], float(np.mean(ground[:, 2])),
                                   bool(camera.apply_atmospheric), bool(camera.apply_earth_curvature),
                                   camera.earth_radius_m or corrections.EARTH_RADIUS_M)
        try:
            result = resect(film, ground, camera.focal_mm, initial=result.eo)
        except Exception as exc:  # noqa: BLE001
            return clean({**base, "ready": False, "message": f"Could not solve: {exc}"})

    pixel_mm = camera.pixel_scale_mm(fit)
    verdict = plausibility.judge_solution(result.eo, ground, result.rms_mm, pixel_mm,
                                          camera.focal_mm, image.get("name", "this photo"))
    points = []
    for index, o in enumerate(measured):
        dx, dy = result.residuals_mm[index]
        scale = (result.eo[2] - ground[index][2]) / camera.focal_mm
        points.append({
            "pointId": o["pointId"],
            "px": float(math.hypot(dx, dy) / pixel_mm) if pixel_mm else None,
            "groundM": float(math.hypot(dx, dy) * scale),
        })
    # Which point is to blame? Not the one with the biggest residual: a bad
    # point drags the solution towards itself and spreads its error over the
    # others. Nor the one predicted worst when left out: leaving out a corner
    # forces an extrapolation, so corners always predict worst. The honest
    # question is which point, once removed, leaves the rest in agreement.
    # Its miss from that agreement is then the size of its error. This needs
    # redundancy after removing a point, so five or more.
    worst = None
    note = ""
    if len(measured) >= 5:
        for index, point in enumerate(points):
            keep = [k for k in range(len(measured)) if k != index]
            try:
                alone = resect(film[keep], ground[keep], camera.focal_mm)
                predicted = project_points(ground[[index]], alone.eo, camera.focal_mm)[0]
                miss_mm = float(np.hypot(*(predicted - film[index])))
                point["withoutItM"] = miss_mm * (alone.eo[2] - ground[index][2]) / camera.focal_mm
                point["restPx"] = alone.rms_mm / pixel_mm if pixel_mm else None
            except Exception:  # noqa: BLE001
                point["withoutItM"] = None
                point["restPx"] = None
        scored = sorted((p for p in points if p.get("restPx") is not None),
                        key=lambda p: p["restPx"])
        if len(scored) >= 2 and verdict["quality"] != "good":
            best, runner_up = scored[0], scored[1]
            if best["restPx"] < 0.5 * runner_up["restPx"]:
                worst = best["pointId"]
    elif verdict["quality"] != "good":
        note = ("With four points one bad point cannot be singled out: each of them "
                "moves the solution. A fifth makes that possible.")

    return clean({
        **base,
        "ready": True,
        "quality": verdict["quality"],
        "problems": verdict["problems"],
        "rmsPx": verdict["rmsPx"],
        "rmsGroundM": verdict["rmsGroundM"],
        "degreesOfFreedom": 2 * len(measured) - 6,
        "points": points,
        "worst": worst,
        "note": note,
    })


@app.post("/model/compute")
def model_compute(payload: dict = Body(default={})) -> dict:
    """Solve the block. Resection for a single photo, bundle for several."""
    project = require_project()
    camera = CameraModel.from_dict(project.state.get("camera") or {})

    # A camera that contradicts its images is refused here, not just flagged
    # in the interface: the adjustment can absorb much of such an error and
    # return residuals that look merely mediocre, which is worse than failing.
    camera_blockers = [p for p in plausibility.camera_problems(
        project.state.get("camera") or {},
        [i for i in project.state.get("images", []) if i.get("online")])
        if p["severity"] == "blocker"]
    if camera_blockers:
        raise HTTPException(400, "The camera description does not match the images, so "
                                 "no solution from it could be trusted. "
                                 + " ".join(p["message"] for p in camera_blockers))
    _refuse_impossible_control_crs(project.state)
    settings = _adjustment_settings(project.state, payload)

    observations, focal_by_image, problems = _film_observations(project)
    if not observations:
        raise HTTPException(400, "No usable image measurements. " + "; ".join(problems))

    control = {
        g["id"]: (float(g["x"]), float(g["y"]), float(g["z"]))
        for g in project.state.get("gcps", [])
        if all(g.get(k) is not None for k in ("x", "y", "z"))
    }
    check_points = {
        g["id"] for g in project.state.get("gcps", []) if g.get("isCheckPoint")
    }

    image_ids = sorted({o.image_id for o in observations})

    def work(progress, should_cancel):
        progress(0.1, "Preparing observations")

        def solve(observations):
            if len(image_ids) == 1 and not project.state.get("tiePoints"):
                image_id = image_ids[0]
                usable = [o for o in observations if o.point_id in control]
                if len(usable) < 3:
                    raise ValueError(
                        f"Single-photo resection needs 3 control points, found {len(usable)}"
                    )
                progress(0.4, "Space resection")
                result = resect(
                    np.array([[o.x_mm, o.y_mm] for o in usable]),
                    np.array([control[o.point_id] for o in usable]),
                    camera.focal_mm,
                    weights=np.array([o.weight for o in usable]),
                    image_scale=camera.image_scale,
                )
                residuals = {
                    f"{image_id}|{o.point_id}": [float(result.residuals_mm[i, 0]),
                                                 float(result.residuals_mm[i, 1])]
                    for i, o in enumerate(usable)
                }
                image_record = project.image(image_id)
                fit = (FiducialFit.from_dict(image_record["fiducialFit"])
                       if image_record.get("fiducialFit") else None)
                verdict = plausibility.judge_solution(
                    result.eo, np.array([control[o.point_id] for o in usable]), result.rms_mm,
                    camera.pixel_scale_mm(fit), camera.focal_mm, image_record.get("name", "the photo"),
                )
                covariance = _resection_covariance(
                    result, np.array([[o.x_mm, o.y_mm] for o in usable]),
                    np.array([control[o.point_id] for o in usable]), camera.focal_mm)
                precision = {}
                if covariance is not None:
                    ground_z = float(np.mean([control[o.point_id][2] for o in usable]))
                    precision[image_id] = _precision_summary(camera, image_record, result.eo,
                                                             covariance, ground_z)
                weak = _precision_problems(precision, {image_id: image_record.get("name", image_id)})
                if weak and verdict["quality"] == "good":
                    verdict["quality"] = "poor"
                verdict["problems"] = verdict["problems"] + weak
                summary = {
                    "method": "resection",
                    "precision": precision,
                    "converged": bool(result.converged) and verdict["ok"],
                    "quality": verdict["quality"],
                    "qualityProblems": verdict["problems"],
                    "rmsPx": verdict["rmsPx"],
                    "iterations": result.iterations,
                    "sigma0": result.sigma0,
                    "rmsImageMm": result.rms_mm,
                    "rmsControlM": None,
                    "rmsCheckM": None,
                    "degreesOfFreedom": max(2 * len(usable) - 6, 1),
                    "message": (verdict["problems"][0] if verdict["problems"] and not verdict["ok"]
                                else result.message),
                    "residuals": residuals,
                    "solvedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "exterior": {image_id: result.eo.tolist()},
                }
            else:
                progress(0.3, f"Bundle adjustment — {len(image_ids)} photos, "
                              f"{len(observations)} observations")
                pixel_mm = 2.0 * _default_image_sigma_mm(project, camera)
                images_state = {i["id"]: i for i in project.state["images"]}
                eo_observed = {}
                if settings["useExteriorObservations"]:
                    eo_observed = {i: images_state[i]["exteriorObserved"] for i in image_ids
                                   if (images_state.get(i) or {}).get("exteriorObserved")}
                angle_sigma = settings.get("eoSigmaAngleDeg")
                data = BundleInput(
                    image_ids=image_ids,
                    focal_by_image=focal_by_image,
                    observations=observations,
                    control=control,
                    check_points=check_points,
                    control_sigma_m=float(settings["controlSigmaXY"]),
                    control_sigma_z_m=(float(settings["controlSigmaZ"])
                                       if settings.get("controlSigmaZ") else None),
                    control_sigma_by_point={
                        g["id"]: (g["sigma"] if isinstance(g["sigma"], (list, tuple))
                                  else float(g["sigma"]))
                        for g in project.state.get("gcps", [])
                        if g.get("sigma") not in (None, "", 0)
                    },
                    image_sigma_mm=(float(payload["imageSigma"]) if payload.get("imageSigma")
                                    else float(settings["imageSigmaPx"]) * pixel_mm),
                    tie_sigma_mm=float(settings["tieSigmaPx"]) * pixel_mm,
                    image_scale=camera.image_scale,
                    eo_observations=eo_observed,
                    eo_sigma_xy_m=float(settings["eoSigmaXY"]) if settings.get("eoSigmaXY") else None,
                    eo_sigma_z_m=float(settings["eoSigmaZ"]) if settings.get("eoSigmaZ") else None,
                    eo_sigma_angle_rad=math.radians(float(angle_sigma)) if angle_sigma else None,
                    gnss_shift=bool(settings["gnssShift"]),
                    self_calibration=settings["selfCalibration"],
                    lens_priors=settings.get("lensPriors") or "metric",
                    ground_z=(project.state.get("droneBlock") or {}).get("groundZ"),
                    robust=bool(settings["robust"]),
                    auto_reject_ties=bool(settings["autoRejectTies"]),
                    automatic_ties={t["id"] for t in project.state.get("tiePoints", [])
                                    if t.get("source") in tiepoints.MATCH_METHODS},
                )
                result = adjust_block(data)
                images_by_id = {i["id"]: i for i in project.state["images"]}
                control_xyz = np.array(list(control.values())) if control else np.zeros((0, 3))
                verdicts = []
                for image_id, eo in result.eo.items():
                    record = images_by_id.get(image_id, {})
                    fit = FiducialFit.from_dict(record["fiducialFit"]) if record.get("fiducialFit") else None
                    verdicts.append(plausibility.judge_solution(
                        eo, control_xyz if len(control_xyz) else np.array([[0, 0, 0]]),
                        result.per_image_rms_mm.get(image_id, result.rms_image_mm),
                        camera.pixel_scale_mm(fit), camera.focal_mm, record.get("name", image_id),
                    ))
                if len(control_xyz):
                    mean_z = float(np.mean(control_xyz[:, 2]))
                elif result.object_points:
                    # No control (a GNSS-only drone block): the ground is where
                    # its tie points are.
                    mean_z = float(np.median([p[2] for p in result.object_points.values()]))
                else:
                    mean_z = 0.0
                precision = {
                    image_id: _precision_summary(camera, images_by_id.get(image_id, {}), eo,
                                                 result.eo_covariance.get(image_id), mean_z)
                    for image_id, eo in result.eo.items()
                    if result.eo_covariance.get(image_id) is not None
                }
                block_ok = all(v["ok"] for v in verdicts)
                block_problems = [p for v in verdicts for p in v["problems"]]
                # Held by GNSS alone, every photo's corners carry the GPS's own
                # metres of uncertainty: one statement for the block, not a
                # warning per photo. Only photos beyond that are singled out.
                gnss_only = not control and bool(data.eo_observations) and bool(data.eo_sigma_xy_m)
                block_problems += _precision_problems(
                    precision, {k: v.get("name", k) for k, v in images_by_id.items()},
                    data.eo_sigma_xy_m if gnss_only else data.control_sigma_m)
                summary = {
                    "method": "bundle",
                    "precision": precision,
                    "pointSigma": {k: v for k, v in result.point_sigma.items()},
                    "converged": bool(result.converged) and block_ok,
                    "quality": ("failed" if not block_ok
                                else "poor" if block_problems else "good"),
                    "qualityProblems": block_problems,
                    "iterations": result.iterations,
                    "sigma0": result.sigma0,
                    "rmsImageMm": result.rms_image_mm,
                    "rmsControlM": result.rms_control_m,
                    "rmsCheckM": result.rms_check_m,
                    "degreesOfFreedom": result.degrees_of_freedom,
                    "message": (block_problems[0] if not block_ok else result.message),
                    "residuals": {
                        f"{k[0]}|{k[1]}": [v[0], v[1]] for k, v in result.residuals_mm.items()
                    },
                    "imageSigmaMm": data.image_sigma_mm,
                    "tieSigmaMm": data.tie_sigma_mm,
                    "controlSigmaM": data.control_sigma_m,
                    "controlSigmaZM": data.control_sigma_z_m or data.control_sigma_m,
                    "settings": settings,
                    "perImageRmsMm": result.per_image_rms_mm,
                    "perPointRmsMm": result.per_point_rms_mm,
                    "controlResidualsM": result.control_residuals_m,
                    "rmsControlXyzM": list(result.rms_control_xyz_m),
                    "rmsCheckXyzM": list(result.rms_check_xyz_m),
                    "suspects": result.suspects,
                    "snooping": result.snooping,
                    "meanRedundancy": result.mean_redundancy,
                    "rejectedTies": result.rejected_ties,
                    "singleRayPoints": result.single_ray_points,
                    "horizontalChecks": result.horizontal_checks,
                    "selfCalibration": {
                        "set": data.self_calibration,
                        "radiusMm": result.format_radius_mm,
                        "values": {k: v[0] for k, v in result.additional_parameters.items()},
                        "sigmas": {k: v[1] for k, v in result.additional_parameters.items()},
                    } if result.additional_parameters else None,
                    "gnssShiftM": result.gnss_shift_m,
                    "exteriorObservationResiduals": result.eo_observation_residuals,
                    "objectPoints": {k: v.tolist() for k, v in result.object_points.items()},
                    "solvedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "exterior": {k: v.tolist() for k, v in result.eo.items()},
                }
                summary["accuracy"] = reports.accuracy_statement(
                    result.control_residuals_m, check_points)
                notes = []
                if result.rejected_ties:
                    test = "data-snooping test" if result.snooping else "outlier screen"
                    notes.append(
                        f"{len(result.rejected_ties)} automatic tie points failed the "
                        f"{test} and were left out.")
                if gnss_only and precision:
                    corner = float(np.median([p["cornerM"] for p in precision.values()
                                              if p and p.get("cornerM")] or [0.0]))
                    notes.append(
                        "Placed by GPS alone: the photos fit one another closely, but the "
                        f"block as a whole is only as well placed as the drone's GPS, about "
                        f"{corner:.1f} m at the photo corners (median, 1 sigma). Ground "
                        "control would tie it down.")
                if notes:
                    summary["notes"] = notes

            return summary

        summary = solve(observations)

        # Refraction and earth curvature depend on each photo's flying
        # height, which only a first solution gives. So: solve, correct every
        # measurement for its own photo's height, and solve again. The change
        # in height between the passes is far too small to matter.
        refraction = bool(camera.apply_atmospheric)
        curvature = bool(camera.apply_earth_curvature)
        if (refraction or curvature) and summary.get("exterior"):
            progress(0.6, "Correcting for refraction and earth curvature")
            ground_z = _mean_control_height(project.state)
            # Each point at its own height from the first solution: both
            # effects depend on the height of the ground below the camera.
            heights = {pid: float(xyz[2]) for pid, xyz in (summary.get("objectPoints") or {}).items()}
            heights.update({pid: float(xyz[2]) for pid, xyz in control.items()})
            radius = camera.earth_radius_m or corrections.EARTH_RADIUS_M
            corrected = []
            for o in observations:
                eo = summary["exterior"].get(o.image_id)
                if eo is None:
                    corrected.append(o)
                    continue
                flat = corrections.to_flat([[o.x_mm, o.y_mm]], camera.focal_mm, eo[2],
                                           heights.get(o.point_id, ground_z),
                                           refraction, curvature, radius)[0]
                corrected.append(Observation(o.image_id, o.point_id, float(flat[0]),
                                             float(flat[1]), o.weight))
            summary = solve(corrected)
        summary["corrections"] = {"refraction": refraction, "curvature": curvature}

        # A drone block has no DEM of its own: its tie points become the
        # surface its orthophotos are made on.
        surface = None
        block = project.state.get("droneBlock")
        if block and not control and summary.get("objectPoints") and summary.get("exterior"):
            progress(0.85, "Building a ground surface from the tie points")
            try:
                surface = _drone_surface(project, summary, block)
            except Exception as exc:  # noqa: BLE001 -- the solution stands without it
                summary.setdefault("notes", []).append(
                    f"No ground surface could be built from the tie points: {exc}")

        progress(0.9, "Storing solution")

        def apply(state: dict) -> None:
            state["model"] = summary
            if surface:
                block = state.setdefault("droneBlock", {})
                previous = (block.get("surface") or {}).get("path")
                block["surface"] = surface
                if block.get("dense"):
                    # Built from the orientations this solve has just replaced.
                    block["dense"]["stale"] = True
                dem = state.setdefault("dem", {})
                # Never over an operator's own DEM; replacing our own is fine.
                if not dem.get("referencePath") or dem.get("referencePath") == previous:
                    dem["referencePath"] = surface["path"]
            for image in state["images"]:
                eo = summary["exterior"].get(image["id"])
                if eo:
                    image["exterior"] = eo
                    image["exteriorSource"] = summary["method"]
            # The camera every product is made with is the one this solution
            # was made with: the certificate plus this self-calibration, or
            # the certificate alone.
            calibration = summary.get("selfCalibration")
            camera_state = dict(state.get("camera") or {})
            camera_state["adjustment"] = (
                {key: calibration[key] for key in ("set", "radiusMm", "values", "sigmas")}
                if calibration and calibration.get("values") else None)
            state["camera"] = camera_state

        project.mutate("model_compute", apply)
        broadcast({"type": "project.changed", "operation": "model_compute"})
        return clean({k: v for k, v in summary.items() if k != "residuals"})

    job = jobs.submit("model", "Computing sensor model", work)
    return {"job": job.to_dict()}


def _drone_surface(project, summary: dict, block: dict) -> dict:
    """Grid a solved drone block's tie points into the surface its orthos use."""
    from fiducia import drone

    state = project.state
    camera = state.get("camera") or {}
    points = np.array(list(summary["objectPoints"].values()), dtype=float)
    centres = np.array([eo[:2] for eo in summary["exterior"].values()], dtype=float)
    gsd = float(block.get("gsdM") or 0.02)
    footprint = max(int(camera.get("columns") or 0), int(camera.get("rows") or 0)) * gsd
    reach = 0.75 * footprint   # half a footprint, and room for tilt
    bounds = (centres[:, 0].min() - reach, centres[:, 1].min() - reach,
              centres[:, 0].max() + reach, centres[:, 1].max() + reach)
    cell = max(0.5, round(footprint / 60.0, 1))
    path = str(Path(project.outputs_dir) / "drone_surface.tif")
    raster.close_raster(path)   # may be open in the viewer
    crs = (state.get("projection") or {}).get("output") or block.get("crs")
    info = drone.surface_from_points(points, path, geodesy.resolve_crs(crs).to_wkt(), bounds, cell)
    info["builtAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return info


@app.post("/model/checkpoint")
def model_checkpoint(payload: dict = Body(...)) -> dict:
    """Toggle a GCP between control and check point."""
    project = require_project()

    def apply(state: dict) -> dict:
        for gcp in state["gcps"]:
            if gcp["id"] == payload["pointId"]:
                gcp["isCheckPoint"] = bool(payload.get("isCheckPoint", not gcp.get("isCheckPoint")))
                return gcp
        raise HTTPException(404, f"No point {payload['pointId']}")

    gcp = project.mutate("model_checkpoint", apply)
    return clean({"gcp": gcp, "state": project.state})


@app.get("/model/residuals")
def model_residuals(units: str = Query("ground"), show: str = Query("all")) -> dict:
    project = require_project()
    return clean({"rows": reports.residual_table(project.state, units, show)})


# -- ortho -----------------------------------------------------------------


@app.post("/ortho/generate")
def ortho_generate(payload: dict = Body(...)) -> dict:
    project = require_project()
    state = project.state
    _refuse_impossible_control_crs(state)
    camera_dict = state.get("camera") or {}
    projection = state.get("projection") or {}

    image_ids = payload.get("imageIds") or [i["id"] for i in project.images_online()]
    dem_path = payload.get("demPath") or (state.get("dem") or {}).get("referencePath")
    output_dir = Path(payload.get("outputDir") or project.outputs_dir)

    from fiducia import export

    out_format, out_compress = export.normalise(payload.get("format"), payload.get("compression"))
    extension = export.extension_for(out_format)

    # Hidden ground is left out by default when the surface has trees and
    # buildings in it -- a drone block's own surface model. A bare-earth DEM
    # hides nothing, and testing it would only cost time.
    drone_surfaces = {(state.get("droneBlock") or {}).get(k, {}).get("path")
                      for k in ("dense", "surface")} - {None}
    occlusion = bool(payload.get("occlusion", bool(dem_path) and dem_path in drone_surfaces))

    specs = []
    for image_id in image_ids:
        image = project.image(image_id)
        if not image.get("exterior"):
            raise HTTPException(400, f"{image['name']} has no solved exterior orientation")

        output_path = payload.get("outputPaths", {}).get(image_id) or str(
            output_dir / f"o{image['name']}{extension}"
        )
        specs.append(
            ortho.OrthoSpec(
                image_path=image["path"],
                output_path=output_path,
                camera=camera_dict,
                fiducial_fit=image.get("fiducialFit"),
                exterior=image["exterior"],
                output_crs=projection.get("output"),
                control_crs=projection.get("gcpSource") or projection.get("output"),
                refraction=bool((state.get("camera") or {}).get("applyAtmospheric")),
                curvature=bool((state.get("camera") or {}).get("applyEarthCurvature")),
                earth_radius_m=float((state.get("camera") or {}).get("earthRadiusM") or corrections.EARTH_RADIUS_M),
                pixel_size_x=float(payload.get("pixelSizeX") or projection.get("pixelSpacingX") or 0.5),
                pixel_size_y=float(payload.get("pixelSizeY") or projection.get("pixelSpacingY") or 0.5),
                dem_path=dem_path,
                # Where the DEM runs out (or there is none): the mean height of
                # the control, never sea level by default.
                background_elevation=payload.get("backgroundElevation")
                if payload.get("backgroundElevation") is not None
                else _mean_control_height(state),
                elevation_scale=float(payload.get("elevationScale", 1.0)),
                elevation_offset=float(payload.get("elevationOffset", 0.0)),
                resampling=payload.get("resampling", "bilinear"),
                clip_region=image.get("clipRegion"),
                nodata=float(payload.get("nodata", 0.0)),
                reserve_nodata=bool(payload.get("reserveNodata", True)),
                output_format=out_format,
                compress=out_compress or "DEFLATE",
                # Whoever generates the orthophoto; the project's author if unknown.
                author=(payload.get("author") or state.get("author") or "").strip(),
                project_name=state.get("name") or "",
                jpeg_quality=int(payload.get("jpegQuality", 90)),
                max_workers=int(payload.get("maxWorkers", 0)),
                occlusion=occlusion,
            )
        )

    def work(progress, should_cancel):
        produced = []
        for index, spec in enumerate(specs):
            if should_cancel():
                raise InterruptedError("Cancelled")
            name = Path(spec.image_path).stem

            def sub(fraction, message, index=index, name=name):
                progress((index + fraction) / len(specs), f"{name} — {message}")

            raster.close_raster(spec.output_path)   # may be open in the viewer
            result = ortho.generate_ortho(spec, sub, should_cancel)
            produced.append(result.to_dict())

        def apply(state: dict) -> None:
            for entry, spec, image_id in zip(produced, specs, image_ids):
                record = {
                    "id": f"ortho_{image_id}",
                    "imageId": image_id,
                    "name": Path(entry["outputPath"]).stem,
                    "path": entry["outputPath"],
                    "width": entry["width"],
                    "height": entry["height"],
                    "bounds": entry["bounds"],
                    "pixelSize": entry["pixelSize"],
                    "validFraction": entry["validFraction"],
                    "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                state["orthos"] = [o for o in state["orthos"] if o["id"] != record["id"]]
                state["orthos"].append(record)

        project.mutate("ortho_generate", apply)
        broadcast({"type": "project.changed", "operation": "ortho_generate"})
        return {"orthos": produced}

    label = f"Orthorectifying {len(specs)} image" + ("s" if len(specs) != 1 else "")
    job = jobs.submit("ortho", label, work, meta={"imageIds": image_ids})
    return {"job": job.to_dict()}


@app.post("/images/transfer")
def images_transfer(payload: dict = Body(...)) -> dict:
    """The pixel of another photo that sees the same ground as this one.

    Used to line two views up on the same place, e.g. the two photos of a
    stereo pair. The height comes from the reference DEM where there is one,
    otherwise from the control points' mean height.
    """
    project = require_project()
    state = project.state
    source = project.image(payload["fromImageId"])
    target = project.image(payload["toImageId"])
    for image in (source, target):
        if not image.get("exterior"):
            raise HTTPException(409, f"{image.get('name', 'A photo')} has no solved orientation yet")

    heights = [float(g["z"]) for g in state.get("gcps", [])
               if isinstance(g.get("z"), (int, float))]
    fallback = sum(heights) / len(heights) if heights else 0.0
    source_of_height = "control mean height" if heights else "sea level"

    height_at = None
    dem_path = (state.get("dem") or {}).get("referencePath")
    if dem_path and Path(dem_path).exists():
        projection = state.get("projection") or {}
        control_crs = projection.get("gcpSource") or projection.get("output")
        to_dem = ortho._converter(control_crs, ortho._dem_crs(dem_path))

        def height_at(xs, ys):
            if to_dem:
                xs, ys = to_dem(xs, ys)
            return raster.sample_elevation(dem_path, xs, ys)

    try:
        result = ortho.transfer_pixel(
            float(payload["col"]), float(payload["row"]), source, target,
            CameraModel.from_dict(state.get("camera") or {}),
            height_at=height_at, fallback_height=fallback,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    width, height = int(target.get("width") or 0), int(target.get("height") or 0)
    result["inside"] = bool(result["ahead"] and 0 <= result["col"] < width
                            and 0 <= result["row"] < height)
    result["heightSource"] = "reference DEM" if result["fromTerrain"] else source_of_height
    return clean(result)


@app.post("/ortho/footprint")
def ortho_footprint(payload: dict = Body(...)) -> dict:
    """Preview the ground footprint and output size before committing."""
    project = require_project()
    image = project.image(payload["imageId"])
    projection = project.state.get("projection") or {}

    if not image.get("exterior"):
        raise HTTPException(400, "Exterior orientation is not solved for this image")

    pixel_x = float(payload.get("pixelSizeX") or projection.get("pixelSpacingX") or 0.5)
    pixel_y = float(payload.get("pixelSizeY") or projection.get("pixelSpacingY") or 0.5)

    spec = ortho.OrthoSpec(
        image_path=image["path"],
        output_path="",
        camera=project.state.get("camera") or {},
        fiducial_fit=image.get("fiducialFit"),
        exterior=image["exterior"],
        output_crs=projection.get("output"),
        pixel_size_x=pixel_x,
        pixel_size_y=pixel_y,
        dem_path=payload.get("demPath") or (project.state.get("dem") or {}).get("referencePath"),
        clip_region=image.get("clipRegion"),
    )

    try:
        bounds = ortho.compute_footprint(spec)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    west, south, east, north = bounds
    width = int(math.ceil((east - west) / pixel_x))
    height = int(math.ceil((north - south) / pixel_y))

    return clean({
        "bounds": list(bounds),
        "width": width,
        "height": height,
        "megapixels": width * height / 1e6,
        "estimatedBytes": width * height * max(image.get("bandCount", 1), 1),
    })


# -- mosaic ----------------------------------------------------------------


@app.post("/mosaic/preview")
def mosaic_preview_endpoint(payload: dict = Body(...)) -> dict:
    project = require_project()
    inputs = payload.get("inputs") or [o["path"] for o in project.state.get("orthos", [])]
    if len(inputs) < 1:
        raise HTTPException(400, "No orthoimages available to mosaic")

    spec = mosaic.MosaicSpec(
        inputs=inputs,
        output_path="",
        color_balance=payload.get("colorBalance", "linear"),
        cutline_method=payload.get("cutlineMethod", "distance"),
        normalization=payload.get("normalization", "none"),
        blend_width_px=float(payload.get("blendWidth", 40.0)),
        starting_image=payload.get("startingImage"),
        sort_method=payload.get("sortMethod", "nearest_center"),
        resampling=payload.get("resampling", "nearest"),
        bounds=payload.get("bounds"),
    )
    try:
        return clean(mosaic.mosaic_preview(spec))
    except Exception as exc:
        raise HTTPException(400, f"Preview failed: {exc}") from exc


@app.post("/mosaic/generate")
def mosaic_generate(payload: dict = Body(...)) -> dict:
    project = require_project()
    inputs = payload.get("inputs") or [o["path"] for o in project.state.get("orthos", [])]
    if not inputs:
        raise HTTPException(400, "No orthoimages available to mosaic")

    output_path = payload.get("outputPath") or str(
        project.outputs_dir / f"{project.state.get('name', 'mosaic')}_mosaic.tif"
    )

    spec = mosaic.MosaicSpec(
        inputs=inputs,
        output_path=output_path,
        color_balance=payload.get("colorBalance", "linear"),
        cutline_method=payload.get("cutlineMethod", "distance"),
        normalization=payload.get("normalization", "none"),
        blend_width_px=float(payload.get("blendWidth", 40.0)),
        starting_image=payload.get("startingImage"),
        sort_method=payload.get("sortMethod", "nearest_center"),
        resampling=payload.get("resampling", "nearest"),
        bounds=payload.get("bounds"),
        pixel_size=payload.get("pixelSize"),
    )

    def work(progress, should_cancel):
        # The viewer may be showing the previous mosaic from this file.
        raster.close_raster(spec.output_path)
        result = mosaic.generate_mosaic(spec, progress, should_cancel)

        def apply(state: dict) -> None:
            record = {
                "id": f"mosaic_{int(time.time())}",
                "name": Path(result.output_path).stem,
                "path": result.output_path,
                "width": result.width,
                "height": result.height,
                "bounds": list(result.bounds),
                "inputs": result.inputs_used,
                "settings": spec.to_dict(),
                "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            # Regenerating overwrites the file; keep one row per file.
            state["mosaics"] = [m for m in state.get("mosaics", [])
                                if m.get("path") != record["path"]]
            state["mosaics"].append(record)

        project.mutate("mosaic_generate", apply)
        broadcast({"type": "project.changed", "operation": "mosaic_generate"})
        return result.to_dict()

    job = jobs.submit("mosaic", f"Mosaicking {len(inputs)} images", work)
    return {"job": job.to_dict()}


# -- stereo DEM ------------------------------------------------------------


@app.post("/dem/pairs")
def dem_pairs(payload: dict = Body(default={})) -> dict:
    """List candidate stereo pairs with their base-to-height ratios.

    B/H is the single best predictor of how good a stereo DEM will be, so it is
    shown up front rather than discovered after a long extraction.
    """
    project = require_project()
    images = [i for i in project.images_online() if i.get("exterior")]
    if len(images) < 2:
        raise HTTPException(400, "At least two images with solved orientation are needed")

    state = project.state
    mappings = {i["id"]: _ground_mapping(i, state) for i in images}
    ground_z = _mean_control_height(state)

    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            left, right = images[i], images[j]
            left_eo = np.asarray(left["exterior"], dtype=float)
            right_eo = np.asarray(right["exterior"], dtype=float)
            baseline = float(np.linalg.norm(right_eo[:3] - left_eo[:3]))
            # Base to *height above the ground*, not above the datum.
            height = float((left_eo[2] + right_eo[2]) / 2.0) - ground_z
            ratio = baseline / max(height, 1e-6)
            overlap = _footprint_overlap(left, right, mappings[left["id"]], mappings[right["id"]])

            # Photos that share no ground are not a pair at all.
            if overlap is not None and overlap < 0.02:
                continue

            if overlap is not None and overlap < 0.3:
                quality, note = "poor", (f"Only {overlap:.0%} overlap; the DEM would cover a "
                                         "narrow strip. Neighbouring photos make better pairs")
            elif ratio < 0.05:
                quality, note = "poor", "Baseline too short for reliable elevation"
            elif ratio < 0.2:
                quality, note = "fair", "Usable; expect noise in low-texture areas"
            elif ratio <= 1.0:
                quality, note = "good", "Well-conditioned stereo geometry"
            else:
                quality, note = "fair", "Very wide baseline; matching may fail on relief"

            pairs.append({
                "leftId": left["id"], "rightId": right["id"],
                "leftName": left["name"], "rightName": right["name"],
                "baselineM": baseline, "meanHeightM": height,
                "baseHeightRatio": ratio, "overlap": overlap,
                "quality": quality, "note": note,
            })

    # Best first: good geometry, then the most shared ground.
    rank = {"good": 0, "fair": 1, "poor": 2}
    pairs.sort(key=lambda p: (rank[p["quality"]], -(p["overlap"] or 0)))
    return clean({"pairs": pairs})


def _pair_ground(left: dict, right: dict, state: dict):
    """Expected ground heights and output-CRS bounds for a stereo pair.

    Heights come from the reference DEM sampled over the pair's shared ground
    where there is one, else from the control. They centre the matcher's
    search; the bounds keep the DEM to ground the pair can actually see.
    """
    projection = state.get("projection") or {}
    control_crs = projection.get("gcpSource") or projection.get("output")
    output_crs = projection.get("output") or control_crs

    corners = []
    for image in (left, right):
        mapping = _ground_mapping(image, state)
        if mapping is None:
            continue
        w, h = float(image.get("width") or 0), float(image.get("height") or 0)
        xs, ys = mapping[0](np.array([0, w, w, 0, w / 2]), np.array([0, 0, h, h, h / 2]))
        corners.append(np.column_stack([xs, ys]))
    if not corners:
        return None, None, None
    points = np.vstack(corners)
    points = points[np.isfinite(points).all(axis=1)]

    low = high = None
    dem_path = (state.get("dem") or {}).get("referencePath")
    if dem_path and Path(dem_path).exists():
        try:
            west, south = points.min(axis=0)
            east, north = points.max(axis=0)
            gx, gy = np.meshgrid(np.linspace(west, east, 40), np.linspace(south, north, 40))
            gx, gy = gx.ravel(), gy.ravel()
            to_dem = ortho._converter(control_crs, ortho._dem_crs(dem_path))
            if to_dem:
                gx, gy = to_dem(gx, gy)
            z = np.asarray(raster.sample_elevation(dem_path, gx, gy), dtype=float)
            z = z[np.isfinite(z)]
            if z.size >= 20:
                low, high = float(np.percentile(z, 1)), float(np.percentile(z, 99))
        except Exception:  # noqa: BLE001 -- fall back to the control
            low = high = None
    if low is None:
        heights = [float(g["z"]) for g in state.get("gcps", []) if isinstance(g.get("z"), (int, float))]
        if heights:
            low, high = min(heights), max(heights)

    to_output = ortho._converter(control_crs, output_crs)
    if to_output:
        xs, ys = to_output(points[:, 0], points[:, 1])
        points = np.column_stack([xs, ys])
    west, south = points.min(axis=0)
    east, north = points.max(axis=0)
    pad = 0.05 * max(east - west, north - south)
    return low, high, (float(west - pad), float(south - pad), float(east + pad), float(north + pad))


@app.post("/dem/extract")
def dem_extract(payload: dict = Body(...)) -> dict:
    project = require_project()
    state = project.state
    _refuse_impossible_control_crs(state)
    work_dir = Path(payload.get("workDir") or (project.directory / "outputs" / "demwork"))

    requested = payload.get("pairs") or []
    if not requested:
        raise HTTPException(400, "No stereo pairs selected")

    options_payload = payload.get("options") or {}

    def work(progress, should_cancel):
        produced = []
        for index, entry in enumerate(requested):
            if should_cancel():
                raise InterruptedError("Cancelled")

            left = project.image(entry["leftId"])
            right = project.image(entry["rightId"])

            def sub(fraction, message, index=index):
                progress((index + fraction * 0.5) / len(requested), message)

            low_z, high_z, bounds = _pair_ground(left, right, state)
            detail = options_payload.get("detail", "high")
            pair = stereo_dem.generate_epipolar_pair(
                left, right, state.get("camera") or {}, str(work_dir),
                max_px=stereo_dem.DETAIL_MAX_PX.get(detail, 4000), progress=sub,
                ground_height_m=(0.5 * (low_z + high_z) if low_z is not None and high_z is not None
                                 else _mean_control_height(state)),
            )

            options = stereo_dem.DemOptions(
                method=options_payload.get("method", "sgm"),
                detail=options_payload.get("detail", "high"),
                terrain=options_payload.get("terrain", "rolling"),
                smoothing=options_payload.get("smoothing", "medium"),
                apply_wallis=bool(options_payload.get("applyWallis", False)),
                min_elevation=options_payload.get("minElevation"),
                max_elevation=options_payload.get("maxElevation"),
                expected_min_z=low_z,
                expected_max_z=high_z,
                bounds=bounds,
                pixel_sampling=int(options_payload.get("pixelSampling", 4)),
                output_resolution=options_payload.get("outputResolution"),
                output_crs=(state.get("projection") or {}).get("output"),
                control_crs=(state.get("projection") or {}).get("gcpSource")
                or (state.get("projection") or {}).get("output") or "",
                output_path=str(
                    work_dir / f"dem_{left['name']}_{right['name']}.tif"
                ),
            )

            def sub2(fraction, message, index=index):
                progress((index + 0.5 + fraction * 0.5) / len(requested), message)

            raster.close_raster(options.output_path)   # may be open in the viewer
            result = stereo_dem.extract_dem(pair, options, sub2, should_cancel)
            result["pair"] = pair.to_dict()
            produced.append(result)

        # One geocoded surface from all the pairs, as the deliverable is
        # usually wanted; the per-pair DEMs are kept for inspection. Each pair
        # is levelled onto its neighbours first, so the joins carry no step,
        # and blunders are outvoted where three or more overlap.
        merged = None
        merged_path = str(project.outputs_dir / "stereo_dem_merged.tif")
        if len(produced) >= 2 and payload.get("merge", True):
            progress(0.97, f"Merging {len(produced)} DEMs into one surface")
            raster.close_raster(merged_path)
            merged = terrain.merge(
                [entry["outputPath"] for entry in produced], merged_path,
                "feather", align=True, trim_cells=3,
            )
            merged["outputPath"] = merged_path

        def apply(state: dict) -> None:
            # A re-run overwrites the same files; keep one row per file.
            rewritten = {entry["outputPath"] for entry in produced}
            if merged:
                rewritten.add(merged_path)
            state["stereoDems"] = [d for d in state.get("stereoDems", [])
                                   if d.get("path") not in rewritten]
            for entry in produced:
                state["stereoDems"].append({
                    "path": entry["outputPath"],
                    "leftId": entry["leftId"],
                    "rightId": entry["rightId"],
                    "resolution": entry["resolution"],
                    "coverage": entry["coverage"],
                    "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
            if merged:
                state["stereoDems"].append({
                    "path": merged_path,
                    "merged": True,
                    "inputs": len(produced),
                    "resolution": produced[0].get("resolution"),
                    "coverage": merged.get("coverage", produced[0].get("coverage", 1.0)),
                    "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                state["dem"]["extractedPath"] = state["stereoDems"][-1]["path"]
            elif produced:
                state["dem"]["extractedPath"] = produced[-1]["outputPath"]

        project.mutate("dem_extract", apply)
        broadcast({"type": "project.changed", "operation": "dem_extract"})
        return {"dems": produced, "merged": merged}

    job = jobs.submit("dem", f"Extracting DEM from {len(requested)} pair(s)", work)
    return {"job": job.to_dict()}


# -- LiDAR -----------------------------------------------------------------


@app.post("/lidar/inspect")
def lidar_inspect(payload: dict = Body(...)) -> dict:
    try:
        return clean(lidar.inspect_cloud(payload["path"]).to_dict())
    except Exception as exc:
        raise HTTPException(400, f"Could not read point cloud: {exc}") from exc


@app.post("/lidar/rasterize")
def lidar_rasterize(payload: dict = Body(...)) -> dict:
    project = require_project()
    options = lidar.RasterizeOptions(
        output_path=payload.get("outputPath")
        or str(project.outputs_dir / f"{Path(payload['path']).stem}_dtm.tif"),
        cell_size=float(payload.get("cellSize", 1.0)),
        classes=payload.get("classes", [2]),
        returns=payload.get("returns", "last"),
        cell_assignment=payload.get("cellAssignment", "idw"),
        void_fill=payload.get("voidFill", "natural_neighbor"),
        crs=payload.get("crs") or (project.state.get("projection") or {}).get("output"),
    )
    source = payload["path"]

    def work(progress, should_cancel):
        result = lidar.rasterize(source, options, progress, should_cancel)

        def apply(state: dict) -> None:
            state.setdefault("lidar", []).append({
                "source": source,
                "outputPath": result["outputPath"],
                "cellSize": result["cellSize"],
                "pointsUsed": result["pointsUsed"],
                "classes": result["classes"],
                "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        project.mutate("lidar_rasterize", apply)
        broadcast({"type": "project.changed", "operation": "lidar_rasterize"})
        return result

    job = jobs.submit("lidar", f"Rasterising {Path(source).name}", work)
    return {"job": job.to_dict()}


@app.post("/lidar/classify-ground")
def lidar_classify_ground(payload: dict = Body(...)) -> dict:
    project = require_project()
    source = payload["path"]
    stem = Path(source).stem
    options = lidar.GroundOptions(
        output_path=payload.get("outputPath")
        or str(project.outputs_dir / f"{stem}_ground{Path(source).suffix.lower() or '.laz'}"),
        cell_size=float(payload.get("cellSize", 1.0)),
        slope=float(payload.get("slope", 0.15)),
        window=float(payload.get("window", 18.0)),
        elevation_threshold=float(payload.get("elevationThreshold", 0.5)),
        elevation_scalar=float(payload.get("elevationScalar", 1.25)),
        low_noise=bool(payload.get("lowNoise", True)),
        crs=payload.get("crs") or (project.state.get("projection") or {}).get("output"),
    )

    def work(progress, should_cancel):
        result = lidar.classify_ground(source, options, progress, should_cancel)

        def apply(state: dict) -> None:
            state.setdefault("lidarClouds", []).append({
                "source": source,
                "outputPath": result["outputPath"],
                "groundFraction": result["groundFraction"],
                "lowNoisePoints": result["lowNoisePoints"],
                "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        project.mutate("lidar_classify_ground", apply)
        broadcast({"type": "project.changed", "operation": "lidar_classify_ground"})
        return result

    job = jobs.submit("lidar", f"Finding the ground in {Path(source).name}", work)
    return {"job": job.to_dict()}


@app.post("/lidar/height")
def lidar_height(payload: dict = Body(...)) -> dict:
    project = require_project()
    source = payload["path"]
    method = payload.get("method", "highest")
    max_height = payload.get("maxHeight")
    options = lidar.HeightOptions(
        output_path=payload.get("outputPath")
        or str(project.outputs_dir / f"{Path(source).stem}_height.tif"),
        cell_size=float(payload.get("cellSize", 0.5)),
        returns=payload.get("returns", "first"),
        classes=payload.get("classes") or [],
        method=method,
        dtm_path=payload.get("dtmPath") or None,
        max_height=float(max_height) if max_height not in (None, "") else None,
        crs=payload.get("crs") or (project.state.get("projection") or {}).get("output"),
    )

    def work(progress, should_cancel):
        result = lidar.height_above_ground(source, options, progress, should_cancel)

        def apply(state: dict) -> None:
            state.setdefault("lidar", []).append({
                "kind": "height",
                "source": source,
                "outputPath": result["outputPath"],
                "cellSize": result["cellSize"],
                "pointsUsed": result["pointsUsed"],
                "classes": result["classes"],
                "method": result["method"],
                "maxHeight": result["maxHeight"],
                "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        project.mutate("lidar_height", apply)
        broadcast({"type": "project.changed", "operation": "lidar_height"})
        return result

    job = jobs.submit("lidar", f"Measuring heights in {Path(source).name}", work)
    return {"job": job.to_dict()}


@app.post("/lidar/compare")
def lidar_compare(payload: dict = Body(...)) -> dict:
    try:
        return clean(lidar.compare_to_reference(payload["derived"], payload["reference"]))
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


# -- satellite -------------------------------------------------------------


@app.post("/satellite/rpc")
def satellite_rpc(payload: dict = Body(...)) -> dict:
    model = satellite_rpc.load_rpc(payload["path"])
    if model is None:
        raise HTTPException(404, "No rational polynomial coefficients found for this scene")
    return clean({"rpc": model.to_dict()})


@app.post("/satellite/refine")
def satellite_refine(payload: dict = Body(...)) -> dict:
    model = satellite_rpc.RpcModel.from_dict(payload["rpc"])
    observations = [
        (o["sample"], o["line"], o["lon"], o["lat"], o["height"])
        for o in payload.get("observations", [])
    ]
    try:
        return clean(satellite_rpc.refine_rpc(model, observations, payload.get("order", "auto")).to_dict())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# -- reports ---------------------------------------------------------------


def _classic_save(text: str, payload: dict) -> Response:
    if payload.get("savePath"):
        path = classic_report.write(text, payload["savePath"])
        return JSONResponse({"path": path, "text": text})
    return PlainTextResponse(text)


@app.post("/reports/project")
def reports_project(payload: dict = Body(default={})) -> Response:
    project = require_project()
    if payload.get("layout") == "classic":
        return _classic_save(
            classic_report.project_report(project.state, project.directory.name,
                                     payload.get("imageIds")), payload
        )
    text = reports.project_report(
        project.state, payload.get("include"), payload.get("imageIds")
    )
    if payload.get("savePath"):
        Path(payload["savePath"]).write_text(text, encoding="utf-8")
        return JSONResponse({"path": payload["savePath"], "text": text})
    return PlainTextResponse(text)


@app.post("/reports/residual")
def reports_residual(payload: dict = Body(default={})) -> Response:
    project = require_project()
    if payload.get("layout") == "classic":
        return _classic_save(classic_report.residual_report(project.state), payload)
    text = reports.residual_report(
        project.state,
        payload.get("units", "ground"),
        payload.get("show", "all"),
        payload.get("residualType", "rms"),
    )
    if payload.get("savePath"):
        Path(payload["savePath"]).write_text(text, encoding="utf-8")
        return JSONResponse({"path": payload["savePath"], "text": text})
    return PlainTextResponse(text)


# -- jobs ------------------------------------------------------------------


@app.get("/jobs")
def jobs_list(limit: int = Query(50), activeOnly: bool = Query(False)) -> dict:
    return clean({"jobs": jobs.list(limit, activeOnly)})


@app.post("/jobs/{job_id}/cancel")
def jobs_cancel(job_id: str) -> dict:
    return {"cancelled": jobs.cancel(job_id)}


@app.post("/jobs/{job_id}/dismiss")
def jobs_dismiss(job_id: str) -> dict:
    return {"dismissed": jobs.dismiss(job_id)}


@app.post("/jobs/clear")
def jobs_clear() -> dict:
    return {"cleared": jobs.clear_finished()}


# -- entry point -----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fiducia engine")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--project", help="Open a project bundle on startup")
    args = parser.parse_args()

    if args.project:
        global _current
        _current = Project.open(args.project)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
