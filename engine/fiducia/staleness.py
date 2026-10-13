"""Which results are out of date.

Every result records a fingerprint of what it was made from: the solution
of the inputs it solved (the control, the measurements, the camera, the
settings), an orthophoto or stereo DEM of the solution it used, a mosaic of
the orthophotos it joined. A result is out of date when what it was made
from has changed since -- not merely when something was re-run, so solving
the same block twice leaves its orthophotos alone.

Results made before fingerprints were recorded carry none and are never
called out of date: there is no way to tell, and a false alarm is worse
than silence.
"""

from __future__ import annotations

import hashlib
import json

__all__ = ["model_inputs", "solution", "report"]


def _digest(value) -> str:
    text = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _round(values, digits):
    return [round(float(v), digits) for v in values]


def model_inputs(state: dict) -> str:
    """What a solution is made from. Its own outputs (each photo's exterior,
    the self-calibration it writes into the camera) are left out."""
    camera = {k: v for k, v in (state.get("camera") or {}).items() if k != "adjustment"}
    images = [{k: image.get(k) for k in ("id", "fiducialFit", "exteriorObserved", "excluded")}
              for image in state.get("images", [])]
    gcps = [{k: g.get(k) for k in ("id", "x", "y", "z", "isCheckPoint", "sigma")}
            for g in state.get("gcps", [])]
    observations = sorted(
        ([o.get("pointId"), o.get("imageId"), round(float(o.get("col") or 0), 3),
          round(float(o.get("row") or 0), 3), o.get("kind"), o.get("weight")]
         for o in state.get("observations", [])),
        key=lambda o: (str(o[0]), str(o[1])))
    projection = {k: v for k, v in (state.get("projection") or {}).items()
                  if not k.startswith("pixelSpacing")}
    return _digest({
        "camera": camera, "images": images, "gcps": gcps, "observations": observations,
        "mathModel": state.get("mathModel"), "adjustment": state.get("adjustment"),
        "projection": projection,
    })


def solution(model: dict | None) -> str | None:
    """The solution itself, to a millimetre and a microradian: solving the
    same block again gives the same fingerprint."""
    if not model or not model.get("exterior"):
        return None
    exterior = {k: _round(v[:3], 3) + _round(v[3:6], 6) for k, v in model["exterior"].items()}
    calibration = (model.get("selfCalibration") or {}).get("values") or {}
    return _digest({"exterior": exterior,
                    "calibration": {k: round(float(v), 9) for k, v in calibration.items()}})


def report(state: dict) -> dict:
    """What is out of date now, for the project summary."""
    model = state.get("model") or {}
    current = model.get("solutionFingerprint")
    model_stale = bool(model.get("inputsFingerprint")) and model["inputsFingerprint"] != model_inputs(state)
    orthos = [o["id"] for o in state.get("orthos", [])
              if o.get("solution") and current and o["solution"] != current]
    dems = [d.get("path") for d in state.get("stereoDems", [])
            if d.get("solution") and current and d["solution"] != current]
    made = {o["id"]: o.get("generatedAt") for o in state.get("orthos", [])}
    mosaics = [m["id"] for m in state.get("mosaics", [])
               if m.get("orthos") and any(made.get(i) != at for i, at in m["orthos"].items())]
    return {"model": model_stale, "orthos": orthos, "stereoDems": dems, "mosaics": mosaics}
