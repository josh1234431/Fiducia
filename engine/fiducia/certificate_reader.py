"""Reading a camera calibration certificate.

A calibration certificate is the optical specification of a camera: focal
length, principal point offset, the position of eight fiducial marks, and a
table of radial lens distortion. Every project begins with somebody
transcribing roughly forty numbers out of a scanned PDF by hand, and a single
transposed digit silently poisons the entire block -- it does not announce
itself as a typo, it announces itself three hours later as a residual nobody
can explain.

This is the one place in Fiducia where a language model earns its keep. The
task is reading a badly-scanned technical document with inconsistent layout,
which is exactly what vision models are good at and what regular-expression
parsing is hopeless at.

Two safeguards, because the model is not trusted blindly:

**Every field is a proposal carrying its own evidence.** Each value comes back
with the verbatim text it was read from and a confidence, so the operator
confirms against the source rather than accepting a number on faith. Nothing
is written to the project until they do.

**The distortion table is checked by arithmetic, not by the model.** The
extracted radii and distortions are fitted to the polynomial with the same
least-squares routine the Camera panel uses. If a digit was misread the fit
residual jumps, and that is a mechanical check no amount of model confidence
can talk its way past.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "CertificateReadError",
    "read_certificate",
    "available",
    "configure",
    "PROVIDERS",
    "MODEL",
]

# Three ways to read. Offline uses only what is on the computer. The two
# services take the operator's own key; Fiducia has no account of its own and
# never proxies anything.
PROVIDERS = {
    "offline": {
        "label": "Offline (basic)",
        "package": None,
        "defaultModel": "",
        "envKeys": (),
        "keyUrl": "",
        "needsKey": False,
    },
    "anthropic": {
        "label": "Claude (Anthropic)",
        "package": "anthropic",
        "defaultModel": "claude-opus-5",
        "envKeys": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
        "keyUrl": "https://console.anthropic.com/settings/keys",
        "needsKey": True,
    },
    "openai": {
        "label": "ChatGPT (OpenAI)",
        "package": "openai",
        "defaultModel": "gpt-6-astra",
        "envKeys": ("OPENAI_API_KEY",),
        "keyUrl": "https://platform.openai.com/api-keys",
        "needsKey": True,
    },
}

MODEL = PROVIDERS["anthropic"]["defaultModel"]

# Held in memory only. The desktop shell stores keys encrypted with the
# operating system's credential protection and hands them over each time the
# engine starts, so a key never lands in a project bundle or a plain file.
_config: dict = {"provider": "offline", "keys": {}, "models": {}}


def configure(provider: Optional[str] = None, keys: Optional[dict] = None,
              models: Optional[dict] = None) -> None:
    """Set the reading service, and any keys or model overrides.

    A key of None or an empty string forgets that key. Keys not mentioned are
    left alone.
    """
    if provider is not None:
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown reading service: {provider}")
        _config["provider"] = provider
    for name, key in (keys or {}).items():
        if name not in PROVIDERS:
            continue
        if key and str(key).strip():
            _config["keys"][name] = str(key).strip()
        else:
            _config["keys"].pop(name, None)
    for name, model in (models or {}).items():
        if name not in PROVIDERS:
            continue
        if model and str(model).strip():
            _config["models"][name] = str(model).strip()
        else:
            _config["models"].pop(name, None)


def _key_for(provider: str) -> tuple[Optional[str], str]:
    """The key to use and where it came from: 'saved', 'environment' or ''."""
    if _config["keys"].get(provider):
        return _config["keys"][provider], "saved"
    for name in PROVIDERS[provider]["envKeys"]:
        if os.environ.get(name):
            return os.environ[name], "environment"
    return None, ""


def _model_for(provider: str) -> str:
    return _config["models"].get(provider) or PROVIDERS[provider]["defaultModel"]
MAX_BYTES = 30 * 1024 * 1024        # the API accepts 32 MB; leave memory_budget

SUPPORTED = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class CertificateReadError(RuntimeError):
    """Raised with a message meant to be shown to the operator as-is."""

    for_operator = True


def available() -> dict:
    """Whether certificate reading can run, and if not, exactly why.

    Never returns a key -- only whether one is present, where it came from,
    and its last four characters so the operator can tell which one it is.
    """
    provider = _config["provider"]
    spec = PROVIDERS[provider]

    services = {}
    from . import certificate_offline
    offline = certificate_offline.available()

    for name, other in PROVIDERS.items():
        key, origin = _key_for(name)
        if other["package"] is None:
            installed = offline["installed"]
        else:
            try:
                __import__(other["package"])
                installed = True
            except ImportError:
                installed = False
        services[name] = {
            "label": other["label"],
            "needsKey": other["needsKey"],
            "installed": installed,
            "hasKey": bool(key),
            "keyOrigin": origin,
            "keyHint": key[-4:] if key and len(key) >= 12 else "",
            "model": _model_for(name),
            "defaultModel": other["defaultModel"],
            "keyUrl": other["keyUrl"],
        }

    services["offline"].update(pdf=offline["pdf"], ocr=offline["ocr"], note=offline["reason"])
    info = {"provider": provider, "model": _model_for(provider), "services": services}
    current = services[provider]

    if provider == "offline":
        info.update(available=offline["installed"], reason=offline["reason"]
                    if not offline["installed"] else "")
    elif not current["installed"]:
        info.update(available=False, reason=(
            f"The {spec['package']} package is not installed in the engine's "
            f"Python environment. Run: pip install {spec['package']}"
        ))
    elif not current["hasKey"]:
        info.update(available=False, reason=(
            f"Add a {spec['label']} API key in Settings, or choose "
            "Offline (basic)."
        ))
    else:
        info.update(available=True, reason="")
    return info


# -- the shape we ask for -------------------------------------------------
#
# Every extracted number is wrapped with the text it was read from and a
# confidence, so the interface can show the operator what to check.


def _reading(description: str, unit: str = "mm") -> dict:
    return {
        "type": "object",
        "description": description,
        "properties": {
            "value": {"type": "number", "description": f"The value in {unit}"},
            "sourceText": {
                "type": "string",
                "description": "The exact text this was read from, verbatim, "
                               "so the operator can find it on the page",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "low if the scan is unclear or the value was inferred",
            },
        },
        "required": ["value", "sourceText", "confidence"],
        "additionalProperties": False,
    }


def _nullable(schema: dict) -> dict:
    """A reading that may legitimately be absent from the certificate."""
    return {"anyOf": [schema, {"type": "null"}]}


CERTIFICATE_SCHEMA = {
    "type": "object",
    "properties": {
        "cameraName": {
            "type": "string",
            "description": "Camera make and model, e.g. 'Wild RC30'. Empty if absent.",
        },
        "lensType": {"type": "string", "description": "Lens type, e.g. '15/4 UAG-S'"},
        "serialNumber": {"type": "string", "description": "Lens or camera serial number"},
        "calibrationDate": {"type": "string", "description": "As printed, or empty"},
        "cameraKind": {
            "type": "string",
            "enum": ["film", "digital", "unknown"],
            "description": "film if it has fiducial marks; digital if it quotes a "
                           "pixel pitch or chip size",
        },

        "focalLength": _reading("Calibrated focal length, in millimetres"),

        "principalPointMode": {
            "type": "string",
            "enum": ["ppo", "ppa_pps", "none"],
            "description": "ppo if a single principal point offset is given; "
                           "ppa_pps if the certificate quotes PPA and PPS separately",
        },
        "ppoX": _nullable(_reading("Principal point offset in x")),
        "ppoY": _nullable(_reading("Principal point offset in y")),
        "ppaX": _nullable(_reading("Principal point of autocollimation, x")),
        "ppaY": _nullable(_reading("Principal point of autocollimation, y")),
        "ppsX": _nullable(_reading("Principal point of symmetry, x")),
        "ppsY": _nullable(_reading("Principal point of symmetry, y")),

        "fiducials": {
            "type": "array",
            "description": "Calibrated fiducial mark coordinates in millimetres. "
                           "Empty for a digital camera.",
            "items": {
                "type": "object",
                "properties": {
                    "slot": {
                        "type": "string",
                        "enum": [
                            "top_left", "top_middle", "top_right", "right_middle",
                            "bottom_right", "bottom_middle", "bottom_left", "left_middle",
                        ],
                        "description": "Which position this mark occupies. Work it out "
                                       "from the signs of x and y: +y is top, -y is "
                                       "bottom, +x is right, -x is left, and a "
                                       "coordinate near zero means middle.",
                    },
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "label": {
                        "type": "string",
                        "description": "How the certificate labels this mark, verbatim",
                    },
                },
                "required": ["slot", "x", "y", "label"],
                "additionalProperties": False,
            },
        },

        "distortionTable": {
            "type": "array",
            "description": "Radial distortion measurements. Read the MEAN distortion "
                           "column if several directions are tabulated. Give radius in "
                           "millimetres and distortion in micrometres.",
            "items": {
                "type": "object",
                "properties": {
                    "radiusMm": {"type": "number"},
                    "distortionUm": {"type": "number"},
                },
                "required": ["radiusMm", "distortionUm"],
                "additionalProperties": False,
            },
        },
        "distortionUnitsAsPrinted": {
            "type": "string",
            "description": "The units the certificate actually printed the distortion "
                           "in, e.g. 'micrometres' or 'mm'. State what you converted from.",
        },

        "radialCoefficients": {
            "type": "array",
            "description": "K0..K3 if the certificate states them directly. Usually "
                           "empty -- most certificates give only the table.",
            "items": {"type": "number"},
        },
        "decenteringP1": _nullable(_reading("Decentering coefficient P1", "")),
        "decenteringP2": _nullable(_reading("Decentering coefficient P2", "")),

        "pixelPitchMm": _nullable(_reading("Pixel pitch or chip size, digital only")),
        "columns": _nullable(_reading("Sensor width in pixels", "pixels")),
        "rows": _nullable(_reading("Sensor height in pixels", "pixels")),

        "notes": {
            "type": "string",
            "description": "Anything the operator should check: an unreadable figure, "
                           "an ambiguous sign convention, a value you inferred rather "
                           "than read. Empty if the document was clean.",
        },
        "unreadable": {
            "type": "array",
            "description": "Fields you could not read at all",
            "items": {"type": "string"},
        },
    },
    "required": [
        "cameraName", "lensType", "serialNumber", "calibrationDate", "cameraKind",
        "focalLength", "principalPointMode", "ppoX", "ppoY", "ppaX", "ppaY",
        "ppsX", "ppsY", "fiducials", "distortionTable", "distortionUnitsAsPrinted",
        "radialCoefficients", "decenteringP1", "decenteringP2",
        "pixelPitchMm", "columns", "rows", "notes", "unreadable",
    ],
    "additionalProperties": False,
}


# What the models are actually asked for. CERTIFICATE_SCHEMA above is the
# shape the rest of Fiducia reads, but written out for strict structured
# output its thirteen optional readings compile into a grammar the Anthropic
# API rejects as too large. Here each reading is one entry in a single list,
# named from an enum, and _from_model turns the reply back into that shape.
READING_NAMES = [
    "focalLength", "ppoX", "ppoY", "ppaX", "ppaY", "ppsX", "ppsY",
    "decenteringP1", "decenteringP2", "pixelPitchMm", "columns", "rows",
]

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "cameraName": CERTIFICATE_SCHEMA["properties"]["cameraName"],
        "lensType": CERTIFICATE_SCHEMA["properties"]["lensType"],
        "serialNumber": CERTIFICATE_SCHEMA["properties"]["serialNumber"],
        "calibrationDate": CERTIFICATE_SCHEMA["properties"]["calibrationDate"],
        "cameraKind": CERTIFICATE_SCHEMA["properties"]["cameraKind"],
        "principalPointMode": CERTIFICATE_SCHEMA["properties"]["principalPointMode"],
        "readings": {
            "type": "array",
            "description": (
                "One entry per value found. focalLength, principal point values, "
                "pixelPitchMm in millimetres; columns and rows in pixels; "
                "decentering coefficients as printed. Leave out any the "
                "certificate does not give."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": READING_NAMES},
                    "value": {"type": "number"},
                    "sourceText": {
                        "type": "string",
                        "description": "The exact text this was read from, verbatim",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["name", "value", "sourceText", "confidence"],
                "additionalProperties": False,
            },
        },
        "rotationTable": {
            "type": "array",
            "description": (
                "Only if the certificate tabulates the principal point for each "
                "image rotation (digital cameras delivered turned, e.g. UltraCam "
                "Level 3 at 0, 90, 180, 270 degrees clockwise): one entry per row, "
                "in millimetres. Empty otherwise."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "rotationDeg": {"type": "integer", "enum": [0, 90, 180, 270]},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["rotationDeg", "x", "y"],
                "additionalProperties": False,
            },
        },
        "fiducials": CERTIFICATE_SCHEMA["properties"]["fiducials"],
        "distortionTable": CERTIFICATE_SCHEMA["properties"]["distortionTable"],
        "distortionUnitsAsPrinted": CERTIFICATE_SCHEMA["properties"]["distortionUnitsAsPrinted"],
        "radialCoefficients": CERTIFICATE_SCHEMA["properties"]["radialCoefficients"],
        "notes": CERTIFICATE_SCHEMA["properties"]["notes"],
        "unreadable": CERTIFICATE_SCHEMA["properties"]["unreadable"],
    },
    "required": [
        "cameraName", "lensType", "serialNumber", "calibrationDate", "cameraKind",
        "principalPointMode", "readings", "rotationTable", "fiducials", "distortionTable",
        "distortionUnitsAsPrinted", "radialCoefficients", "notes", "unreadable",
    ],
    "additionalProperties": False,
}


def _from_model(reply: dict) -> dict:
    """A reply in MODEL_SCHEMA's shape, as the CERTIFICATE_SCHEMA shape."""
    if "readings" not in reply:
        return reply
    data = {key: value for key, value in reply.items() if key != "readings"}
    for name in READING_NAMES:
        data[name] = None
    for entry in reply.get("readings") or []:
        name = entry.get("name")
        if name in READING_NAMES and data[name] is None and entry.get("value") is not None:
            data[name] = {
                "value": entry["value"],
                "sourceText": entry.get("sourceText", ""),
                "confidence": entry.get("confidence", "medium"),
            }
    for key in ("fiducials", "distortionTable", "radialCoefficients", "unreadable"):
        data[key] = data.get(key) or []
    for key in ("cameraName", "lensType", "serialNumber", "calibrationDate",
                "distortionUnitsAsPrinted", "notes"):
        data[key] = data.get(key) or ""
    data.setdefault("cameraKind", "unknown")
    data.setdefault("principalPointMode", "none")
    return data


INSTRUCTIONS = """\
You are reading an aerial camera calibration certificate for a photogrammetric \
workstation. Extract the calibration parameters exactly as printed.

Rules that matter:

- Transcribe digits exactly. Do not round, tidy, or "correct" a value that looks \
odd -- an unusual number is usually real, and a silently corrected one is a \
blunder nobody can trace later.
- Preserve signs. Principal point offsets and fiducial coordinates are signed, \
and a dropped minus mirrors the whole photograph.
- Radial distortion is tabulated against radial distance. If the table gives \
distortion for several diagonals or directions, read the MEAN column. Convert \
distortion to micrometres and radial distance to millimetres, and say in \
distortionUnitsAsPrinted what the page actually used.
- Assign each fiducial to a position from the signs of its coordinates, not from \
the order it appears in the table.
- If the principal point is tabulated per delivered image rotation, copy every \
row into rotationTable. Use the calibrated values, never illustrative example \
figures.
- If a figure is illegible, list it in `unreadable` rather than guessing. A \
missing value the operator types in is harmless; an invented one is not.
- Put the verbatim source text in every sourceText field. The operator checks \
your reading against the page, so it must be findable.
- Set confidence to low whenever the scan is poor, the figure is ambiguous, or \
you inferred rather than read it.
"""


def _encode(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    media_type = SUPPORTED.get(suffix)
    if media_type is None:
        guessed, _ = mimetypes.guess_type(str(path))
        media_type = guessed if guessed in SUPPORTED.values() else None
    if media_type is None:
        raise CertificateReadError(
            f"{path.name} is not a format that can be read. "
            "Supply the certificate as a PDF, PNG or JPEG."
        )

    size = path.stat().st_size
    if size > MAX_BYTES:
        raise CertificateReadError(
            f"{path.name} is {size / 1e6:.0f} MB, over the {MAX_BYTES / 1e6:.0f} MB "
            "limit. Export a smaller PDF, or send only the pages carrying the "
            "calibration figures."
        )

    return base64.standard_b64encode(path.read_bytes()).decode("ascii"), media_type


def _verify_distortion(table: list[dict]) -> dict:
    """Fit the extracted table and report how well the polynomial reproduces it.

    This is the check the model cannot argue with. A correctly transcribed
    certificate fits to a fraction of a micrometre; a misread digit does not.
    """
    from .camera import fit_radial_from_table

    if len(table) < 2:
        return {
            "fitted": False,
            "reason": "Fewer than two distortion measurements were found.",
        }

    radii = [row["radiusMm"] for row in table]
    distortions = [row["distortionUm"] for row in table]

    try:
        fit = fit_radial_from_table(radii, distortions)
    except ValueError as exc:
        return {"fitted": False, "reason": str(exc)}

    rms = fit["rmsUm"]
    worst = max(fit["samples"], key=lambda s: abs(s["residualUm"])) if fit["samples"] else None

    if rms < 0.5:
        verdict, note = "good", (
            f"The distortion table fits the polynomial to {rms:.3f} um. "
            "That is consistent with a clean transcription."
        )
    elif rms < 2.0:
        verdict, note = "check", (
            f"The distortion table fits to {rms:.2f} um, which is looser than a "
            "clean certificate usually manages. Check the tabulated figures."
        )
    else:
        verdict, note = "suspect", (
            f"The distortion table only fits to {rms:.1f} um. One of the values "
            "has very likely been misread"
            + (f" -- the worst is {worst['observedUm']:.1f} um at "
               f"r = {worst['radiusMm']:.0f} mm." if worst else ".")
        )

    return {
        "fitted": True,
        "verdict": verdict,
        "note": note,
        "rmsUm": rms,
        "maxUm": fit["maxUm"],
        "k0": fit["k0"], "k1": fit["k1"], "k2": fit["k2"], "k3": fit["k3"],
        "termsFitted": fit["termsFitted"],
        "samples": fit["samples"],
    }


_OPPOSITE = [("top_left", "bottom_right"), ("top_right", "bottom_left"),
             ("top_middle", "bottom_middle"), ("left_middle", "right_middle")]


def _check_fiducials(marks: dict) -> list[str]:
    """Opposite fiducials mirror each other through the centre.

    On a real camera each pair sums to within a fraction of a millimetre of
    zero. A dropped minus sign or a misread digit breaks that by millimetres,
    so this catches the blunder a model or a recogniser is most likely to make.
    """
    notes = []
    for first, second in _OPPOSITE:
        if first not in marks or second not in marks:
            continue
        (x1, y1), (x2, y2) = marks[first], marks[second]
        off = max(abs(x1 + x2), abs(y1 + y2))
        if off > 1.0:
            notes.append(
                f"The {first.replace('_', ' ')} and {second.replace('_', ' ')} "
                f"fiducials do not mirror each other (off by {off:.2f} mm). One of "
                "them has very likely been misread; check the signs and digits."
            )
    return notes


def _to_camera_patch(data: dict, distortion: dict) -> dict:
    """Translate the extraction into the shape the Camera panel already speaks."""
    def value_of(entry) -> Optional[float]:
        if isinstance(entry, dict) and entry.get("value") is not None:
            return float(entry["value"])
        return None

    kind = "digital" if data.get("cameraKind") == "digital" else "film"

    ppo_x = value_of(data.get("ppoX"))
    ppo_y = value_of(data.get("ppoY"))
    if ppo_x is None and ppo_y is None and data.get("principalPointMode") == "ppa_pps":
        # PPO = PPA + PPS, an addition otherwise done by hand for every camera.
        ppa_x, ppa_y = value_of(data.get("ppaX")), value_of(data.get("ppaY"))
        pps_x, pps_y = value_of(data.get("ppsX")), value_of(data.get("ppsY"))
        if ppa_x is not None or pps_x is not None:
            ppo_x = (ppa_x or 0.0) + (pps_x or 0.0)
        if ppa_y is not None or pps_y is not None:
            ppo_y = (ppa_y or 0.0) + (pps_y or 0.0)

    coefficients = data.get("radialCoefficients") or []
    if distortion.get("fitted") and len(coefficients) < 4:
        coefficients = [distortion["k0"], distortion["k1"],
                        distortion["k2"], distortion["k3"]]
    coefficients = list(coefficients) + [0.0] * (4 - len(coefficients))

    patch = {
        "kind": kind,
        "name": " ".join(
            part for part in (data.get("cameraName"), data.get("lensType"),
                              data.get("serialNumber")) if part
        ).strip(),
        "focalMm": value_of(data.get("focalLength")) or 0.0,
        "ppoXMm": ppo_x or 0.0,
        "ppoYMm": ppo_y or 0.0,
        "k0": float(coefficients[0]), "k1": float(coefficients[1]),
        "k2": float(coefficients[2]), "k3": float(coefficients[3]),
        "p1": value_of(data.get("decenteringP1")) or 0.0,
        "p2": value_of(data.get("decenteringP2")) or 0.0,
        "fiducialsMm": {
            entry["slot"]: [float(entry["x"]), float(entry["y"])]
            for entry in (data.get("fiducials") or [])
            if entry.get("slot")
        },
        "ppoRotations": {
            str(int(entry["rotationDeg"])): [float(entry["x"]), float(entry["y"])]
            for entry in (data.get("rotationTable") or [])
            if entry.get("rotationDeg") in (0, 90, 180, 270)
            and entry.get("x") is not None and entry.get("y") is not None
        },
    }

    if kind == "digital":
        patch["pixelPitchMm"] = value_of(data.get("pixelPitchMm")) or 0.0
        columns = value_of(data.get("columns"))
        rows = value_of(data.get("rows"))
        patch["columns"] = int(columns) if columns else 0
        patch["rows"] = int(rows) if rows else 0

    return patch


def _extract_anthropic(key: str, model: str, source: Path, encoded: str,
                       media_type: str) -> tuple[dict, dict]:
    try:
        import anthropic
    except ImportError as exc:
        raise CertificateReadError(
            "Reading with Claude needs the anthropic package. Run: pip install anthropic"
        ) from exc

    block_type = "document" if media_type == "application/pdf" else "image"
    client = anthropic.Anthropic(api_key=key)

    def ask(strict: bool):
        # Strict: the reply is constrained to MODEL_SCHEMA. Loose: the schema
        # goes in the request as text, for when the API will not compile it.
        request = REQUEST if strict else (
            REQUEST + " Reply with only a JSON object, no other text, matching "
            "this JSON Schema:\n" + json.dumps(MODEL_SCHEMA)
        )
        config = {"effort": "high"}
        if strict:
            config["format"] = {"type": "json_schema", "schema": MODEL_SCHEMA}
        return client.messages.create(
            model=model,
            max_tokens=16000,
            system=INSTRUCTIONS,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": block_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    },
                    {"type": "text", "text": request},
                ],
            }],
            output_config=config,
        )

    try:
        try:
            response = ask(strict=True)
        except anthropic.BadRequestError as exc:
            if "grammar" not in str(exc).lower() and "schema" not in str(exc).lower():
                raise
            response = ask(strict=False)
    except anthropic.AuthenticationError as exc:
        raise CertificateReadError(
            "Anthropic rejected the API key. Check it was copied in full, or "
            "replace it in Settings, under Certificate reading."
        ) from exc
    except anthropic.PermissionDeniedError as exc:
        raise CertificateReadError(
            f"This Anthropic key does not have access to {model}: {exc.message}"
        ) from exc
    except anthropic.NotFoundError as exc:
        raise CertificateReadError(
            f"Anthropic does not recognise the model {model}."
            + _suggest_models(client, model)
            + " Change it in Settings, under Certificate reading, or clear the "
            "field to use the default."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise CertificateReadError(
            "Rate limited by the Anthropic API. Wait a moment and try again."
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise CertificateReadError(OFFLINE) from exc
    except anthropic.APIStatusError as exc:
        raise CertificateReadError(
            f"The Anthropic API returned {exc.status_code}: {exc.message}"
        ) from exc

    if response.stop_reason == "refusal":
        raise CertificateReadError(REFUSED)
    if response.stop_reason == "max_tokens":
        raise CertificateReadError(TRUNCATED)

    text = next((b.text for b in response.content if b.type == "text"), None)
    usage = {
        "inputTokens": response.usage.input_tokens,
        "outputTokens": response.usage.output_tokens,
    }
    return _parse(text), usage


def _extract_openai(key: str, model: str, source: Path, encoded: str,
                    media_type: str) -> tuple[dict, dict]:
    try:
        import openai
    except ImportError as exc:
        raise CertificateReadError(
            "Reading with ChatGPT needs the openai package. Run: pip install openai"
        ) from exc

    data_url = f"data:{media_type};base64,{encoded}"
    if media_type == "application/pdf":
        document = {"type": "input_file", "filename": source.name,
                    "file_data": data_url, "detail": "high"}
    else:
        document = {"type": "input_image", "image_url": data_url, "detail": "high"}

    client = openai.OpenAI(api_key=key)

    try:
        response = client.responses.create(
            model=model,
            instructions=INSTRUCTIONS,
            input=[{
                "role": "user",
                "content": [document, {"type": "input_text", "text": REQUEST}],
            }],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "calibration_certificate",
                    "schema": MODEL_SCHEMA,
                    "strict": True,
                },
            },
            # Reasoning models spend output tokens thinking before they answer.
            max_output_tokens=32000,
        )
    except openai.AuthenticationError as exc:
        raise CertificateReadError(
            "OpenAI rejected the API key. Check it was copied in full, or "
            "replace it in Settings, under Certificate reading."
        ) from exc
    except openai.PermissionDeniedError as exc:
        raise CertificateReadError(
            f"This OpenAI key does not have access to {model}: {exc.message}"
        ) from exc
    except openai.NotFoundError as exc:
        raise CertificateReadError(
            f"OpenAI does not recognise the model {model}. Change it in "
            "Settings, under Certificate reading."
        ) from exc
    except openai.RateLimitError as exc:
        raise CertificateReadError(
            "Rate limited by OpenAI, or the account has no credit left. Check "
            "billing on the OpenAI platform and try again."
        ) from exc
    except openai.APIConnectionError as exc:
        raise CertificateReadError(OFFLINE) from exc
    except openai.APIStatusError as exc:
        raise CertificateReadError(
            f"The OpenAI API returned {exc.status_code}: {exc.message}"
        ) from exc

    for item in response.output or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", "") == "refusal":
                raise CertificateReadError(REFUSED)
    if response.status == "incomplete":
        raise CertificateReadError(TRUNCATED)

    usage = {
        "inputTokens": response.usage.input_tokens if response.usage else 0,
        "outputTokens": response.usage.output_tokens if response.usage else 0,
    }
    return _parse(response.output_text), usage


REQUEST = "Extract the calibration parameters from this certificate."

OFFLINE = (
    "Could not reach the reading service. Certificate reading is the one part "
    "of Fiducia that needs a network connection -- everything else works "
    "offline. Enter the camera parameters by hand for now."
)
REFUSED = (
    "The model declined to read this document. If it is genuinely a "
    "calibration certificate, enter the values by hand and report this."
)
TRUNCATED = (
    "The model ran out of room before finishing. Try again, or send only the "
    "pages carrying the calibration figures."
)


class _waiting:
    """Report progress once a second while a single long request runs.

    The request to Claude or ChatGPT takes about a minute and reports
    nothing, and the job queue throttles updates, so a message sent just
    before it could be dropped and the bar sat on its previous value. This
    advances the bar towards 80% and shows the seconds elapsed.
    """

    def __init__(self, progress, label: str):
        import threading
        self.progress, self.label = progress, label
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import math
        import time
        started = time.time()
        while True:
            elapsed = time.time() - started
            self.progress(0.25 + 0.55 * (1 - math.exp(-elapsed / 45)),
                          f"Reading the certificate with {self.label}, {elapsed:.0f} s")
            if self.stop.wait(1.0):
                return

    def __enter__(self):
        if self.progress:
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        return False


def _suggest_models(client, model: str) -> str:
    """Model names this key can use that look like the one typed, as a sentence."""
    import difflib
    try:
        names = [m.id for m in client.models.list(limit=100)]
    except Exception:
        return ""
    close = difflib.get_close_matches(model, names, n=3, cutoff=0.4) or names[:3]
    return f" Available models include {', '.join(close)}." if close else ""


def _parse(text: Optional[str]) -> dict:
    if not text:
        raise CertificateReadError("The model returned no readable result")
    body = text.strip()
    # A loose reply may arrive in a code fence, or with a sentence around it.
    if not body.startswith("{"):
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            body = body[start:end + 1]
    try:
        return _from_model(json.loads(body))
    except json.JSONDecodeError as exc:
        raise CertificateReadError(f"The extraction was not valid JSON: {exc}") from exc


def read_certificate(
    path: str,
    model: Optional[str] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    provider: Optional[str] = None,
) -> dict:
    """Read a calibration certificate and return camera parameters as proposals.

    Nothing is applied to the project here. The caller shows the operator each
    value with the text it came from, and writes only what they accept.
    """
    provider = provider or _config["provider"]
    if provider not in PROVIDERS:
        raise CertificateReadError(f"Unknown reading service: {provider}")
    model = model or _model_for(provider)

    # Checked here rather than left to the SDK, whose own complaint about a
    # missing key names HTTP headers instead of telling anyone what to do.
    key, _ = _key_for(provider)
    if PROVIDERS[provider]["needsKey"] and not key:
        raise CertificateReadError(
            f"No {PROVIDERS[provider]['label']} API key has been added. Add one in "
            "Settings, under Certificate reading."
        )

    source = Path(path)
    if not source.exists():
        raise CertificateReadError(f"No file at {path}")

    if provider == "offline":
        from . import certificate_offline
        if source.suffix.lower() not in SUPPORTED:
            raise CertificateReadError(
                f"{source.name} is not a format that can be read. "
                "Supply the certificate as a PDF, PNG or JPEG."
            )
        if progress:
            progress(0.1, f"Reading {source.name} offline")
        try:
            data, usage = certificate_offline.read(source, progress)
        except certificate_offline.OfflineReadError as exc:
            raise CertificateReadError(str(exc)) from exc
        model = ""
    else:
        if progress:
            progress(0.1, f"Encoding {source.name}")

        encoded, media_type = _encode(source)

        extract = _extract_openai if provider == "openai" else _extract_anthropic
        with _waiting(progress, PROVIDERS[provider]["label"]):
            data, usage = extract(key, model, source, encoded, media_type)

    if progress:
        progress(0.85, "Checking the distortion table")

    distortion = _verify_distortion(data.get("distortionTable") or [])
    patch = _to_camera_patch(data, distortion)

    # Collect everything the operator should look at twice.
    review: list[str] = []
    if data.get("notes"):
        review.append(data["notes"])
    for name in data.get("unreadable") or []:
        review.append(f"Could not read: {name}. Enter it by hand.")
    if distortion.get("fitted") and distortion["verdict"] != "good":
        review.append(distortion["note"])

    low_confidence = [
        key for key in (
            "focalLength", "ppoX", "ppoY", "ppaX", "ppaY", "ppsX", "ppsY",
            "pixelPitchMm", "columns", "rows",
        )
        if isinstance(data.get(key), dict) and data[key].get("confidence") == "low"
    ]
    if low_confidence:
        review.append(
            "Read with low confidence: " + ", ".join(low_confidence)
            + ". Check these against the page."
        )

    review += _check_fiducials(patch["fiducialsMm"])

    fiducial_count = len(patch["fiducialsMm"])
    if patch["kind"] == "film" and 0 < fiducial_count < 4:
        review.append(
            f"Only {fiducial_count} fiducial marks were found. Interior "
            "orientation needs at least three, and four or more to show its error."
        )

    if progress:
        progress(1.0, "Certificate read")

    return {
        "extraction": data,
        "camera": patch,
        "distortionCheck": distortion,
        "review": review,
        "provider": provider,
        "model": model,
        "usage": usage,
        "sourcePath": str(source.resolve()),
    }
