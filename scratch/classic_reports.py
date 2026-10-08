"""Parsers for classic project and residual reports.

Deliberately independent of Fiducia's own formatter, so they can check it,
and so real reports can be read back as test data (test_reference_block.py).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from fiducia import classic_report  # noqa: E402

SLOT = {label: slot for slot, label in classic_report.FIDUCIAL_ORDER}
LABEL = {slot: label for slot, label in classic_report.FIDUCIAL_ORDER}


def value_after_colon(line):
    return line.split(" : ", 1)[1] if " : " in line else ""


def parse_project(text: str) -> dict:
    lines = text.split("\r\n")
    doc = {"title": lines[0], "camera": {"radial": [], "decentering": [], "fiducials": []},
           "images": []}
    camera = doc["camera"]
    image = None
    table = None
    for line in lines[1:]:
        stripped = line.strip()
        if line.startswith("Image "):
            m = re.match(r"Image (\S+) GCPs : (\d+)  TPs : (\d+)  Check Pts : (\d+)", line)
            image = {"name": m[1], "gcpCount": int(m[2]), "tpCount": int(m[3]),
                     "checkCount": int(m[4]), "fiducials": [], "gcps": [], "tiePoints": [],
                     "exterior": [], "dem": {}}
            doc["images"].append(image)
            table = None
            continue
        if not stripped:
            table = None if table != "pending" else table
            continue
        key = stripped.split(" : ")[0].strip() if " : " in stripped else None
        value = value_after_colon(stripped)

        if image is None:
            if key == "Filename":
                doc["filename"] = value
            elif key == "Description":
                doc["description"] = value
            elif key == "Focal length":
                camera["focalMm"] = float(value.split()[0])
            elif key == "Principal point":
                parts = value.replace("mm", "").replace(",", " ").split()
                camera["ppoXMm"], camera["ppoYMm"] = float(parts[0]), float(parts[1])
            elif key and key.startswith("Radial Lens Distortion"):
                camera["radial"].append(float(value))
            elif key and key.startswith("Decentering Distortion"):
                camera["decentering"].append(float(value))
            elif key == "Fiducial Type":
                camera["fiducialType"] = value
            elif key in SLOT:
                parts = value.replace("mm", "").replace(",", " ").split()
                camera["fiducials"].append((key, float(parts[0]), float(parts[1])))
            elif key == "Scale":
                camera["scale"] = float(value.split(":")[1])
            elif key == "Earth Radius":
                camera["earthRadiusM"] = None if value == "Not Defined" else float(value)
            continue

        if stripped.startswith("GCP ID Status"):
            table = "gcp"
            continue
        if stripped.startswith("GCP ID") and "Georef" in stripped:
            table = "georef"
            continue
        if stripped.startswith("TP ID"):
            table = "tp"
            continue
        if stripped.startswith("---"):
            continue

        if table == "gcp":
            p = stripped.split()
            image["gcps"].append({"id": p[0], "status": p[1], "z": float(p[2]),
                                  "zSigma": float(p[4]), "col": float(p[5]),
                                  "pixelSigma": float(p[7]), "row": float(p[8])})
            continue
        if table == "georef":
            m = re.match(r"\s*(\S+) (.{16}) +(-?[\d.]+) \+/- ([\d.]+) +(-?[\d.]+) \+/- ([\d.]+)", line)
            gcp = next(g for g in image["gcps"] if g["id"] == m[1])
            gcp.update(georef=m[2], x=float(m[3]), xySigma=float(m[4]), y=float(m[5]))
            continue
        if table == "tp":
            p = stripped.split()
            image["tiePoints"].append({"id": p[0], "col": float(p[1]),
                                       "pixelSigma": float(p[3]), "row": float(p[4])})
            continue

        if key == "Date Added":
            image["dateAdded"] = value
        elif key == "Date Updated":
            image["dateUpdated"] = value
        elif key == "Uncorrected File":
            image["path"] = value
        elif key == "Channels":
            image["channels"] = [int(c) for c in value.split()]
        elif key == "Size":
            p = value.split()
            image["width"], image["height"] = int(p[0]), int(p[3])
        elif key == "Orthorectified File":
            image["ortho"] = {"path": value}
        elif key == "Upper Left":
            p = value.split()
            image["ortho"].update(west=float(p[0]), north=float(p[1]))
        elif key == "Lower Right":
            p = value.split()
            image["ortho"].update(east=float(p[0]), south=float(p[1]))
        elif key == "DEM File":
            image["dem"]["path"] = value
        elif key == "Channel":
            image["dem"]["channel"] = int(value)
        elif key == "Background Elevation":
            image["dem"]["background"] = None if value == "Not Defined" else float(value)
        elif key == "Clip Area Offset":
            a, b = value.strip("()").split(",")
            image["clip"] = [int(a), int(b)]
        elif key == "Clip Area Size":
            p = value.split()
            image["clip"] += [int(p[0]), int(p[3])]
        elif key == "Fiducial Type":
            image["fiducialType"] = value
        elif key in SLOT:
            parts = value.replace("P", "").replace("L", "").replace(",", " ").split()
            image["fiducials"].append((key, float(parts[0]), float(parts[1])))
        elif key == "Calibration edge":
            image["calibrationEdge"] = value
        elif key and key.startswith("Exterior Orientation"):
            image["exterior"].append(float(value))
        elif key == "Status" and value.startswith("Model"):
            image["modelStatus"] = value
    return doc


def parse_residual(text: str) -> dict:
    lines = text.split("\r\n")
    doc = {"units": lines[2].split(": ", 1)[1], "listing": "", "rows": [], "images": []}
    kinds = {"GCP": "gcp", "TP": "tie", "Check": "check"}
    for line in lines:
        if line.startswith("Listing: "):
            doc["listing"] = line[len("Listing: "):]
        elif line.startswith("Residual Summary for ") and not line.endswith(" Images"):
            doc["images"].append(line[len("Residual Summary for "):])
        parts = line.split()
        if len(parts) == 16 and parts[5] in kinds:
            v = [float(x) for x in parts[1:5]] + [float(x) for x in parts[7:]]
            doc["rows"].append({
                "pointId": parts[0], "kind": kinds[parts[5]], "active": True,
                "imageName": parts[6],
                "resXY": v[0], "resX": v[1], "resY": v[2], "resZ": v[3],
                "groundX": v[4], "groundY": v[5], "groundZ": v[6], "compX": v[7], "compY": v[8],
                "dsXY": v[9], "dsX": v[10], "dsY": v[11], "dsZ": v[12],
            })
    return doc
