"""An original aerial landscape for promotional screenshots.

Everything here is generated, so the pictures carry no third-party imagery:
vineyards in rows, ploughed and fallow fields, a river with a riparian strip,
farm roads and a farmstead, lit from the north-west the way a morning survey
flight sees the Western Cape. Three overlapping film frames are then cut from
it, each with fiducial marks and a data strip, and loaded into a Fiducia
project with control and tie points measured on them.

    python scratch/ads/make_scene.py [engine-port]
"""

import json
import math
import os
import sys
from pathlib import Path
from urllib import request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(os.environ["LOCALAPPDATA"]) / "Fiducia" / "adscene"
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
rng = np.random.default_rng(242)

W, H = 3200, 2100          # ground raster, 0.8 m pixels


def noise(shape, scale, octaves=5, seed=0):
    """Fractal value noise, normalised to 0..1."""
    g = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float32)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        cells = (max(2, int(shape[0] / scale * 2 ** octave)),
                 max(2, int(shape[1] / scale * 2 ** octave)))
        base = g.random(cells).astype(np.float32)
        layer = np.asarray(Image.fromarray(base).resize((shape[1], shape[0]), Image.BICUBIC))
        out += layer * amplitude
        total += amplitude
        amplitude *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# -- relief ------------------------------------------------------------------
height = noise((H, W), 900, 6, seed=1) * 60 + (xx / W) * 25
gy, gx = np.gradient(height)
az, alt = math.radians(315), math.radians(38)
shade = np.clip((math.sin(alt) + math.cos(alt) * (-gx * math.sin(az) + gy * math.cos(az)) * 3.0), 0.55, 1.25)

# -- parcels -------------------------------------------------------------------
# Voronoi fields, straightened a little so they read as surveyed boundaries.
seeds = np.column_stack([rng.uniform(0, W, 70), rng.uniform(0, H, 70)])
small = (H // 2, W // 2)
sy, sx = np.mgrid[0:small[0], 0:small[1]] * 2.0
warp = noise(small, 240, 3, seed=5) * 90
distance = np.full(small, np.inf, dtype=np.float32)
label_small = np.zeros(small, dtype=np.int32)
for index, (px, py) in enumerate(seeds):
    d = np.abs(sx - px) * 1.0 + np.abs(sy - py) * 1.15 + warp
    closer = d < distance
    distance[closer] = d[closer]
    label_small[closer] = index
labels = np.asarray(Image.fromarray(label_small.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(np.int32)

# Boundaries become hedgerows and tracks.
edge = np.zeros((H, W), dtype=bool)
edge[1:, :] |= labels[1:, :] != labels[:-1, :]
edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
edge_img = Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
edge = np.asarray(edge_img) > 0

kinds = rng.choice(["vineyard", "vineyard", "vineyard", "ploughed", "fallow", "pasture", "orchard", "stubble"],
                   size=len(seeds))
angles = rng.uniform(0, math.pi, len(seeds))

palette = {
    "vineyard": ((88, 104, 58), (156, 140, 104)),
    "orchard": ((58, 84, 44), (150, 138, 104)),
    "ploughed": ((122, 96, 70), (150, 120, 88)),
    "fallow": ((176, 160, 122), (196, 182, 146)),
    "pasture": ((112, 132, 72), (138, 150, 90)),
    "stubble": ((190, 170, 118), (210, 192, 140)),
}

rgb = np.zeros((H, W, 3), dtype=np.float32)
texture = noise((H, W), 40, 4, seed=9)
for index, kind in enumerate(kinds):
    mask = labels == index
    if not mask.any():
        continue
    a = angles[index]
    u = xx[mask] * math.cos(a) + yy[mask] * math.sin(a)
    v = -xx[mask] * math.sin(a) + yy[mask] * math.cos(a)
    dark, light = (np.array(c, dtype=np.float32) for c in palette[kind])
    if kind == "vineyard":
        rows = (np.sin(u * 2 * math.pi / 3.6) > 0.35).astype(np.float32)
        t = 1.0 - rows * 0.85
    elif kind == "orchard":
        rows = (np.sin(u * 2 * math.pi / 8) > 0.4) & (np.sin(v * 2 * math.pi / 8) > 0.4)
        t = 1.0 - rows.astype(np.float32) * 0.9
    elif kind == "ploughed":
        t = 0.5 + 0.5 * np.sin(u * 2 * math.pi / 3.2) * 0.6
    elif kind == "stubble":
        t = 0.6 + 0.25 * np.sin(u * 2 * math.pi / 9)
    else:
        t = 0.55 + 0.25 * texture[mask]
    t = np.clip(t + (texture[mask] - 0.5) * 0.35, 0, 1)[:, None]
    rgb[mask] = dark * (1 - t) + light * t

hedge = np.array((52, 70, 40), dtype=np.float32)
rgb[edge] = rgb[edge] * 0.25 + hedge * 0.75

# -- river and riparian strip --------------------------------------------------
t = np.linspace(0, 1, 400)
river_x = W * (0.12 + 0.8 * t)
river_y = H * (0.62 + 0.12 * np.sin(t * 7.0) + 0.05 * np.sin(t * 17.0 + 1))
river_img = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(river_img)
d.line(list(zip(river_x, river_y)), fill=255, width=22, joint="curve")
river = np.asarray(river_img) > 0
bank_img = river_img.filter(ImageFilter.MaxFilter(39))
bank = (np.asarray(bank_img) > 0) & ~river

tree_noise = noise((H, W), 14, 3, seed=21)
trees_bank = bank & (tree_noise > 0.42)
rgb[bank] = rgb[bank] * 0.5 + np.array((96, 104, 66)) * 0.5
rgb[trees_bank] = np.array((40, 58, 34)) + tree_noise[trees_bank, None] * 25
rgb[river] = np.array((58, 72, 70)) + noise((H, W), 30, 2, seed=4)[river, None] * 18

# -- roads ---------------------------------------------------------------------
road_img = Image.new("L", (W, H), 0)
d = ImageDraw.Draw(road_img)
d.line([(0, H * 0.28), (W * 0.35, H * 0.31), (W * 0.62, H * 0.24), (W, H * 0.3)],
       fill=255, width=11, joint="curve")
d.line([(W * 0.46, 0), (W * 0.49, H * 0.3), (W * 0.52, H * 0.6), (W * 0.47, H)],
       fill=200, width=7, joint="curve")
road = np.asarray(road_img)
rgb[road > 0] = np.where((road[road > 0] > 220)[:, None], (198, 192, 180), (176, 164, 138))

# -- farmstead -----------------------------------------------------------------
farm = Image.new("RGB", (W, H))
buildings = Image.new("L", (W, H), 0)
shadows = Image.new("L", (W, H), 0)
bd, sd = ImageDraw.Draw(buildings), ImageDraw.Draw(shadows)
cx, cy = W * 0.5, H * 0.3
for _ in range(9):
    w, h = rng.uniform(12, 38), rng.uniform(9, 22)
    x, y = cx + rng.uniform(-95, 95), cy + rng.uniform(-70, 70)
    sd.rectangle([x + 6, y + 6, x + w + 6, y + h + 6], fill=255)
    bd.rectangle([x, y, x + w, y + h], fill=int(rng.choice([235, 205, 150])))
for _ in range(40):  # yard trees
    x, y = cx + rng.uniform(-150, 150), cy + rng.uniform(-110, 110)
    r = rng.uniform(4, 9)
    sd.ellipse([x - r + 5, y - r + 5, x + r + 5, y + r + 5], fill=255)
    bd.ellipse([x - r, y - r, x + r, y + r], fill=60)
b = np.asarray(buildings).astype(np.float32)
s = np.asarray(shadows) > 0
rgb[s] *= 0.55
roofs = b > 100
trees = (b > 0) & ~roofs
rgb[roofs] = np.stack([b[roofs] * 0.95, b[roofs] * 0.9, b[roofs] * 0.86], axis=1)
rgb[trees] = np.array((44, 64, 36))

# -- light, grain ---------------------------------------------------------------
rgb *= shade[..., None]
rgb += rng.normal(0, 5, rgb.shape).astype(np.float32)
ground = np.clip(rgb, 0, 255).astype(np.uint8)
Image.fromarray(ground).save(OUT / "ground.jpg", quality=92)
print("ground", ground.shape)

# =============================================================================
# Film frames
# =============================================================================

FRAME = 1800               # scan size in pixels
IMAGE = 1560               # imaged area inside the frame border
try:
    font = ImageFont.truetype("consola.ttf", 22)
    small_font = ImageFont.truetype("consola.ttf", 18)
except OSError:
    font = small_font = ImageFont.load_default()

centres = [(W * 0.30, H * 0.47), (W * 0.50, H * 0.49), (W * 0.70, H * 0.46)]
kappas = [1.2, -0.8, 0.6]
frames = []
ground_img = Image.fromarray(ground)

for number, ((gx0, gy0), kappa) in enumerate(zip(centres, kappas), start=1):
    # Ground-to-photo: rotate by kappa about the centre, 1:1 pixel scale.
    patch = ground_img.rotate(kappa, center=(gx0, gy0), resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    crop = patch.crop((int(gx0 - IMAGE / 2), int(gy0 - IMAGE / 2),
                       int(gx0 + IMAGE / 2), int(gy0 + IMAGE / 2)))
    arr = np.asarray(crop).astype(np.float32)
    ry, rx = np.mgrid[0:IMAGE, 0:IMAGE]
    radius = np.hypot(rx - IMAGE / 2, ry - IMAGE / 2) / (IMAGE / 2)
    arr *= (1 - 0.22 * radius ** 2)[..., None]          # lens falloff
    crop = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    scan = Image.new("RGB", (FRAME, FRAME), (14, 14, 13))
    offset = (FRAME - IMAGE) // 2
    scan.paste(crop, (offset, offset))
    draw = ImageDraw.Draw(scan)

    # Fiducials: corners and edge midpoints, a cross in a ring.
    marks = {}
    inset, edge_inset = 44, 34
    positions = {
        "top_left": (inset, inset), "top_middle": (FRAME / 2, edge_inset),
        "top_right": (FRAME - inset, inset), "right_middle": (FRAME - edge_inset, FRAME / 2),
        "bottom_right": (FRAME - inset, FRAME - inset), "bottom_middle": (FRAME / 2, FRAME - edge_inset),
        "bottom_left": (inset, FRAME - inset), "left_middle": (edge_inset, FRAME / 2),
    }
    for slot, (fx, fy) in positions.items():
        draw.ellipse([fx - 14, fy - 14, fx + 14, fy + 14], outline=(235, 235, 230), width=2)
        draw.line([fx - 19, fy, fx + 19, fy], fill=(235, 235, 230), width=2)
        draw.line([fx, fy - 19, fx, fy + 19], fill=(235, 235, 230), width=2)
        marks[slot] = [fx, fy]

    # Data strip down the left edge, as a mapping camera prints it.
    strip = Image.new("RGB", (1000, 40), (14, 14, 13))
    sdraw = ImageDraw.Draw(strip)
    sdraw.text((6, 6), f"BLOCK 7   FRAME {number:03d}   c = 153.690 mm   H 3880 m",
               fill=(220, 220, 210), font=small_font)
    scan.paste(strip.rotate(90, expand=True), (4, FRAME // 2 - 500))

    name = f"frame_{number:03d}"
    path = OUT / f"{name}.tif"
    scan.save(path, compression="tiff_lzw")
    frames.append({"name": name, "path": str(path), "centre": (gx0, gy0), "kappa": kappa,
                   "offset": offset, "fiducials": marks})
    print("frame", name)


def ground_to_frame(frame, x, y):
    """Where ground pixel (x, y) lands on a frame scan."""
    gx0, gy0 = frame["centre"]
    a = math.radians(frame["kappa"])
    dx, dy = x - gx0, y - gy0
    # PIL rotates counter-clockwise for positive angles in image coordinates.
    rx = dx * math.cos(a) + dy * math.sin(a)
    ry = -dx * math.sin(a) + dy * math.cos(a)
    return rx + IMAGE / 2 + frame["offset"], ry + IMAGE / 2 + frame["offset"]


# =============================================================================
# Project
# =============================================================================


def call(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method,
                          headers={"Content-Type": "application/json"} if data else {})
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


bundle = OUT / "Block_7.fidu"
call("POST", "/project/close", {})
if bundle.exists():
    import shutil
    shutil.rmtree(bundle)

call("POST", "/project/new", {"directory": str(bundle), "name": "Block 7 — Riverside farms",
                              "description": "Three-frame strip, 1:20 000"})
call("PATCH", "/project", {"projection": {"output": "ZALO19", "gcpSource": "ZALO19",
                                          "pixelSpacingX": 0.5, "pixelSpacingY": 0.5}})
call("POST", "/camera", {"kind": "film", "name": "RC30 15/4 UAG-S", "focalMm": 153.69,
                         "imageScale": 20000, "fiducialPosition": "edge_and_corner",
                         "fiducialsMm": {"top_left": [-106.0, 106.0], "top_middle": [0.0, 112.0],
                                         "top_right": [106.0, 106.0], "right_middle": [112.0, 0.0],
                                         "bottom_right": [106.0, -106.0], "bottom_middle": [0.0, -112.0],
                                         "bottom_left": [-106.0, -106.0], "left_middle": [-112.0, 0.0]}})
added = call("POST", "/images/add", {"paths": [f["path"] for f in frames]})["added"]
ids = {entry["name"]: entry["id"] for entry in added}

for frame in frames:
    image_id = ids[frame["name"]]
    for slot, (col, row) in frame["fiducials"].items():
        call("POST", f"/images/{image_id}/fiducials", {"slot": slot, "col": col, "row": row})

# Control on features a surveyor would pick: road junctions, a farm building
# corner, a bridge.
control = [(W * 0.36, H * 0.305), (W * 0.49, H * 0.30), (W * 0.52, H * 0.60),
           (W * 0.58, H * 0.42), (W * 0.42, H * 0.55), (W * 0.64, H * 0.27),
           (W * 0.27, H * 0.36), (W * 0.73, H * 0.55)]
for number, (x, y) in enumerate(control, start=1):
    observations = []
    for frame in frames:
        col, row = ground_to_frame(frame, x, y)
        if 150 < col < FRAME - 150 and 150 < row < FRAME - 150:
            observations.append({"imageId": ids[frame["name"]], "col": col, "row": row})
    if not observations:
        continue
    call("POST", "/points/gcp", {
        "id": f"G{number:04d}", "x": 18000 + x * 0.8, "y": -3790000 - y * 0.8,
        "z": float(40 + 30 * math.sin(x / 700)), "measurements": observations})

print(json.dumps(ids))
