"""Validation for photos without positions: phone photos to a scaled dense cloud.

A small scene -- two textured boxes on textured ground -- is photographed by
two "phones" walking round it: a main camera (32 photos, 26 mm equivalent, with some
barrel distortion), a quarter of its photos taken in portrait and stored on
their side with an orientation tag, as phones do, and a zoom camera (52 mm
equivalent) from further back. The photos carry only what a phone records:
make, model, focal length and its 35 mm equivalent.

Everything then runs through the engine's HTTP interface, as the app drives
it: the photos are added, aligned, scaled from one known distance, densely
matched and exported. Cameras, scale and the dense surface are compared
with the truth after the one similarity transform the photos cannot know.
"""

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, request

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

from fiducia import sfm  # noqa: E402

PY = Path(sys.executable)
PORT = 8793
BASE = f"http://127.0.0.1:{PORT}"
PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def call(method, path, payload=None, raw=False):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(BASE + path, data=data, method=method,
                          headers={"Content-Type": "application/json"} if data else {})
    try:
        with request.urlopen(req, timeout=600) as response:
            body = response.read()
            return body if raw else (json.loads(body.decode()) if body else None)
    except error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> {exc.code}: {exc.read().decode()[:400]}") from None


def wait_for_job(job_id, timeout=1800):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = next((j for j in call("GET", "/jobs?limit=60")["jobs"] if j["id"] == job_id), None)
        if job and job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise TimeoutError(job_id)


# -- the scene ------------------------------------------------------------------------------

rng = np.random.default_rng(5)
GROUND = 8.0                                                  # the ground runs +-8 m
BOXES = [(np.array([-1.0, -0.75, 0.0]), np.array([1.0, 0.75, 1.2])),
         (np.array([1.6, -1.2, 0.0]), np.array([2.4, -0.4, 0.6]))]
LIGHT = np.array([0.4, 0.3, 0.85]) / np.linalg.norm([0.4, 0.3, 0.85])


def make_texture(size=2048):
    import cv2

    texture = np.zeros((size, size, 3), np.float32)
    for sigma, weight in ((1.0, 0.35), (3.0, 0.3), (9.0, 0.25), (27.0, 0.25), (80.0, 0.3)):
        noise = rng.normal(0, 1, (size, size, 3)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
        noise /= noise.std() + 1e-9
        texture += weight * noise
    texture = 128 + 38 * texture
    for _ in range(400):
        x, y = rng.integers(0, size, 2)
        colour = rng.integers(20, 235, 3).tolist()
        if rng.random() < 0.5:
            cv2.circle(texture, (int(x), int(y)), int(rng.integers(4, 30)), colour, -1)
        else:
            w, h = rng.integers(4, 40, 2)
            cv2.rectangle(texture, (int(x), int(y)), (int(x + w), int(y + h)), colour, -1)
    return np.clip(texture, 0, 255).astype(np.float32)


TEXTURE = None
TEXELS_PER_M = 128.0


def look_at(centre, target, roll=0.0):
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.array([right, down, forward])
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ R
    return R


def render(camera, R, centre, width, height):
    """What the camera sees: rays through each pixel, through the lens's distortion."""
    import cv2

    cols, rows = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    px = np.column_stack([cols.ravel(), rows.ravel()])
    xy = sfm._normalise(px, camera["focal"], width / 2.0, height / 2.0, camera["k1"], 0.0)
    rays = np.column_stack([xy, np.ones(len(xy))]) @ R          # into the world (R^T d)
    best = np.full(len(rays), np.inf)
    u = np.zeros(len(rays))
    v = np.zeros(len(rays))
    shade = np.ones(len(rays))
    # The ground.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -centre[2] / rays[:, 2]
    hit = (t > 0) & np.isfinite(t)
    X = centre + t[:, None] * rays
    hit &= (np.abs(X[:, 0]) <= GROUND) & (np.abs(X[:, 1]) <= GROUND)
    best[hit] = t[hit]
    u[hit], v[hit] = X[hit, 0], X[hit, 1]
    shade[hit] = 0.55 + 0.45 * LIGHT[2]
    # The boxes, face by face.
    for b, (lo, hi) in enumerate(BOXES):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - centre) / rays
            t2 = (hi - centre) / rays
        near = np.minimum(t1, t2)
        far = np.maximum(t1, t2)
        enter = near.max(axis=1)
        leave = far.min(axis=1)
        axis = near.argmax(axis=1)
        hit = (enter <= leave) & (enter > 0) & (enter < best)
        best[hit] = enter[hit]
        X = centre + enter[hit, None] * rays[hit]
        a = axis[hit]
        normal = np.zeros((hit.sum(), 3))
        normal[np.arange(hit.sum()), a] = -np.sign(rays[hit][np.arange(hit.sum()), a])
        # Each face its own part of the texture.
        other = [(1, 2), (0, 2), (0, 1)]
        fu = np.where(a == 0, X[:, 1], np.where(a == 1, X[:, 0], X[:, 0]))
        fv = np.where(a == 0, X[:, 2], np.where(a == 1, X[:, 2], X[:, 1]))
        offset = 3.1 * (b * 6 + a * 2 + (normal[np.arange(len(a)), a] > 0))
        u[hit], v[hit] = fu + offset, fv + 1.7 * offset
        shade[hit] = 0.55 + 0.45 * np.clip(normal @ LIGHT, 0, 1)
        del other
    sky = ~np.isfinite(best)
    size = TEXTURE.shape[0]
    map_x = np.mod(u * TEXELS_PER_M + size / 2, size).astype(np.float32).reshape(height, width)
    map_y = np.mod(v * TEXELS_PER_M + size / 2, size).astype(np.float32).reshape(height, width)
    colour = cv2.remap(TEXTURE, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    colour = colour * shade.reshape(height, width, 1)
    colour.reshape(-1, 3)[sky] = (200, 215, 235)
    colour += rng.normal(0, 2.0, colour.shape)
    return np.clip(colour, 0, 255).astype(np.uint8)


def distance_to_scene(P):
    """Each point's distance to the nearest true surface."""
    best = np.abs(P[:, 2])
    outside = (np.abs(P[:, 0]) > GROUND) | (np.abs(P[:, 1]) > GROUND)
    best[outside] = np.inf
    for lo, hi in BOXES:
        c, h = (lo + hi) / 2, (hi - lo) / 2
        q = np.abs(P - c) - h
        out = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(q.max(axis=1), 0)
        best = np.minimum(best, np.abs(out + inside))
    return best


def save_photo(image, path, make, model, focal_mm, focal35, orientation=1):
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    picture = Image.fromarray(image)
    if orientation == 6:
        # Stored on its side: shown upright after turning a quarter clockwise.
        picture = picture.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0112] = orientation
    detail = exif.get_ifd(0x8769)
    detail[0x920A] = IFDRational(int(round(focal_mm * 100)), 100)
    detail[0xA405] = int(focal35)
    picture.save(path, "JPEG", quality=93, exif=exif)


# Two cameras. Main: 2000 x 1500, 26 mm equivalent. Zoom: 1600 x 1200, 52 mm.
DIAG_MM = math.hypot(36, 24)
MAIN = {"make": "Testphone", "model": "TP 1", "focal_mm": 4.3, "focal35": 26, "k1": -0.03,
        "size": (2000, 1500)}
ZOOM = {"make": "Testphone", "model": "TP 1", "focal_mm": 8.6, "focal35": 52, "k1": 0.0,
        "size": (1600, 1200)}
for cam in (MAIN, ZOOM):
    cam["focal"] = cam["focal35"] / DIAG_MM * math.hypot(*cam["size"])

work = Path(tempfile.mkdtemp(prefix="fiducia-scene-"))
photos_dir = work / "phone photos"
photos_dir.mkdir()
print(f"\nWorking in {work}")
print("\n=== Photographing the scene ===")
TEXTURE = make_texture()
target = np.array([0.4, -0.3, 0.5])
truth = {}
start = time.time()
shots = []
for k in range(32):
    angle = 2 * math.pi * k / 32 + rng.normal(0, 0.02)
    centre = np.array([0.4 + 6.0 * math.cos(angle), -0.3 + 6.0 * math.sin(angle), 1.5 + rng.normal(0, 0.12)])
    shots.append(("main", centre, k % 4 == 1))
for k in range(8):
    angle = 2 * math.pi * (k + 0.5) / 8
    centre = np.array([0.4 + 9.5 * math.cos(angle), -0.3 + 9.5 * math.sin(angle), 1.7 + rng.normal(0, 0.1)])
    shots.append(("zoom", centre, False))
for n, (kind, centre, portrait) in enumerate(shots):
    cam = MAIN if kind == "main" else ZOOM
    width, height = cam["size"]
    if portrait:
        width, height = height, width
    aim = target + rng.normal(0, 0.15, 3)
    R = look_at(centre, aim, roll=rng.normal(0, 0.03))
    image = render(cam, R, centre, width, height)
    name = f"IMG_{n + 1:04d}"
    save_photo(image, photos_dir / f"{name}.jpg", cam["make"], cam["model"], cam["focal_mm"],
               cam["focal35"], orientation=6 if portrait else 1)
    truth[name] = {"centre": centre, "R": R, "kind": kind, "portrait": portrait}
print(f"  {len(shots)} photos ({sum(1 for s in shots if s[2])} portrait, stored on their side) "
      f"in {time.time() - start:.0f} s")

# -- the engine ---------------------------------------------------------------------------

print("\n=== Starting the engine ===")
log = open(work / "engine.log", "w")
engine_exe = os.environ.get("FIDUCIA_ENGINE_EXE")
command = [engine_exe] if engine_exe else [str(PY), str(ROOT / "engine" / "server.py")]
server = subprocess.Popen(command + ["--port", str(PORT)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
for _ in range(120):
    try:
        if call("GET", "/health")["ok"]:
            break
    except Exception:
        time.sleep(0.5)

try:
    bundle = work / "Scene.fidu"
    call("POST", "/project/new", {"directory": str(bundle), "name": "Boxes"})
    call("PATCH", "/project", {"workflow": "scene"})
    paths = sorted(str(p) for p in photos_dir.glob("*.jpg"))
    added = call("POST", "/images/add", {"paths": paths})["added"]
    check("every photo added", len(added) == len(paths), f"{len(added)} of {len(paths)}")
    portrait_ids = {i["id"] for i in added if truth[i["name"]]["portrait"]}
    upright_ok = all(i["height"] > i["width"] for i in added if i["id"] in portrait_ids)
    check("portrait photos stood upright", upright_ok and all(i.get("sourcePath") for i in added if i["id"] in portrait_ids),
          f"{len(portrait_ids)} portrait photos, copies in the project's photos folder")
    names = {i["id"]: i["name"] for i in added}

    print("\n=== Alignment ===")
    start = time.time()
    job = wait_for_job(call("POST", "/scene/align")["job"]["id"])
    check("alignment ran", job["status"] == "done", job.get("error") or job["message"])
    scene = call("GET", "/project")["state"]["scene"]
    check("every photo placed", scene["placed"] == len(paths),
          f"{scene['placed']} of {len(paths)} in {time.time() - start:.0f} s, {scene['points']:,} points, "
          f"{scene['reprojectionRmsPx']:.2f} px RMS")
    cams = scene["cameras"]
    check("two cameras told apart", len(cams) == 2, "; ".join(f"{c['name']}: {c['photos']} photos" for c in cams))
    for c in cams:
        true = MAIN if "26" in c["name"] else ZOOM
        check(f"focal length of the {true['focal35']} mm camera", abs(c["focalPx"] / true["focal"] - 1) < 0.01,
              f"{c['focalPx']:.0f} px against {true['focal']:.0f}, k1 {c['k1']:+.4f} (true {true['k1']:+.3f})")
    check("unscaled until a distance is given", scene["transform"]["scaleSource"] == "none")

    view = call("GET", "/scene/view")
    ids = [c["id"] for c in view["cameras"]]
    model_c = np.array([c["centre"] for c in view["cameras"]])
    true_c = np.array([truth[names[i]]["centre"] for i in ids])
    s, Rm, t = sfm.similarity(model_c, true_c)
    error_c = np.linalg.norm((s * model_c @ Rm.T + t) - true_c, axis=1)
    check("camera positions", np.sqrt(np.mean(error_c ** 2)) < 0.02,
          f"RMS {100 * np.sqrt(np.mean(error_c ** 2)):.2f} cm, worst {100 * error_c.max():.2f} cm, ring 12 m across")
    up = Rm.T @ np.array([0, 0, 1.0])
    check("model stood upright", math.degrees(math.acos(min(1.0, up[2]))) < 3,
          f"vertical {math.degrees(math.acos(min(1.0, up[2]))):.2f} degrees off")

    print("\n=== Scale from a known distance ===")
    # The two top corners along the big box's long edge, 2.000 m apart, as
    # the person would pick them in the 3D view.
    corners = np.array([[-1.0, -0.75, 1.2], [1.0, -0.75, 1.2]])
    picked = ((corners - t) @ Rm) / s
    call("POST", "/scene/scale", {"a": picked[0].tolist(), "b": picked[1].tolist(), "distance": 2.0})
    view = call("GET", "/scene/view")
    model_c = np.array([c["centre"] for c in view["cameras"]])
    s2, Rm2, t2 = sfm.similarity(model_c, true_c)
    check("scaled to metres", abs(s2 - 1) < 0.003, f"scale {s2:.5f} against the truth")

    sparse = call("GET", "/scene/cloud?which=sparse", raw=True)
    count = struct.unpack("<I", sparse[:4])[0]
    check("sparse cloud for the 3D view", len(sparse) == 4 + 15 * count, f"{count:,} points")

    print("\n=== Dense cloud ===")
    start = time.time()
    job = wait_for_job(call("POST", "/scene/dense", {"detail": "medium"})["job"]["id"])
    check("dense matching ran", job["status"] == "done", job.get("error") or job["message"])
    dense_state = call("GET", "/project")["state"]["scene"]["dense"]
    print(f"  {dense_state['points']:,} points from {dense_state['references']} photos, "
          f"{len(dense_state['pairsFailed'])} pairs failed, {time.time() - start:.0f} s")
    raw = call("GET", "/scene/cloud?which=dense&limit=5000000", raw=True)
    count = struct.unpack("<I", raw[:4])[0]
    P = np.frombuffer(raw[4:4 + 12 * count], "<f4").reshape(-1, 3).astype(float)
    P = P @ Rm2.T * s2 + t2
    d = distance_to_scene(P)
    check("dense points on the true surfaces", np.median(d) < 0.006 and np.percentile(d, 90) < 0.02,
          f"median {1000 * np.median(d):.1f} mm, 90% within {1000 * np.percentile(d, 90):.1f} mm, "
          f"{100 * np.mean(d < 0.01):.0f}% within 1 cm")
    from scipy.spatial import cKDTree

    tree = cKDTree(P)
    lo, hi = BOXES[0]
    samples = []
    for z in np.arange(0.05, 1.2, 0.03):
        for x in np.arange(-0.95, 1.0, 0.03):
            samples += [[x, lo[1], z], [x, hi[1], z]]
        for y in np.arange(-0.7, 0.75, 0.03):
            samples += [[lo[0], y, z], [hi[0], y, z]]
    # Its sides: from standing height its top is seen only edge-on.
    near, _ = tree.query(np.array(samples))
    check("the big box covered", np.mean(near < 0.02) > 0.85,
          f"{100 * np.mean(near < 0.02):.0f}% of its sides within 2 cm of a point")
    laz = call("GET", "/scene/view")["lazPath"]
    check("dense cloud ready for the point cloud tools", laz and Path(laz).exists(), Path(laz).name if laz else "")
    inspected = call("POST", "/lidar/inspect", {"path": laz})
    check("point cloud tools read it", inspected["pointCount"] == dense_state["points"],
          f"{inspected['pointCount']:,} points")

    print("\n=== Export ===")
    for suffix in (".ply", ".laz"):
        written = call("POST", "/scene/export", {"which": "dense", "path": str(work / f"export{suffix}")})
        check(f"export {suffix}", Path(written["path"]).exists() and written["points"] == dense_state["points"],
              f"{Path(written['path']).stat().st_size / 1e6:.1f} MB")
finally:
    server.terminate()
    server.wait(timeout=20)
    log.close()

print("\n" + "=" * 64)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
print("=" * 64)
if not FAIL and not os.environ.get("FIDUCIA_KEEP"):
    shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if FAIL else 0)
