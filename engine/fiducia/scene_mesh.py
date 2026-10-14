"""A coloured surface mesh from the dense matching's depth maps.

Points show where the surface is; a mesh joins them into one, which is
what 3D software, game engines and a walk-through need. The dense step
keeps each photo's checked depths, and here they are fused as in the
scanning systems that followed KinectFusion (Curless and Levoy, 1996;
Newcombe et al., 2011):

1. A thin shell of small cubes (voxels) is set round the dense points: the
   only place the surface can be.
2. Each photo says, for every voxel it sees, how far in front of or behind
   its measured surface the voxel lies. These signed distances, cut off at
   a few voxels, are averaged over all the photos (a truncated signed
   distance function), so each photo's noise averages out and a voxel seen
   empty by one photo and solid by another is settled by the majority.
3. The surface is where the average distance is zero. Surface nets
   (Gibson, 1998) puts one vertex in each cube the surface passes through,
   at the mean of its edge crossings, and joins neighbouring cubes' vertices
   into faces: smooth, and with no ambiguous cases to resolve.
4. Each vertex takes its colour from the photos that saw it, and stray
   fragments, far smaller than the subject, are dropped.

Everything is in the reconstruction's own frame; ``scene`` scales and turns it.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

__all__ = ["MESH_DETAIL", "MeshOptions", "build_mesh", "vertex_normals", "write_mesh"]

# The voxel size as a multiple of the dense cloud's point spacing, and the most
# voxels the surface may occupy: finer detail is coarsened to stay within it.
MESH_DETAIL = {"low": (3.0, 1_000_000), "medium": (2.0, 2_500_000), "high": (1.25, 5_000_000)}

_BITS = 21
_HALF = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1
_EX, _EY, _EZ = 1 << (2 * _BITS), 1 << _BITS, 1


@dataclass
class MeshOptions:
    detail: str = "medium"
    truncation: float = 3.0          # voxels: how far from the surface distances count
    band: int = 2                    # voxels each side of the dense points that may hold the surface
    min_weight: float = 1.0          # photos that must have seen a voxel
    min_fragment: float = 0.002      # share of the faces below which a separate piece is dropped
    fill: int = 120                  # enclosed depth-map holes up to this many pixels across are filled where flat
    chunk: int = 2_000_000           # voxels projected at a time


def _pack(q: np.ndarray) -> np.ndarray:
    q = np.clip(q + _HALF, 0, _MASK)
    return (q[:, 0] << (2 * _BITS)) | (q[:, 1] << _BITS) | q[:, 2]


def _unpack(keys: np.ndarray) -> np.ndarray:
    return np.column_stack([(keys >> (2 * _BITS)) & _MASK, (keys >> _BITS) & _MASK, keys & _MASK]) - _HALF


def _find(keys: np.ndarray, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each wanted key sits in the sorted keys, and whether it is there."""
    at = np.minimum(np.searchsorted(keys, wanted), len(keys) - 1)
    return at, keys[at] == wanted


def _shell(points: np.ndarray, origin: np.ndarray, voxel: float, band: int) -> np.ndarray:
    """The sorted keys of the voxels within ``band`` of any point."""
    keys = np.unique(_pack(np.floor((points - origin) / voxel).astype(np.int64)))
    for _ in range(band):
        keys = np.unique(np.concatenate([keys] + [keys + s * o for o in (_EX, _EY, _EZ) for s in (1, -1)]))
    return keys


def _fill_small_holes(depth: np.ndarray, size: int) -> np.ndarray:
    """A depth map with its enclosed holes filled where the surface round them is flat.

    Plain paint, a white wall or a still puddle has nothing to match, so its
    depths are missing; such patches are nearly always flat. Each hole that is
    enclosed by measured depths and no more than ``size`` pixels across is
    filled from a plane fitted to the ring of depths round it -- in inverse
    depth, which runs linearly across the photo over a plane -- but only if
    the ring fits that plane closely. Holes at the photo's edge (sky, the
    world beyond the subject) and holes over an edge or a corner stay open.
    """
    import cv2

    known = np.isfinite(depth)
    if known.all() or not known.any():
        return depth
    count, label, stats, _ = cv2.connectedComponentsWithStats((~known).astype(np.uint8), connectivity=4)
    height, width = depth.shape
    out = depth.copy()
    ring_kernel = np.ones((5, 5), np.uint8)
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        if w > size or h > size or x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(width, x + w + 3), min(height, y + h + 3)
        hole = label[y0:y1, x0:x1] == k
        ring = cv2.dilate(hole.astype(np.uint8), ring_kernel).astype(bool) & ~hole
        values = depth[y0:y1, x0:x1][ring]
        ok = np.isfinite(values)
        if ok.sum() < max(8, 0.8 * ring.sum()):
            continue
        rows, cols = np.nonzero(ring)
        a = np.column_stack([cols[ok], rows[ok], np.ones(int(ok.sum()))]).astype(float)
        inverse = 1.0 / values[ok]
        plane, *_ = np.linalg.lstsq(a, inverse, rcond=None)
        residual = np.sqrt(np.mean((a @ plane - inverse) ** 2))
        if residual > 0.012 * np.mean(inverse):
            continue
        rows, cols = np.nonzero(hole)
        fitted = np.column_stack([cols, rows, np.ones(len(rows))]) @ plane
        if np.any(fitted <= 0):
            continue
        out[y0 + rows, x0 + cols] = 1.0 / fitted
    return out

def build_mesh(photos: dict, depth_maps: dict, stride: int, images, points: np.ndarray, spacing: float,
               options: Optional[MeshOptions] = None,
               progress: Optional[Callable[[float, str], None]] = None,
               should_cancel: Optional[Callable[[], bool]] = None) -> dict:
    """Vertices, triangles and vertex colours of the surface.

    ``photos`` are ``scene_dense._Photo`` by index, ``depth_maps`` the paths of
    their checked depths (every ``stride``-th working pixel), ``images`` their
    working-size colours, ``points`` the dense cloud and ``spacing`` its point
    spacing, all in the reconstruction's frame.
    """
    options = options or MeshOptions()
    say = progress or (lambda fraction, message: None)
    cancelled = should_cancel or (lambda: False)
    factor, budget = MESH_DETAIL.get(options.detail, MESH_DETAIL["medium"])
    points = np.asarray(points, float)
    origin = points.mean(axis=0)

    # A voxel a little larger than the point spacing, coarsened until the
    # surface fits the budget: surface voxels grow as the inverse square.
    voxel = factor * spacing
    surface = len(np.unique(_pack(np.floor((points - origin) / voxel).astype(np.int64))))
    if surface > budget:
        voxel *= math.sqrt(surface / budget)
    say(0.02, "Filling small gaps")
    # Each photo's depths with their small flat gaps filled; the filled
    # places join the shell, which otherwise follows only the dense points.
    filled: dict = {}
    extra = []
    for i in sorted(depth_maps):
        raw = np.load(depth_maps[i])
        filled[i] = _fill_small_holes(raw, options.fill)
        rows, cols = np.nonzero(np.isfinite(filled[i]) & ~np.isfinite(raw))
        if rows.size:
            photo = photos[i]
            z = filled[i][rows, cols]
            xy = photo.rays(cols * stride, rows * stride)
            extra.append((np.column_stack([xy * z[:, None], z]) - photo.t) @ photo.R)
    say(0.04, "Setting out the voxels")
    keys = _shell(np.concatenate([points] + extra) if extra else points, origin, voxel, options.band)
    n = len(keys)
    trunc = options.truncation * voxel

    tsdf = np.zeros(n, np.float32)
    weight = np.zeros(n, np.float32)
    colour = np.zeros((n, 3), np.float32)
    colour_weight = np.zeros(n, np.float32)

    order = sorted(depth_maps)
    for done, i in enumerate(order):
        if cancelled():
            raise InterruptedError("Cancelled")
        photo = photos[i]
        depth = filled.pop(i)
        picture = images[i]
        for start in range(0, n, options.chunk):
            part = slice(start, min(n, start + options.chunk))
            centres = origin + (_unpack(keys[part]) + 0.5) * voxel
            px, z = photo.project(centres)
            # Behind the camera, or far off to the side, a voxel has no pixel.
            px = np.where(np.isfinite(px) & (np.abs(px) < 1e6), px, -1.0)
            u = np.rint(px[:, 0] / stride).astype(np.int64)
            v = np.rint(px[:, 1] / stride).astype(np.int64)
            seen = (z > 0) & (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
            measured = np.full(len(z), np.nan, np.float32)
            measured[seen] = depth[v[seen], u[seen]]
            sdf = measured - z
            # In front of the surface counts as far as it is seen; behind it,
            # only just behind: further back the photo cannot see.
            use = np.isfinite(sdf) & (sdf > -trunc)
            if not use.any():
                continue
            idx = np.arange(part.start, part.stop)[use]
            observed = np.clip(sdf[use] / trunc, -1.0, 1.0)
            w = weight[idx]
            tsdf[idx] = (tsdf[idx] * w + observed) / (w + 1.0)
            weight[idx] = w + 1.0
            near = np.abs(sdf[use]) < trunc
            if near.any():
                cols = np.clip(np.rint(px[use][near, 0]).astype(np.int64), 0, picture.shape[1] - 1)
                rows = np.clip(np.rint(px[use][near, 1]).astype(np.int64), 0, picture.shape[0] - 1)
                colour[idx[near]] += picture[rows, cols, :3]
                colour_weight[idx[near]] += 1.0
        say(0.05 + 0.65 * (done + 1) / len(order), f"Fused the depths of {done + 1} of {len(order)} photos")

    say(0.72, "Finding the surface")
    vertices, faces, colours = _surface_nets(keys, tsdf, weight, colour, colour_weight, origin, voxel,
                                             options.min_weight)
    if not len(faces):
        raise ValueError("No surface was found: too few photos agree on any part of it.")
    say(0.9, "Dropping stray fragments")
    vertices, faces, colours, dropped = _keep_large(vertices, faces, colours, options.min_fragment)
    say(1.0, f"{len(faces):,} triangles")
    return {
        "vertices": vertices, "faces": faces, "colours": colours,
        "summary": {"vertices": int(len(vertices)), "triangles": int(len(faces)), "voxel": voxel,
                    "voxels": int(n), "fragmentsDropped": dropped, "detail": options.detail},
    }


_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
          if sum(x != y for x, y in zip(_CORNERS[a], _CORNERS[b])) == 1]


def _surface_nets(keys, tsdf, weight, colour, colour_weight, origin, voxel, min_weight):
    good = weight >= min_weight
    cand = np.nonzero(good)[0]
    offsets = np.array([dx * _EX + dy * _EY + dz * _EZ for dx, dy, dz in _CORNERS], np.int64)
    at, found = _find(keys, keys[cand][:, None] + offsets[None, :])
    valid = found.all(axis=1) & good[at].all(axis=1)
    t = tsdf[at]
    inside = t < 0
    crossing = valid & inside.any(axis=1) & ~inside.all(axis=1)
    cubes, at, t, inside = cand[crossing], at[crossing], t[crossing], inside[crossing]

    corner = np.array(_CORNERS, float)
    total = np.zeros((len(cubes), 3))
    count = np.zeros(len(cubes))
    for a, b in _EDGES:
        cross = inside[:, a] != inside[:, b]
        s = t[cross, a] / (t[cross, a] - t[cross, b])
        total[cross] += corner[a] + s[:, None] * (corner[b] - corner[a])
        count[cross] += 1
    local = total / count[:, None]
    vertices = origin + (_unpack(keys[cubes]) + 0.5 + local) * voxel

    cw = colour_weight[at]
    summed = (colour[at]).sum(axis=1)
    colours = np.where(cw.sum(axis=1)[:, None] > 0, summed / np.maximum(cw.sum(axis=1), 1)[:, None], 160.0)
    colours = np.clip(np.rint(colours), 0, 255).astype(np.uint8)

    # A face for each voxel edge the surface crosses, joining the vertices of
    # the four cubes round that edge, wound so it faces the empty side.
    cube_keys = keys[cubes]
    faces = []
    for ea, eb, ec in ((_EX, _EY, _EZ), (_EY, _EZ, _EX), (_EZ, _EX, _EY)):
        here = cand
        there, ok = _find(keys, keys[here] + ea)
        ok &= good[there]
        here, there = here[ok], there[ok]
        below = tsdf[here] < 0
        cross = below != (tsdf[there] < 0)
        here, below = here[cross], below[cross]
        ring = keys[here][:, None] - np.array([eb + ec, ec, 0, eb], np.int64)[None, :]
        slot, present = _find(cube_keys, ring)
        whole = present.all(axis=1)
        quad = slot[whole]
        flip = ~below[whole]
        quad[flip] = quad[flip][:, ::-1]
        faces.append(np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]))
    faces = np.concatenate(faces).astype(np.int64)
    return vertices, faces, colours


def _keep_large(vertices, faces, colours, min_share):
    """The mesh without pieces far smaller than the whole, and without unused vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(vertices)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    count, label = connected_components(graph, directed=False)
    face_label = label[faces[:, 0]]
    sizes = np.bincount(face_label, minlength=count)
    smallest = max(50, min_share * len(faces))
    dropped = int(((sizes > 0) & (sizes < smallest)).sum())
    faces = faces[(sizes >= smallest)[face_label]]
    used = np.zeros(n, bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    return vertices[used], remap[faces], colours[used], dropped


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normals at each vertex: the area-weighted mean of its faces'."""
    v = np.asarray(vertices, float)
    face = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    normals = np.zeros_like(v)
    for k in range(3):
        np.add.at(normals, faces[:, k], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def write_mesh(path: str, vertices: np.ndarray, faces: np.ndarray, colours: np.ndarray) -> dict:
    """A coloured mesh to a file: GLB (glTF) for 3D software, game engines and
    the web, PLY for meshing software, or OBJ."""
    suffix = Path(path).suffix.lower()
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, np.int64)
    colours = np.asarray(colours, np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    partial = str(path) + ".part"
    if suffix == ".glb":
        _write_glb(partial, vertices, faces, colours)
    elif suffix == ".ply":
        record = np.zeros(len(vertices), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                                                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        record["x"], record["y"], record["z"] = vertices.T
        record["red"], record["green"], record["blue"] = colours.T
        triangles = np.zeros(len(faces), dtype=[("n", "u1"), ("v", "<i4", (3,))])
        triangles["n"] = 3
        triangles["v"] = faces
        header = ("ply\nformat binary_little_endian 1.0\ncomment Fiducia\n"
                  f"element vertex {len(vertices)}\n"
                  "property double x\nproperty double y\nproperty double z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        with open(partial, "wb") as handle:
            handle.write(header.encode("ascii"))
            handle.write(record.tobytes())
            handle.write(triangles.tobytes())
    elif suffix == ".obj":
        # Colours after the position, as MeshLab, Blender and CloudCompare read them.
        rgb = colours / 255.0
        with open(partial, "w", encoding="ascii") as handle:
            handle.write("# Fiducia\n")
            np.savetxt(handle, np.column_stack([vertices, rgb]), fmt="v %.4f %.4f %.4f %.3f %.3f %.3f")
            np.savetxt(handle, faces + 1, fmt="f %d %d %d")
    else:
        raise ValueError(f"Meshes are written as .glb, .ply or .obj, not {suffix or 'no extension'}")
    Path(partial).replace(path)
    return {"path": str(path), "vertices": int(len(vertices)), "triangles": int(len(faces))}


def _write_glb(path, vertices, faces, colours):
    """glTF 2.0, binary. glTF is y-up and the scene z-up, so (x, y, z) is
    written (x, z, -y); positions are relative to the mesh's centre, which the
    node carries, so map coordinates keep their precision."""
    centre = vertices.mean(axis=0) if len(vertices) else np.zeros(3)
    local = vertices - centre
    position = np.column_stack([local[:, 0], local[:, 2], -local[:, 1]]).astype("<f4")
    normal = vertex_normals(position.astype(float), faces).astype("<f4")
    rgba = np.column_stack([colours, np.full(len(colours), 255, np.uint8)]).astype(np.uint8)
    index = faces.astype("<u4").ravel()

    blobs, views = [], []
    offset = 0
    for blob, target in ((position.tobytes(), 34962), (normal.tobytes(), 34962), (rgba.tobytes(), 34962),
                         (index.tobytes(), 34963)):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": target})
        pad = (-len(blob)) % 4
        blobs.append(blob + b"\0" * pad)
        offset += len(blob) + pad
    binary = b"".join(blobs)
    gltf = {
        "asset": {"version": "2.0", "generator": "Fiducia"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "translation": [float(centre[0]), float(centre[2]), float(-centre[1])]}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0,
                                                "roughnessFactor": 1.0}, "doubleSided": True}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(position), "type": "VEC3",
             "min": position.min(axis=0).tolist(), "max": position.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(normal), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "normalized": True, "count": len(rgba), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": len(index), "type": "SCALAR"},
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * ((-len(text)) % 4)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(binary)))
        handle.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        handle.write(struct.pack("<II", len(binary), 0x004E4942) + binary)
