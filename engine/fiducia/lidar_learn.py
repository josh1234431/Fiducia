"""Point classification learned from the operator's own labels.

The operator labels a few points of each class in a section view; a random
forest learns from simple, explainable measurements of each point and its
neighbourhood, and labels the rest of the cloud. Nothing is tied to one use:
the classes are whatever the operator labels -- vegetation from noise in a
dense canopy, wires from poles, water from ground.

The measurements are the standard ones for point clouds (Weinmann et al.,
2015): height above the ground, the return it was, and the shape of its
neighbourhood at two scales -- how line-like, plane-like or scattered it is,
how vertical, how dense, and where the point sits within it. The forest is a
small histogram-based implementation on numpy, so the method is visible and
the engine needs nothing beyond what it already ships.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import lidar

SCALES = (10, 30)          # neighbours per scale
CHUNK = 100_000            # points per pass when measuring the whole cloud


# -- measurements ---------------------------------------------------------------

def _ground_reference(path: str, cloud) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A ground height at any (x, y): from class 2 when there is enough of it,
    otherwise a coarse minimum surface, which is close enough for a feature.
    Streamed into one grid for the whole cloud, so every tile measures height
    against the same ground."""
    from .lidar_tiles import arrays, stream

    has_ground = int(cloud.manifest["classHistogram"].get("2", 0)) >= 50
    cell = 1.0 if has_ground else 5.0
    west, south, east, north = cloud.bounds
    bounds = lidar._snapped_bounds(np.array([west, east]), np.array([south, north]), cell)
    grid = lidar._Grid(bounds, cell, "mean" if has_ground else "minimum")
    for _, points in stream(path):
        a = arrays(points)
        use = (a["classification"] == 2) if has_ground else ~np.isin(a["classification"], lidar.NOISE_CLASSES)
        grid.add(a["x"][use], a["y"][use], a["z"][use])
    surface = lidar._fill_all(grid.result())
    return lambda px, py: lidar._sample_grid(surface, bounds, cell, px, py)


def feature_names(with_intensity: bool) -> list[str]:
    names = ["height above ground", "return number", "number of returns", "return ratio",
             "first return", "last return", "nearest neighbour", "nearest neighbour above",
             "nearest neighbour below"]
    for k in SCALES:
        names += [f"linearity ({k})", f"planarity ({k})", f"scattering ({k})", f"verticality ({k})",
                  f"height spread ({k})", f"above neighbours' lowest ({k})",
                  f"below neighbours' highest ({k})", f"density ({k})"]
    if with_intensity:
        names.append("intensity")
    return names


class Features:
    """Measures points of one tile against that tile and a margin around it.

    The margin must reach further than the widest neighbourhood, so a point at
    the tile's edge is measured exactly as anywhere else.
    """

    def __init__(self, d: dict, ground: Callable, intensity_scale: Optional[float]):
        from scipy.spatial import cKDTree

        self.x, self.y, self.z = d["x"], d["y"], d["z"]
        self.classification = d["classification"]
        self.return_number = d["return_number"].astype(np.float64)
        self.number_of_returns = np.maximum(d["number_of_returns"].astype(np.float64), 1)
        self.intensity = d["intensity"] / intensity_scale if intensity_scale else None
        self.names = feature_names(self.intensity is not None)
        self.ground = ground
        # Neighbours are searched among real points only, in local coordinates.
        usable = ~np.isin(self.classification, lidar.NOISE_CLASSES)
        self.origin = (float(self.x.min()), float(self.y.min())) if self.x.size else (0.0, 0.0)
        self.xyz = np.column_stack([self.x - self.origin[0], self.y - self.origin[1], self.z])
        self.searchable = np.nonzero(usable)[0]
        self.tree = cKDTree(self.xyz[self.searchable]) if self.searchable.size > max(SCALES) else None

    def measure(self, index: np.ndarray) -> np.ndarray:
        out = np.empty((index.size, len(self.names)), dtype=np.float32)
        for start in range(0, index.size, CHUNK):
            part = index[start:start + CHUNK]
            out[start:start + part.size] = self._measure(part)
        return out

    def _measure(self, index: np.ndarray) -> np.ndarray:
        if self.tree is None:
            return np.zeros((index.size, len(self.names)))
        points = self.xyz[index]
        columns = [
            self.z[index] - self.ground(self.x[index], self.y[index]),
            self.return_number[index],
            self.number_of_returns[index],
            self.return_number[index] / self.number_of_returns[index],
            (self.return_number[index] == 1).astype(float),
            (self.return_number[index] == self.number_of_returns[index]).astype(float),
        ]
        widest = max(SCALES)
        distance, found = self.tree.query(points, k=min(widest + 1, self.searchable.size), workers=-1)
        # A point that is itself searchable finds itself first; drop it.
        itself = distance[:, 0] < 1e-9
        distance = np.where(itself[:, None], distance, np.concatenate([np.zeros((len(points), 1)), distance[:, :-1]], axis=1))[:, 1:]
        found = np.where(itself[:, None], found, np.concatenate([found[:, :1], found[:, :-1]], axis=1))[:, 1:]
        neighbours = self.xyz[self.searchable[found]]
        # How isolated it is: overall, and from anything above or below it.
        rise = neighbours[:, :, 2] - points[:, None, 2]
        columns += [
            distance[:, 0],
            np.where(rise > 0.05, distance, np.inf).min(axis=1).clip(max=50.0),
            np.where(rise < -0.05, distance, np.inf).min(axis=1).clip(max=50.0),
        ]
        for k in SCALES:
            near = neighbours[:, :k]
            centred = near - near.mean(axis=1, keepdims=True)
            covariance = np.einsum("nki,nkj->nij", centred, centred) / k
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values[:, ::-1], 1e-12)       # largest first
            l1, l2, l3 = values[:, 0], values[:, 1], values[:, 2]
            normal_z = np.abs(vectors[:, 2, 0])                # eigenvector of the smallest value
            nz = near[:, :, 2]
            radius = np.maximum(distance[:, k - 1], 1e-3)
            columns += [
                (l1 - l2) / l1,
                (l2 - l3) / l1,
                l3 / l1,
                1.0 - normal_z,
                nz.std(axis=1),
                points[:, 2] - nz.min(axis=1),
                nz.max(axis=1) - points[:, 2],
                k / (4.0 / 3.0 * math.pi * radius ** 3),
            ]
        if self.intensity is not None:
            columns.append(self.intensity[index])
        return np.column_stack(columns)


# -- the forest -----------------------------------------------------------------

class Forest:
    """A random forest on binned features (Breiman, 2001), balanced across classes.

    Each feature is cut into quantile bins once, so finding a split is a
    histogram per node rather than a sort. Classes are weighted so that a
    class with few labels counts as much as one with many.
    """

    def __init__(self, trees: int = 40, max_depth: int = 14, min_leaf: int = 3,
                 bins: int = 32, seed: int = 0):
        self.n_trees, self.max_depth, self.min_leaf, self.n_bins = trees, max_depth, min_leaf, bins
        self.rng = np.random.default_rng(seed)

    def _bin(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.uint8)
        for f, edges in enumerate(self.edges):
            out[:, f] = np.searchsorted(edges, X[:, f], side="right")
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Forest":
        self.classes = np.unique(y)
        C = self.classes.size
        target = np.searchsorted(self.classes, y)
        counts = np.bincount(target, minlength=C).astype(float)
        weight = (target.size / (C * counts))[target]
        X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        self.edges = [np.unique(np.quantile(X[:, f], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                      for f in range(X.shape[1])]
        Xb = self._bin(X)
        n, F = Xb.shape
        per_split = max(1, int(round(math.sqrt(F))))
        self.importance = np.zeros(F)
        self.trees = []
        oob_votes = np.zeros((n, C))

        for _ in range(self.n_trees):
            sample = self.rng.integers(0, n, n)
            in_bag = np.zeros(n, dtype=bool)
            in_bag[sample] = True
            nodes = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}

            def new_node():
                for key in nodes:
                    nodes[key].append(-1 if key != "value" else None)
                return len(nodes["feature"]) - 1

            root = new_node()
            stack = [(root, sample, 0)]
            while stack:
                node, rows, depth = stack.pop()
                w = weight[rows]
                distribution = np.bincount(target[rows], weights=w, minlength=C)
                total = distribution.sum()
                nodes["value"][node] = distribution / max(total, 1e-12)
                if depth >= self.max_depth or rows.size < 2 * self.min_leaf or (distribution > 0).sum() < 2:
                    continue
                parent_gini = 1.0 - ((distribution / total) ** 2).sum()
                best = (0.0, -1, -1)
                for f in self.rng.choice(F, per_split, replace=False):
                    hist = np.bincount(Xb[rows, f].astype(np.int64) * C + target[rows], weights=w,
                                       minlength=(self.n_bins + 1) * C).reshape(-1, C)
                    left = np.cumsum(hist, axis=0)[:-1]
                    right = distribution - left
                    lw, rw = left.sum(axis=1), right.sum(axis=1)
                    valid = (lw > 0) & (rw > 0)
                    if not valid.any():
                        continue
                    gini_l = 1.0 - ((left / np.maximum(lw, 1e-12)[:, None]) ** 2).sum(axis=1)
                    gini_r = 1.0 - ((right / np.maximum(rw, 1e-12)[:, None]) ** 2).sum(axis=1)
                    gain = parent_gini - (lw * gini_l + rw * gini_r) / total
                    gain[~valid] = -1
                    b = int(np.argmax(gain))
                    if gain[b] > best[0]:
                        best = (float(gain[b]), int(f), b)
                gain, f, b = best
                if f < 0:
                    continue
                goes_left = Xb[rows, f] <= b
                if goes_left.sum() < self.min_leaf or (~goes_left).sum() < self.min_leaf:
                    continue
                self.importance[f] += gain * total
                nodes["feature"][node], nodes["threshold"][node] = f, b
                left_node, right_node = new_node(), new_node()
                nodes["left"][node], nodes["right"][node] = left_node, right_node
                stack.append((left_node, rows[goes_left], depth + 1))
                stack.append((right_node, rows[~goes_left], depth + 1))

            tree = {
                "feature": np.array(nodes["feature"]), "threshold": np.array(nodes["threshold"]),
                "left": np.array(nodes["left"]), "right": np.array(nodes["right"]),
                "value": np.array(nodes["value"]),
            }
            self.trees.append(tree)
            out = ~in_bag
            if out.any():
                oob_votes[out] += self._tree_proba(tree, Xb[out])

        self.importance /= max(self.importance.sum(), 1e-12)
        voted = oob_votes.sum(axis=1) > 0
        self.oob_target = target[voted]
        self.oob_predicted = np.argmax(oob_votes[voted], axis=1)
        return self

    def _tree_proba(self, tree, Xb):
        node = np.zeros(Xb.shape[0], dtype=np.int64)
        for _ in range(self.max_depth + 1):
            split = tree["feature"][node] >= 0
            if not split.any():
                break
            rows = np.nonzero(split)[0]
            f = tree["feature"][node[rows]]
            left = Xb[rows, f] <= tree["threshold"][node[rows]]
            node[rows] = np.where(left, tree["left"][node[rows]], tree["right"][node[rows]])
        return tree["value"][node]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = self._bin(np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
        total = np.zeros((X.shape[0], self.classes.size))
        for tree in self.trees:
            total += self._tree_proba(tree, Xb)
        return total / len(self.trees)


# -- training and applying -------------------------------------------------------

@dataclass
class LearnOptions:
    output_path: str = ""
    trees: int = 40
    max_depth: int = 14
    change_classes: Optional[list] = None   # only points now in these classes may change; None: any
    min_confidence: float = 0.0             # below this a point keeps its class
    uncertain_spots: int = 8
    crs: Optional[str] = None


def train_and_apply(
    path: str,
    labels: dict,
    options: LearnOptions,
    progress: Optional[Callable[[float, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Learn the classes from labelled points and label the whole cloud with them.

    Labelled points keep the label they were given. Everything else takes the
    forest's class, unless it is below the confidence asked for, or its
    current class is not one that may change. The out-of-bag estimate (each
    tree judged on the labels it never saw) says how well it learned, per
    class, without setting labels aside. The places it is least sure of come
    back so the operator can label there next.

    The cloud is measured and classified tile by tile, with a margin wide
    enough for every neighbourhood, so memory follows the tile, not the cloud.
    """
    from .lidar_tiles import TiledCloud, ensure_memory, header_of, release, scratch, write_classified

    def step(fraction, message):
        if progress:
            progress(fraction, message)
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    index = np.array([int(k) for k in labels], dtype=np.int64)
    target = np.array([int(v) for v in labels.values()], dtype=np.int64)
    order = np.argsort(index)
    index, target = index[order], target[order]
    present, counts = np.unique(target, return_counts=True)
    if present.size < 2:
        raise ValueError("Label points of at least two classes before training.")
    if counts.min() < 10:
        thin = [int(c) for c, n in zip(present, counts) if n < 10]
        raise ValueError(f"Label at least 10 points of each class; class {thin} has fewer.")

    lidar._resolve_crs(header_of(path), options.crs)
    cloud = TiledCloud(path, lambda f, m: step(0.02 + 0.1 * f, m), should_cancel)
    if index.max() >= cloud.n:
        raise ValueError("The labels belong to a different cloud: some point numbers are past its end.")
    margin = max(3.0, 8.0 * cloud.spacing)
    ensure_memory(int(cloud.largest_tile * (1 + 4 * margin / cloud.tile_size) * 900), "Learning classes")
    intensity = cloud.manifest.get("intensity", {})
    intensity_scale = max(intensity.get("p99", 0.0), 1e-9) if intensity.get("varies") else None

    step(0.13, "Measuring the ground")
    ground = _ground_reference(path, cloud)
    tiles = sorted(cloud.tiles)

    # Measure the labelled points, in whichever tiles they fall.
    X = None
    for done, tile in enumerate(tiles):
        step(0.18 + 0.12 * done / len(tiles), "Measuring labelled points")
        d = cloud.load(tile, margin)
        local = np.nonzero(d["core"] & np.isin(d["index"], index))[0]
        if not local.size:
            continue
        features = Features(d, ground, intensity_scale)
        if X is None:
            X = np.zeros((index.size, len(features.names)), dtype=np.float32)
            names = features.names
        X[np.searchsorted(index, d["index"][local])] = features.measure(local)
    if X is None:
        raise ValueError("None of the labelled points are in this cloud.")

    step(0.32, f"Training on {index.size:,} labelled points")
    forest = Forest(trees=options.trees, max_depth=options.max_depth).fit(X, target)
    classes = forest.classes
    knows_noise = np.isin(classes, lidar.NOISE_CLASSES).any()

    updated = scratch(cloud.n, np.uint8)
    original_counts = np.zeros(256, dtype=np.int64)
    changed = 0
    confidence_hist = np.zeros(10, dtype=np.int64)
    candidates_x, candidates_y, candidates_c, candidates_p = [], [], [], []
    try:
        for done, tile in enumerate(tiles):
            step(0.4 + 0.5 * done / len(tiles), f"Classifying, tile {done + 1} of {len(tiles)}")
            d = cloud.load(tile, margin)
            core = d["core"]
            current = d["classification"]
            updated[d["index"][core]] = current[core]
            np.add.at(original_counts, current[core].clip(0, 255), 1)
            changeable = core.copy()
            if options.change_classes is not None:
                changeable &= np.isin(current, np.asarray(options.change_classes, dtype=np.int64))
            if not knows_noise:
                # Without noise examples the forest cannot know noise, so points
                # already labelled noise stay noise.
                changeable &= ~np.isin(current, lidar.NOISE_CLASSES)
            changeable &= ~np.isin(d["index"], index)
            local = np.nonzero(changeable)[0]
            if not local.size:
                continue
            proba = forest.predict_proba(Features(d, ground, intensity_scale).measure(local))
            best = np.argmax(proba, axis=1)
            confidence = proba[np.arange(local.size), best]
            sure = confidence >= options.min_confidence
            new = classes[best]
            changed += int((sure & (new != current[local])).sum())
            updated[d["index"][local[sure]]] = new[sure]
            confidence_hist += np.histogram(confidence, bins=10, range=(0, 1))[0]
            # Keep the least sure few of each tile, to choose spots from later.
            worst = np.argsort(confidence)[:200]
            candidates_x.append(d["x"][local[worst]])
            candidates_y.append(d["y"][local[worst]])
            candidates_c.append(confidence[worst])
            candidates_p.append(new[worst])

        # Labelled points keep their labels.
        before = np.asarray(updated[index])
        changed += int((before != target).sum())
        updated[index] = target

        step(0.9, "Writing point cloud")
        write_classified(path, options.output_path, updated, lambda f: step(0.9 + 0.09 * f, "Writing point cloud"))
        result_counts = np.zeros(256, dtype=np.int64)
        for lo in range(0, cloud.n, 5_000_000):
            result_counts += np.bincount(np.asarray(updated[lo:lo + 5_000_000]), minlength=256)
    finally:
        release(updated)

    spots = []
    if candidates_x:
        cx, cy = np.concatenate(candidates_x), np.concatenate(candidates_y)
        cc, cp = np.concatenate(candidates_c), np.concatenate(candidates_p)
        west, south, east, north = cloud.bounds
        apart = max(max(east - west, north - south) / 20, 5.0)
        for i in np.argsort(cc):
            if all(math.hypot(cx[i] - s["x"], cy[i] - s["y"]) > apart for s in spots):
                spots.append({"x": float(cx[i]), "y": float(cy[i]), "confidence": float(cc[i]),
                              "predicted": int(cp[i])})
            if len(spots) >= options.uncertain_spots:
                break

    # Out-of-bag quality, per class.
    truth, guess = forest.oob_target, forest.oob_predicted
    confusion = np.zeros((classes.size, classes.size), dtype=np.int64)
    np.add.at(confusion, (truth, guess), 1)
    per_class = []
    for i, code in enumerate(classes):
        tp = confusion[i, i]
        per_class.append({
            "class": int(code),
            "label": lidar.ASPRS_CLASSES.get(int(code), f"Class {int(code)}"),
            "labelled": int((target == code).sum()),
            "precision": float(tp / max(confusion[:, i].sum(), 1)),
            "recall": float(tp / max(confusion[i, :].sum(), 1)),
            "result": int(result_counts[int(code)]),
        })
    ranking = np.argsort(forest.importance)[::-1]
    step(1.0, "Complete")
    return {
        "outputPath": options.output_path,
        "pointsTotal": int(cloud.n),
        "pointsLabelled": int(index.size),
        "pointsChanged": int(changed),
        "accuracy": float((truth == guess).mean()) if truth.size else 0.0,
        "classes": per_class,
        "confusion": confusion.tolist(),
        "importance": [{"feature": names[i], "weight": float(forest.importance[i])} for i in ranking[:8]],
        "confidence": confidence_hist.tolist(),
        "uncertainSpots": spots,
    }
