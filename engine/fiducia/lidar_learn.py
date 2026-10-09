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

def _ground_reference(x, y, z, classification) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """A ground height at any (x, y): from class 2 when there is enough of it,
    otherwise a coarse minimum surface, which is close enough for a feature."""
    ground = classification == 2
    use = ground if ground.sum() >= 50 else ~np.isin(classification, lidar.NOISE_CLASSES)
    cell = 1.0 if ground.sum() >= 50 else 5.0
    gx, gy, gz = x[use], y[use], z[use]
    bounds = lidar._snapped_bounds(x, y, cell)
    surface, _ = lidar._bin_points(gx, gy, gz, bounds, cell, "mean" if ground.sum() >= 50 else "minimum", 2.0)
    surface = lidar._fill_all(surface)
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
    """Measures any subset of a cloud's points against the whole cloud."""

    def __init__(self, las):
        from scipy.spatial import cKDTree

        self.x = np.asarray(las.x, dtype=np.float64)
        self.y = np.asarray(las.y, dtype=np.float64)
        self.z = np.asarray(las.z, dtype=np.float64)
        self.classification = np.asarray(las.classification).astype(np.int64)
        self.return_number = np.asarray(las.return_number).astype(np.float64)
        self.number_of_returns = np.maximum(np.asarray(las.number_of_returns).astype(np.float64), 1)
        self.intensity = None
        try:
            intensity = np.asarray(las.intensity).astype(np.float64)
            if intensity.max() > intensity.min():
                self.intensity = intensity / max(np.percentile(intensity, 99), 1e-9)
        except Exception:
            pass
        self.names = feature_names(self.intensity is not None)
        self.ground = _ground_reference(self.x, self.y, self.z, self.classification)
        # Neighbours are searched among real points only, in local coordinates.
        usable = ~np.isin(self.classification, lidar.NOISE_CLASSES)
        self.origin = (float(self.x.min()), float(self.y.min()))
        self.xyz = np.column_stack([self.x - self.origin[0], self.y - self.origin[1], self.z])
        self.searchable = np.nonzero(usable)[0]
        self.tree = cKDTree(self.xyz[self.searchable])

    def measure(self, index: np.ndarray) -> np.ndarray:
        out = np.empty((index.size, len(self.names)), dtype=np.float32)
        for start in range(0, index.size, CHUNK):
            part = index[start:start + CHUNK]
            out[start:start + part.size] = self._measure(part)
        return out

    def _measure(self, index: np.ndarray) -> np.ndarray:
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
    """
    import laspy

    def step(fraction, message):
        if progress:
            progress(fraction, message)
        if should_cancel and should_cancel():
            raise InterruptedError("Cancelled")

    index = np.array([int(k) for k in labels], dtype=np.int64)
    target = np.array([int(v) for v in labels.values()], dtype=np.int64)
    present, counts = np.unique(target, return_counts=True)
    if present.size < 2:
        raise ValueError("Label points of at least two classes before training.")
    if counts.min() < 10:
        thin = [int(c) for c, n in zip(present, counts) if n < 10]
        raise ValueError(f"Label at least 10 points of each class; class {thin} has fewer.")

    step(0.03, "Reading point cloud")
    las = laspy.read(path)
    lidar._resolve_crs(las.header, options.crs)
    if index.max() >= len(las.x):
        raise ValueError("The labels belong to a different cloud: some point numbers are past its end.")

    step(0.1, "Measuring neighbourhoods")
    features = Features(las)
    X = features.measure(index)

    step(0.25, f"Training on {index.size:,} labelled points")
    forest = Forest(trees=options.trees, max_depth=options.max_depth).fit(X, target)
    classes = forest.classes

    classification = features.classification
    changeable = np.ones(classification.size, dtype=bool)
    if options.change_classes is not None:
        changeable = np.isin(classification, np.asarray(options.change_classes, dtype=np.int64))
    if not np.isin(classes, lidar.NOISE_CLASSES).any():
        # Without noise examples the forest cannot know noise, so points
        # already labelled noise stay noise.
        changeable &= ~np.isin(classification, lidar.NOISE_CLASSES)
    changeable[index] = False
    todo = np.nonzero(changeable)[0]

    updated = classification.copy()
    updated[index] = target
    confidence_hist = np.zeros(10, dtype=np.int64)
    candidates_x, candidates_y, candidates_c, candidates_p = [], [], [], []
    for start in range(0, todo.size, CHUNK):
        step(0.35 + 0.55 * start / max(todo.size, 1), f"Classifying {start:,} of {todo.size:,} points")
        part = todo[start:start + CHUNK]
        proba = forest.predict_proba(features.measure(part))
        best = np.argmax(proba, axis=1)
        confidence = proba[np.arange(part.size), best]
        sure = confidence >= options.min_confidence
        updated[part[sure]] = classes[best[sure]]
        confidence_hist += np.histogram(confidence, bins=10, range=(0, 1))[0]
        # Keep the least sure few of each pass, to choose spots from later.
        worst = np.argsort(confidence)[:200]
        candidates_x.append(features.x[part[worst]])
        candidates_y.append(features.y[part[worst]])
        candidates_c.append(confidence[worst])
        candidates_p.append(classes[best[worst]])

    step(0.92, "Writing point cloud")
    las.classification = updated.astype(np.asarray(las.classification).dtype)
    Path(options.output_path).parent.mkdir(parents=True, exist_ok=True)
    las.write(options.output_path)

    spots = []
    if candidates_x:
        cx, cy = np.concatenate(candidates_x), np.concatenate(candidates_y)
        cc, cp = np.concatenate(candidates_c), np.concatenate(candidates_p)
        span = max(np.ptp(features.x), np.ptp(features.y))
        apart = max(span / 20, 5.0)
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
            "result": int((updated == code).sum()),
        })
    order = np.argsort(forest.importance)[::-1]
    step(1.0, "Complete")
    return {
        "outputPath": options.output_path,
        "pointsTotal": int(classification.size),
        "pointsLabelled": int(index.size),
        "pointsChanged": int((updated != classification).sum()),
        "accuracy": float((truth == guess).mean()) if truth.size else 0.0,
        "classes": per_class,
        "confusion": confusion.tolist(),
        "importance": [{"feature": features.names[i], "weight": float(forest.importance[i])}
                       for i in order[:8]],
        "confidence": confidence_hist.tolist(),
        "uncertainSpots": spots,
    }
