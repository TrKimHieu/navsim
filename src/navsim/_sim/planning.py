"""Shortest paths for a robot on a map (distances to the goal, the time limit, SPL).

The planner works on a grid of 4 cm cells. A cell is "free space" when a disk of radius
(robot radius + clearance) centred there touches no wall: the robot fits there at ANY
heading. Paths through free space are therefore always drivable, at the cost of
possibly missing a narrow passage a long robot could only pass lengthwise.

Each cell is linked to 16 neighbours (4 straight, 4 diagonal, 8 knight moves) without
cutting corners, so grid distances are at most ~2.8% longer than straight-line ones.
"""

from __future__ import annotations

import json
from collections import OrderedDict

import cv2
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

NEIGHBOURS = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (1, -2), (2, -1)]   # + mirrored
TURN_SMOOTH = 0.05      # path simplification tolerance for the time limit (m)


def distance_transform_px(mask: np.ndarray) -> np.ndarray:
    """``ndimage.distance_transform_edt(mask)`` in pixels, ~5x faster with OpenCV. Squared
    distances are integers, so rounding OpenCV's float32 squares gives scipy's values
    bit for bit below ~1000 px (farther only the last bits may differ)."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.sqrt(np.rint(d.astype(np.float64) ** 2))


def _can_move(grid, dr, dc):
    """Mask of the cells (r, c) that can move to (r + dr, c + dc): both ends and the cells
    in between are free."""
    H, W = grid.shape

    def shifted(a, b):
        out = np.zeros_like(grid)
        rs, re = max(0, -a), min(H, H - a)
        cs, ce = max(0, -b), min(W, W - b)
        out[rs:re, cs:ce] = grid[rs + a:re + a, cs + b:ce + b]
        return out

    ok = grid & shifted(dr, dc)
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    if abs(dr) == 1 and abs(dc) == 1:
        ok &= shifted(dr, 0) & shifted(0, dc)
    elif abs(dr) == 1 and abs(dc) == 2:
        ok &= shifted(0, sc) & shifted(dr, sc)
    elif abs(dr) == 2 and abs(dc) == 1:
        ok &= shifted(sr, 0) & shifted(sr, dc)
    return ok


class Planner:
    """Free-space grid of one robot on one map, and shortest-path distances on it."""

    def __init__(self, svgmap, robot_radius, clearance=0.05, res=0.02, plan_res=0.04):
        mask, x0, y1, res = svgmap.raster(res)
        # distance between pixel centres; the real wall may be up to one pixel closer,
        # so one more pixel of margin keeps the test safe
        dist_m = distance_transform_px(mask) * res
        fine = dist_m >= robot_radius + clearance + res
        f = max(1, int(round(plan_res / res)))
        H, W = fine.shape[0] // f, fine.shape[1] // f
        self.grid = fine[:H * f, :W * f].reshape(H, f, W, f).all(axis=(1, 3))   # True = free
        self.x0, self.y1, self.res = x0, y1, res * f
        self.robot_radius, self.clearance = robot_radius, clearance
        self._index_cells()
        rows, cols, weights = [], [], []
        for dr, dc in NEIGHBOURS:
            r, c = np.nonzero(_can_move(self.grid, dr, dc))
            rows.append(self.node[r, c])
            cols.append(self.node[r + dr, c + dc])
            weights.append(np.full(len(r), self.res * np.hypot(dr, dc)))
        n = len(self.cells)
        self.graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows),
                                                           np.concatenate(cols))),
                                shape=(n, n)).tocsr()
        # the nearest free cell of every cell (used when the robot centre is not on one)
        _, self._nearest_free = ndimage.distance_transform_edt(~self.grid, return_indices=True)
        self._nearest_reachable: OrderedDict = OrderedDict()

    def _index_cells(self):
        """``cells``: (row, col) of every free cell; ``node``: its graph node (-1 if blocked)."""
        self.cells = np.argwhere(self.grid)
        self.node = np.full(self.grid.shape, -1, np.int64)
        self.node[self.cells[:, 0], self.cells[:, 1]] = np.arange(len(self.cells))

    # ------------------------------------------------------------ disk cache

    def to_arrays(self) -> dict:
        """Everything the constructor computes, as arrays (see ``from_arrays``)."""
        nearest = self._nearest_free
        if nearest.size and nearest.min() >= 0 and nearest.max() < 2 ** 16:
            nearest = nearest.astype(np.uint16)
        # edge weights take a few values only (one per neighbour offset): store their codes
        w_vals, w_code = np.unique(self.graph.data, return_inverse=True)
        # scalars go into one JSON member (floats round-trip exactly)
        info = {"grid_shape": list(self.grid.shape), "near_dtype": str(self._nearest_free.dtype),
                "frame": [float(v) for v in (self.x0, self.y1, self.res, self.robot_radius,
                                             self.clearance)],
                "w_vals": [float(v) for v in w_vals]}
        return {"grid": np.packbits(self.grid), "g_wcode": w_code.astype(np.uint8),
                "g_indices": self.graph.indices, "g_indptr": self.graph.indptr, "near": nearest,
                "planner": np.array(json.dumps(info))}

    @classmethod
    def from_arrays(cls, a) -> "Planner":
        """The Planner whose ``to_arrays()`` gave ``a``, identical to the original."""
        self = cls.__new__(cls)
        info = json.loads(str(a["planner"]))
        H, W = info["grid_shape"]
        self.grid = np.unpackbits(a["grid"], count=H * W).reshape(H, W).astype(bool)
        self.x0, self.y1, self.res, self.robot_radius, self.clearance = info["frame"]
        self._index_cells()
        n = len(self.cells)
        weights = np.array(info["w_vals"], float)[a["g_wcode"]]
        self.graph = csr_matrix((weights, a["g_indices"], a["g_indptr"]), shape=(n, n))
        self._nearest_free = a["near"].astype(info["near_dtype"])
        self._nearest_reachable = OrderedDict()
        return self

    # ------------------------------------------------------------ queries

    def cell_of(self, x, y):
        """(row, col) of the cell containing (x, y)."""
        return int((self.y1 - y) / self.res), int((x - self.x0) / self.res)

    def cell_center(self, row, col):
        return self.x0 + (col + 0.5) * self.res, self.y1 - (row + 0.5) * self.res

    def _clamp(self, row, col):
        H, W = self.grid.shape
        return min(max(row, 0), H - 1), min(max(col, 0), W - 1)

    def in_free_space(self, x, y) -> bool:
        row, col = self.cell_of(x, y)
        H, W = self.grid.shape
        return 0 <= row < H and 0 <= col < W and bool(self.grid[row, col])

    def nearest_cell(self, x, y):
        """The free cell nearest to (x, y)."""
        row, col = self._clamp(*self.cell_of(x, y))
        return int(self._nearest_free[0, row, col]), int(self._nearest_free[1, row, col])

    def field(self, x, y, predecessors=False):
        """Shortest-path distance (m) from (x, y) to every free cell, inf if unreachable.
        ``predecessors=True`` also returns Dijkstra's predecessor array (for ``path``)."""
        row, col = self.nearest_cell(x, y)
        res = dijkstra(self.graph, directed=False, indices=int(self.node[row, col]),
                       return_predecessors=predecessors)
        dist, pred = res if predecessors else (res, None)
        out = np.full(self.grid.shape, np.inf)
        out[self.cells[:, 0], self.cells[:, 1]] = dist
        return (out, pred) if predecessors else out

    def lookup(self, field, x, y) -> float:
        """Distance at (x, y): the value of the nearest free cell plus the straight hop
        to its centre.

        Free space is a bit more conservative than the real footprint, so a robot can stand
        where the nearest free cell is a small island that cannot reach the field's source
        (inf there). The nearest cell that can reach it is used instead, so the distance
        stays finite; everywhere else nothing changes. inf only if no cell reaches it."""
        row, col = self.nearest_cell(x, y)
        if not np.isfinite(field[row, col]):
            found = self._nearest_finite(field, x, y)
            if found is not None:
                row, col = found
        cx, cy = self.cell_center(row, col)
        return float(field[row, col] + np.hypot(x - cx, y - cy))

    def _nearest_finite(self, field, x, y):
        """The cell with a finite distance nearest to (x, y), or None. Its index image is
        built on first need and kept for the last 8 fields."""
        hit = self._nearest_reachable.get(id(field))
        if hit is None or hit[0] is not field:
            finite = np.isfinite(field)
            if not finite.any():
                return None
            _, idx = ndimage.distance_transform_edt(~finite, return_indices=True)
            hit = self._nearest_reachable[id(field)] = (field, idx)
            while len(self._nearest_reachable) > 8:
                self._nearest_reachable.popitem(last=False)
        row, col = self._clamp(*self.cell_of(x, y))
        return int(hit[1][0, row, col]), int(hit[1][1, row, col])

    def path(self, predecessors, x, y):
        """Shortest path from (x, y) to the field's source as a list of (x, y)."""
        row, col = self.nearest_cell(x, y)
        i = int(self.node[row, col])
        points = []
        while i >= 0:
            points.append(self.cell_center(*self.cells[i]))
            i = int(predecessors[i])
        return points


def simplify(points, tol):
    """Douglas-Peucker: drop the points closer than ``tol`` (m) to the line (removes the
    staircase of grid paths)."""
    P = np.asarray(points, float)
    if len(P) < 3:
        return P
    a, b = P[0], P[-1]
    d = b - a
    n = np.hypot(*d)
    dist = (np.abs(d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])) / n if n > 1e-12
            else np.hypot(*(P - a).T))
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return np.array([a, b])
    return np.vstack([simplify(P[:i + 1], tol)[:-1], simplify(P[i:], tol)])


def turning(points, theta0=None) -> float:
    """Total turning (rad) along a polyline, plus the turn from heading ``theta0`` onto it."""
    P = np.asarray(points, float)
    if len(P) < 2:
        return 0.0
    heading = np.arctan2(*np.diff(P, axis=0)[:, ::-1].T)
    dh = np.diff(heading)
    total = float(np.abs((dh + np.pi) % (2 * np.pi) - np.pi).sum())
    if theta0 is not None:
        total += abs((heading[0] - theta0 + np.pi) % (2 * np.pi) - np.pi)
    return total
