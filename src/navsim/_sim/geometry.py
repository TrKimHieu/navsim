"""Collision checks and LiDAR on a map, in metres.

The map is the ``free`` region of an SvgMap: the area where a robot may be. A robot pose
(x, y, theta) is valid when the robot's footprint polygon crosses no edge of ``free`` and its
centre is inside ``free`` (if no edge is crossed, the footprint is entirely on one side, so
one point decides). Touching an edge counts as a collision.

Two speed-ups never change a result: a distance image to the map edges answers most
``pose_valid`` calls at once, and :mod:`navsim._sim.kernels` runs the exact tests as
compiled numba loops when numba is installed.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import kernels

EPS = 1e-12
GRID_CELL = 0.25       # cell size of the segment grid of the numba kernels (m)
EDGE_MARGIN_PX = 2.0   # error bound of the edge distance image: rounding of points and lines (px)


def place(polygon: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    """A polygon given in the robot frame, moved to the map frame at pose (x, y, theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return polygon @ np.array([[c, s], [-s, c]]) + (x, y)


def polygon_segments(polygon: np.ndarray) -> np.ndarray:
    """The M edges of a closed polygon (M, 2) as rows ``[x1, y1, x2, y2]``."""
    p = np.asarray(polygon, float)
    return np.concatenate([p, np.roll(p, -1, axis=0)], axis=1)


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """(N,) bool: which points are inside the polygon (even-odd rule, any simple polygon)."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x1, y1 = polygon[:, 0][None], polygon[:, 1][None]
    x2, y2 = np.roll(polygon[:, 0], -1)[None], np.roll(polygon[:, 1], -1)[None]
    crosses = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return ((crosses & (x < xi)).sum(axis=1) % 2).astype(bool)


def segments_hit_polygon(segs: np.ndarray, polygon: np.ndarray) -> bool:
    """Whether any segment (K, 4) crosses or touches the polygon (M, 2)."""
    if not len(segs):
        return False
    a, b = segs[:, :2], segs[:, 2:]
    # a segment that crosses no edge is entirely inside or entirely outside:
    # one endpoint inside is enough
    if points_in_polygon(a, polygon).any():
        return True
    edges = np.roll(polygon, -1, axis=0) - polygon
    q1, q2 = polygon[None], (polygon + edges)[None]            # (1, M, 2)
    p1, p2 = a[:, None], b[:, None]                            # (K, 1, 2)
    r = p2 - p1
    s = q2 - q1
    d1 = _cross(s[..., 0], s[..., 1], (p1 - q1)[..., 0], (p1 - q1)[..., 1])
    d2 = _cross(s[..., 0], s[..., 1], (p2 - q1)[..., 0], (p2 - q1)[..., 1])
    d3 = _cross(r[..., 0], r[..., 1], (q1 - p1)[..., 0], (q1 - p1)[..., 1])
    d4 = _cross(r[..., 0], r[..., 1], (q2 - p1)[..., 0], (q2 - p1)[..., 1])
    # compare SIGNS, not products with a threshold: footprints from URDF have millimetre
    # edges, so these values are ~1e-7 and their products (~1e-14) would fall under a
    # threshold like 1e-12 and report crossings that do not exist (the robot freezes)
    tol = 1e-15
    sd1, sd2, sd3, sd4 = (np.where(np.abs(d) <= tol, 0, np.sign(d)) for d in (d1, d2, d3, d4))
    return bool(((sd1 * sd2 <= 0) & (sd3 * sd4 <= 0)).any())


def _segments_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(K, 4), (L, 4) -> (K, L) bool: the segments cross, touch, or overlap on one line.
    Two segments on the same line that do not overlap do not count."""
    p, p2 = a[:, None, :2], a[:, None, 2:]
    q, q2 = b[None, :, :2], b[None, :, 2:]
    r, s = p2 - p, q2 - q
    qp = q - p
    den = _cross(r[..., 0], r[..., 1], s[..., 0], s[..., 1])
    parallel = np.abs(den) <= 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qp[..., 0], qp[..., 1], s[..., 0], s[..., 1]) / den
        u = _cross(qp[..., 0], qp[..., 1], r[..., 0], r[..., 1]) / den
    hit = (~parallel) & (t >= -1e-9) & (t <= 1 + 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
    d1 = _cross(r[..., 0], r[..., 1], qp[..., 0], qp[..., 1])
    d2 = _cross(r[..., 0], r[..., 1], (q2 - p)[..., 0], (q2 - p)[..., 1])
    collinear = parallel & (np.abs(d1) <= 1e-9) & (np.abs(d2) <= 1e-9)
    rr = np.maximum((r * r).sum(-1), EPS)
    t0 = ((q - p) * r).sum(-1) / rr
    t1 = ((q2 - p) * r).sum(-1) / rr
    lo, hi = np.minimum(t0, t1), np.maximum(t0, t1)
    overlap = collinear & (hi >= -1e-9) & (lo <= 1 + 1e-9)
    return hit | overlap


def polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two polygons cross, touch, or one contains the other."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if _segments_cross(polygon_segments(a), polygon_segments(b)).any():
        return True
    return bool(points_in_polygon(a, b).any() or points_in_polygon(b, a).any())


def segments_raycast(ox: float, oy: float, angles: np.ndarray, segs: np.ndarray,
                     r_max: float) -> np.ndarray:
    """Distance from (ox, oy) along each angle to the nearest segment (K, 4), ``r_max``
    when a ray hits nothing. For a few segments (e.g. another robot's footprint)."""
    angles = np.asarray(angles, float)
    out = np.full(len(angles), r_max, float)
    S = np.asarray(segs, float)
    if not len(S):
        return out
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]        # (B, 1)
    ax, ay = S[None, :, 0] - ox, S[None, :, 1] - oy                  # (1, K)
    ex, ey = S[None, :, 2] - S[None, :, 0], S[None, :, 3] - S[None, :, 1]
    den = dx * ey - dy * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ax * ey - ay * ex) / den
        u = (ax * dy - ay * dx) / den
    ok = (np.abs(den) > EPS) & (t >= 0) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf).min(axis=1)
    return np.minimum(out, t)


class SegmentIndex:
    """The map edges sorted into square cells, to find the edges near a box quickly.
    Built on the first query (the numba kernels use their own grid)."""

    def __init__(self, segs, cell: float = 0.25):
        self.segs = np.asarray(segs, float)
        self.cell = cell
        self._buckets = None

    def _build(self):
        lo = np.minimum(self.segs[:, :2], self.segs[:, 2:])
        hi = np.maximum(self.segs[:, :2], self.segs[:, 2:])
        self.origin = lo.min(axis=0) if len(lo) else np.zeros(2)
        first = np.floor((lo - self.origin) / self.cell).astype(int)
        last = np.floor((hi - self.origin) / self.cell).astype(int)
        self.shape = (last.max(axis=0) + 1) if len(last) else np.ones(2, int)
        buckets: dict = {}
        for k, (a, b) in enumerate(zip(first, last)):
            for ix in range(a[0], b[0] + 1):
                for iy in range(a[1], b[1] + 1):
                    buckets.setdefault((ix, iy), []).append(k)
        self._buckets = {key: np.array(v, np.int64) for key, v in buckets.items()}

    def query(self, xmin, ymin, xmax, ymax) -> np.ndarray:
        """Indices of the segments whose cells touch the box."""
        if self._buckets is None:
            self._build()
        c = self.cell
        ix0, iy0 = np.floor((np.array([xmin, ymin]) - self.origin) / c).astype(int)
        ix1, iy1 = np.floor((np.array([xmax, ymax]) - self.origin) / c).astype(int)
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, self.shape[0] - 1), min(iy1, self.shape[1] - 1)
        parts = [self._buckets[(ix, iy)] for ix in range(ix0, ix1 + 1)
                 for iy in range(iy0, iy1 + 1) if (ix, iy) in self._buckets]
        if not parts:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate(parts))


class Collider:
    """Collision and LiDAR queries on one map.

    ``accel``: use the numba kernels (default: when numba is installed). Both backends give
    identical results; False forces the numpy code (tests, debugging).
    """

    def __init__(self, svgmap, cell: float = 0.25, res: float = 0.02, accel: bool | None = None,
                 edge_dist: np.ndarray | None = None):
        self.map = svgmap
        self.index = SegmentIndex(svgmap.segments, cell)
        self.mask, self.x0, self.y1, self.res = svgmap.raster(res)
        self.accel = kernels.ENABLED if accel is None else (accel and kernels.ENABLED)
        self.edge_dist = self._edge_distance() if edge_dist is None else edge_dist
        self._radius_of: dict = {}           # id(polygon) -> (polygon, circumradius)
        self._grid = None

    def _edge_distance(self) -> np.ndarray:
        """For every pixel: distance (whole pixels, uint8, at most 255) to the nearest pixel
        an edge of ``free`` passes through. The true distance from a point in that pixel to
        the nearest edge is at least (value - EDGE_MARGIN_PX) x res."""
        H, W = self.mask.shape
        img = np.full((H, W), 255, np.uint8)
        S = self.index.segs
        if len(S):
            shift = 8
            px = (S[:, [0, 2]] - self.x0) / self.res - 0.5          # pixel-centre coordinates
            py = (self.y1 - S[:, [1, 3]]) / self.res - 0.5
            pts = np.ascontiguousarray(np.round(np.stack([px, py], -1) * (1 << shift)), np.int32)
            cv2.polylines(img, list(pts), False, 0, 1, cv2.LINE_8, shift)
        d = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return np.minimum(np.floor(d), 255).astype(np.uint8)

    @property
    def grid(self):
        """The segment grid of the numba kernels (built on first use)."""
        if self._grid is None:
            S = self.index.segs
            if len(S):
                gx0 = float(min(S[:, 0].min(), S[:, 2].min())) - GRID_CELL
                gy0 = float(min(S[:, 1].min(), S[:, 3].min())) - GRID_CELL
                nx = int((max(S[:, 0].max(), S[:, 2].max()) - gx0) // GRID_CELL) + 2
                ny = int((max(S[:, 1].max(), S[:, 3].max()) - gy0) // GRID_CELL) + 2
            else:
                gx0 = gy0 = 0.0
                nx = ny = 1
            start, items = kernels.build_grid(S, gx0, gy0, GRID_CELL, nx, ny)
            # the last three arrays are scratch space of raycast_grid
            self._grid = (start, items, gx0, gy0, GRID_CELL, nx, ny,
                          np.zeros(len(S), np.int64), np.zeros((len(S), 6), np.int64),
                          np.zeros(1, np.int64))
        return self._grid

    def _clear_of_edges(self, x: float, y: float, r: float) -> bool:
        """True when no edge of ``free`` can be within r of (x, y) (never wrongly True)."""
        col = (x - self.x0) / self.res
        row = (self.y1 - y) / self.res
        H, W = self.edge_dist.shape
        if not (0.0 <= row < H and 0.0 <= col < W):
            return False
        return (float(self.edge_dist[int(row), int(col)]) - EDGE_MARGIN_PX) * self.res > r

    def inside(self, x: float, y: float) -> bool:
        """Whether a point is inside ``free`` (raster lookup: only used once the footprint
        crosses no edge, so rounding at the boundary does not matter)."""
        col = int((x - self.x0) / self.res)
        row = int((self.y1 - y) / self.res)
        H, W = self.mask.shape
        return 0 <= row < H and 0 <= col < W and bool(self.mask[row, col])

    def pose_valid(self, polygon: np.ndarray, x: float, y: float, theta: float) -> bool:
        """Whether a robot with this footprint fits at (x, y, theta)."""
        # two shortcuts with the same answer as the exact test: centre outside free ->
        # invalid; every edge farther than the footprint's radius -> nothing to cross
        if not self.inside(x, y):
            return False
        cached = self._radius_of.get(id(polygon))
        if cached is None or cached[0] is not polygon:
            cached = self._radius_of[id(polygon)] = (
                polygon, float(np.linalg.norm(polygon, axis=1).max()))
        if self._clear_of_edges(x, y, cached[1]):
            return True
        P = place(polygon, x, y, theta)
        lo, hi = P.min(axis=0), P.max(axis=0)
        if self.accel:
            return not kernels.seg_hit_poly_grid(self.index.segs, *self.grid[:7], P,
                                                 lo[0], lo[1], hi[0], hi[1])
        near = self.index.query(lo[0], lo[1], hi[0], hi[1])
        return not segments_hit_polygon(self.index.segs[near], P)

    def wall_distance(self, x: float, y: float) -> float:
        """Exact distance (m) from (x, y) to the nearest edge of ``free``."""
        S = self.index.segs
        if not len(S):
            return float("inf")
        if self.accel:
            return float(kernels.nearest_grid(S, *self.grid[:7], float(x), float(y)))
        a = S[:, :2] - (x, y)
        e = S[:, 2:] - S[:, :2]
        L2 = np.maximum((e * e).sum(axis=1), EPS)
        u = np.clip(-(a * e).sum(axis=1) / L2, 0.0, 1.0)
        return float(np.hypot(a[:, 0] + u * e[:, 0], a[:, 1] + u * e[:, 1]).min())

    def raycast_uniform(self, ox, oy, a0, da, n, r_max, full=False) -> np.ndarray:
        """Distance to the nearest edge along n rays at angles a0 + k x da (``r_max`` when a
        ray hits nothing). ``full``: the rays cover 360 degrees and indices wrap around.

        Each edge is only tested against the rays inside the angle it covers seen from
        (ox, oy): a few thousand (edge, ray) pairs instead of n x all edges."""
        S = self.index.segs
        out = np.full(n, r_max, float)
        if not len(S):
            return out
        if self.accel:
            ang = a0 + np.arange(n) * da              # same values as the numpy code below
            kernels.raycast_grid(S, *self.grid[:7], float(ox), float(oy), float(a0), float(da),
                                 int(n), float(r_max), bool(full), np.cos(ang), np.sin(ang),
                                 out, *self.grid[7:])
            return out
        x1, y1, x2, y2 = S[:, 0] - ox, S[:, 1] - oy, S[:, 2] - ox, S[:, 3] - oy
        ex, ey = x2 - x1, y2 - y1
        L2 = np.maximum(ex * ex + ey * ey, EPS)
        u = np.clip(-(x1 * ex + y1 * ey) / L2, 0, 1)
        near = np.hypot(x1 + u * ex, y1 + u * ey) <= r_max      # drop edges out of range
        x1, y1, x2, y2, ex, ey = x1[near], y1[near], x2[near], y2[near], ex[near], ey[near]
        two_pi = 2 * np.pi
        r1 = (np.arctan2(y1, x1) - a0) % two_pi
        d = (np.arctan2(y2, x2) - a0 - r1 + np.pi) % two_pi - np.pi   # the short arc
        lo = r1 + np.minimum(d, 0)
        hi = lo + np.abs(d)
        ids = np.arange(len(lo))
        if not full:
            # lo is in (-pi, 2pi): an arc may straddle 0 or 2pi -> also try it shifted by
            # +-2pi (with 360 degrees the indices wrap modulo n instead)
            ids = np.concatenate([ids, ids, ids])
            lo = np.concatenate([lo, lo - two_pi, lo + two_pi])
            hi = np.concatenate([hi, hi - two_pi, hi + two_pi])
        k_lo = np.ceil(lo / da - 1e-9).astype(np.int64)
        k_hi = np.floor(hi / da + 1e-9).astype(np.int64)
        count = np.maximum(k_hi - k_lo + 1, 0)
        if not count.sum():
            return out
        seg = np.repeat(ids, count)
        k = np.repeat(k_lo, count) + (np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count))
        if full:
            k %= n
        else:
            ok = (k >= 0) & (k < n)
            seg, k = seg[ok], k[ok]
        ang = a0 + k * da
        dx, dy = np.cos(ang), np.sin(ang)
        ax, ay, sx, sy = x1[seg], y1[seg], ex[seg], ey[seg]
        den = dx * sy - dy * sx
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * sy - ay * sx) / den
            uu = (ax * dy - ay * dx) / den
        hit = (np.abs(den) > EPS) & (t >= 0) & (uu >= -1e-9) & (uu <= 1 + 1e-9)
        np.minimum.at(out, k[hit], t[hit])
        return out
