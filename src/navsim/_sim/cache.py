"""Disk cache of loaded maps: a map loads in a few ms instead of ~100 ms.

What is stored is exactly what the code computes, so results are identical with or
without the cache. Files:

    <root>/v3/<dataset>/maps/<height_class>/<map_id>-<key>.npz   map outline, raster, edge distance
    <root>/v3/<dataset>/<robot>/<map_id>-<key>.npz               task file, path planner

The keys hash the sha256 of the map and task files (from the dataset manifest, so nothing
is re-read to check them) and every parameter the result depends on: changed data gets
new files. Old files are only removed by ``clear()`` (``navsim cache --clear``).

    NAVSIM_CACHE   cache folder, or "off". Default: %LOCALAPPDATA%\\navsim\\cache on
                   Windows, else $XDG_CACHE_HOME/navsim or ~/.cache/navsim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from ..data.dataset import SCHEMA_VERSION
from .planning import TURN_SMOOTH, Planner
from .svgmap import SvgMap

log = logging.getLogger(__name__)

FORMAT = 3
RASTER = (0.02, 0.2)                          # collider raster: resolution, padding (m)
PLANNER = (0.02, 0.04)                        # planner: raster resolution, cell size (m)
OFF = ("off", "0", "false", "none", "")


def cache_root() -> Path | None:
    value = os.environ.get("NAVSIM_CACHE")
    if value is not None:
        return None if value.strip().lower() in OFF else Path(value).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "navsim" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "navsim"


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def _write(path: Path, arrays: dict) -> None:
    """Write an .npz file atomically (temporary file, then rename); failures are only logged."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    except OSError as e:
        log.debug("navsim cache: cannot write %s: %s", path, e)
        tmp.unlink(missing_ok=True)


def _read(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except FileNotFoundError:
        return None
    except Exception as e:                    # corrupt or partial file: rebuild it
        log.debug("navsim cache: unreadable %s: %s", path, e)
        return None


def _pack_rings(layers: dict) -> tuple[dict, dict]:
    """Polygon layers -> (arrays, JSON-able info)."""
    names = list(layers)
    rings = [r for n in names for r in layers[n]]
    return ({"ring_lens": np.array([len(r) for r in rings], np.int64),
             "ring_xy": (np.concatenate(rings) if rings else np.zeros((0, 2))).astype(float)},
            {"layer_names": names, "layer_counts": [len(layers[n]) for n in names]})


def _unpack_rings(arrays: dict, info: dict) -> dict:
    lens = arrays["ring_lens"]
    rings = np.split(arrays["ring_xy"], np.cumsum(lens)[:-1]) if len(lens) else []
    out, i = {}, 0
    for name, count in zip(info["layer_names"], info["layer_counts"]):
        out[name] = list(rings[i:i + count])
        i += count
    return out


class DatasetCache:
    """The cache files of one dataset. ``enabled`` is False when there is no cache folder
    or the manifest has no file hashes."""

    def __init__(self, dataset, root: Path | None = None):
        self.dataset = dataset
        self.root = root if root is not None else cache_root()
        self.hashes = dataset.manifest.get("files") or {}
        self.enabled = self.root is not None and bool(self.hashes)

    def paths(self, robot: str, map_id: str, robot_radius: float, clearance) -> tuple | None:
        """(map file, robot file), or None if this map cannot be cached."""
        if not self.enabled:
            return None
        height_class = self.dataset.manifest["robots"][robot]["height_class"]
        svg_sha = self.hashes.get(f"maps/{height_class}/{map_id}.svg")
        task_sha = self.hashes.get(f"tasks/{robot}/{map_id}.json")
        if not svg_sha or not task_sha:
            return None
        base = self.root / f"v{FORMAT}" / self.dataset.name
        map_key = _key("map", FORMAT, svg_sha, RASTER)
        robot_key = _key("robot", FORMAT, map_key, task_sha, repr(float(robot_radius)),
                         repr(clearance), PLANNER, TURN_SMOOTH)
        return (base / "maps" / height_class / f"{map_id}-{map_key}.npz",
                base / robot / f"{map_id}-{robot_key}.npz")

    def load(self, paths) -> dict | None:
        """Everything needed to rebuild the map, or None if it is not cached."""
        map_file, robot_file = _read(paths[0]), _read(paths[1])
        if map_file is None or robot_file is None:
            return None
        try:
            map_info = json.loads(str(map_file["info"]))
            robot_info = json.loads(str(robot_file["info"]))
            layers = _unpack_rings(map_file, map_info)
            svgmap = SvgMap(layers.pop("free"), map_info["meta"], layers)
            H, W = map_info["mask_shape"]
            mask = np.unpackbits(map_file["mask"], count=H * W).reshape(H, W).astype(bool)
            x0, y1, res = map_info["frame"]
            svgmap._raster[(round(RASTER[0], 6), round(RASTER[1], 6))] = (mask, x0, y1, res)
            header = robot_info["header"]
            if header.get("schema_version") != SCHEMA_VERSION:
                return None
            planner = Planner.from_arrays(robot_file)
            turns = dict(enumerate(robot_info["turns"])) if robot_info.get("turns") is not None else {}
        except (KeyError, ValueError) as e:
            log.debug("navsim cache: bad entry %s: %s", paths[1], e)
            return None
        return {"svgmap": svgmap, "edge_dist": map_file["edge_dist"], "header": header,
                "planner": planner, "turns": turns}

    def save(self, paths, svgmap, collider, header: dict, planner, turns=None) -> None:
        """Write the map file (if missing) and the robot file. Small values go into one
        JSON member per file (JSON floats round-trip exactly)."""
        if not paths[0].exists():
            mask, x0, y1, res = svgmap.raster(*RASTER)
            arrays, info = _pack_rings({"free": svgmap.rings, **svgmap.viz})
            info.update(meta=svgmap.meta, mask_shape=list(mask.shape),
                        frame=[float(x0), float(y1), float(res)])
            _write(paths[0], {**arrays, "mask": np.packbits(mask), "edge_dist": collider.edge_dist,
                              "info": np.array(json.dumps(info, ensure_ascii=False))})
        info = {"header": header, "turns": None if turns is None else [float(t) for t in turns]}
        _write(paths[1], {**planner.to_arrays(),
                          "info": np.array(json.dumps(info, ensure_ascii=False))})


# ------------------------------------------------------------ navsim cache [--clear]

def entries(root: Path | None = None) -> list[dict]:
    """One row per cached dataset: files, size, folder."""
    root = root if root is not None else cache_root()
    base = root / f"v{FORMAT}" if root else None
    out = []
    if base and base.exists():
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            files = list(d.rglob("*.np[yz]"))
            out.append({"dataset": d.name, "files": len(files),
                        "mb": sum(f.stat().st_size for f in files) / 2 ** 20, "path": str(d)})
    return out


def clear(dataset: str | None = None, root: Path | None = None) -> list[Path]:
    """Remove the cache of one dataset, or everything, in every format version."""
    root = root if root is not None else cache_root()
    if root is None or not root.exists():
        return []
    targets = [v / dataset if dataset else v for v in sorted(root.glob("v[0-9]*"))]
    targets = [t for t in targets if t.is_dir()]
    for t in targets:
        shutil.rmtree(t)
    return targets
