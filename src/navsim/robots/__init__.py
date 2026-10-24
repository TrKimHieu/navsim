"""Robots: real models with their size, speed limits and LiDAR.

    navsim.list_robots()                      # ["agilex_limo", "clearpath_jackal", ...]
    robot = navsim.load_robot("turtlebot4")
    robot.footprint, robot.v_max, robot.lidar # body polygon (m), top speed (m/s), LiDAR specs

Each robot is a JSON file (schema 1.0) in ``presets/``. Every number in it cites a source:
``official`` (with a URL) or ``estimated`` (with the method used).

The footprint is the robot's outline seen from above, a polygon in the robot frame
(x forward, y left, origin at the centre of rotation), counter-clockwise. A circle is
replaced by a 16-sided polygon around it (never smaller than the robot).

``height_class`` says which maps a robot uses: obstacles are cut at the robot's height
(h36: up to 0.36 m, h63: 0.626 m, h145: 1.45 m), and robots of one class share the maps.

``load(id)`` looks in: the ``extra`` folders (e.g. a dataset's copy of its robots), then the
folders in $NAVSIM_ROBOTS, then the presets of the package. It also takes a .json path.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
DRIVES = ("differential", "omnidirectional")
REQUIRED = ("schema_version", "id", "name", "footprint", "height", "height_class",
            "drive", "limits", "lidar", "sources")
PRESETS = Path(__file__).resolve().parent / "presets"


@dataclass
class Robot:
    id: str
    name: str
    footprint: np.ndarray            # (M, 2) polygon in the robot frame (m)
    height: float                    # m
    height_class: str                # map set: "h36", "h63" or "h145"
    drive: str                       # "differential" or "omnidirectional"
    v_max: float                     # forward speed limit (m/s)
    v_min: float                     # reverse speed limit (m/s, <= 0)
    vy_max: float                    # sideways speed limit (m/s, omnidirectional only)
    w_max: float                     # turning speed limit (rad/s)
    lidar: dict                      # range_min, range_max (m), fov (rad), beams
    raw: dict = field(repr=False)    # the JSON file

    @property
    def omni(self) -> bool:
        """Whether the robot can also move sideways (mecanum wheels)."""
        return self.drive == "omnidirectional"

    @cached_property
    def r_circ(self) -> float:
        """Radius of the smallest circle around the centre of rotation that holds the robot."""
        return float(np.linalg.norm(self.footprint, axis=1).max())

    @property
    def action_dim(self) -> int:
        """2 actions (v, w), or 3 (v, vy, w) for an omnidirectional robot."""
        return 3 if self.omni else 2

    @classmethod
    def from_dict(cls, raw: dict) -> "Robot":
        validate(raw)
        limits = raw["limits"]
        return cls(id=raw["id"], name=raw["name"], footprint=footprint_polygon(raw["footprint"]),
                   height=float(raw["height"]), height_class=raw["height_class"],
                   drive=raw["drive"], v_max=float(limits["v_max"]), v_min=float(limits["v_min"]),
                   vy_max=float(limits["vy_max"]), w_max=float(limits["w_max"]),
                   lidar=dict(raw["lidar"]), raw=raw)


def load(robot_id: str | Path, extra=None) -> Robot:
    """A robot by id (see the search order above) or from the path of a JSON file."""
    path = Path(robot_id)
    if not (path.suffix == ".json" and path.exists()):
        path = _find(str(robot_id), extra)
    return Robot.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_robots(extra=None) -> list[str]:
    return sorted({p.stem for d in _search_dirs(extra) if d.exists() for p in d.glob("*.json")})


def _search_dirs(extra=None) -> list[Path]:
    dirs = [Path(d) for d in ([extra] if isinstance(extra, (str, Path)) else (extra or []))]
    dirs += [Path(d).expanduser() for d in os.environ.get("NAVSIM_ROBOTS", "").split(os.pathsep) if d]
    return dirs + [PRESETS]


def _find(robot_id: str, extra=None) -> Path:
    for d in _search_dirs(extra):
        if (d / f"{robot_id}.json").exists():
            return d / f"{robot_id}.json"
    searched = [str(d) for d in _search_dirs(extra)]
    raise FileNotFoundError(f"no robot preset '{robot_id}' (searched {searched}); "
                            f"available: {list_robots(extra)}")


def footprint_polygon(spec: dict) -> np.ndarray:
    """The footprint of a JSON file as a counter-clockwise (M, 2) polygon."""
    kind = spec["type"]
    if kind == "rectangle":
        half_length, half_width = spec["length"] / 2, spec["width"] / 2
        return np.array([[half_length, -half_width], [half_length, half_width],
                         [-half_length, half_width], [-half_length, -half_width]], float)
    if kind == "circle":
        sides = 16
        radius = spec["radius"] / math.cos(math.pi / sides)     # polygon AROUND the circle
        angles = (np.arange(sides) + 0.5) * 2 * math.pi / sides
        return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    if kind == "polygon":
        points = np.asarray(spec["points"], float)
        return points[::-1] if _signed_area(points) < 0 else points
    raise ValueError(f"unsupported footprint type: {kind}")


def _signed_area(P: np.ndarray) -> float:
    """Polygon area, negative when the points go clockwise."""
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def validate(raw: dict) -> None:
    """Raise ValueError when a robot file is incomplete, or when a number has no source
    (a ``sources`` key equal to its dotted path, e.g. ``limits.v_max``, or to a prefix)."""
    rid = raw.get("id")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ValueError(f"{rid}: missing {missing}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{rid}: schema_version {raw['schema_version']}")
    if raw["drive"] not in DRIVES:
        raise ValueError(f"{rid}: drive must be one of {DRIVES}")
    for group, keys in (("limits", ("v_max", "v_min", "vy_max", "w_max")),
                        ("lidar", ("range_min", "range_max", "fov", "beams"))):
        for k in keys:
            if k not in raw[group]:
                raise ValueError(f"{rid}: {group}.{k} missing")
    for source in raw["sources"].values():
        if source.get("kind") == "official" and not source.get("url"):
            raise ValueError(f"{rid}: official source without url")
        if source.get("kind") == "estimated" and not source.get("method"):
            raise ValueError(f"{rid}: estimated source without method")
    cited = list(raw["sources"])
    body = {k: raw[k] for k in ("footprint", "height", "limits", "lidar")}
    for name in _number_paths(body):
        if not any(name == k or name.startswith(k + ".") for k in cited):
            raise ValueError(f"{rid}: parameter {name} has no source")


def _number_paths(d: dict, prefix: str = ""):
    """Dotted paths of every number in a nested dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _number_paths(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            yield f"{prefix}{k}"
