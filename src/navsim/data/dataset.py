"""Datasets on disk: finding them, reading them, checking them.

    ds = navsim.load_dataset("demo-svg")              # by name, or the path of a folder
    ds.robots                                         # ["clearpath_jackal", "turtlebot4"]
    ds.maps("turtlebot4", split="train")              # map ids
    ds.tasks("turtlebot4", map_id)                    # start / goal pairs of one map

A dataset name is looked for, in this order:

    1. the folders listed in $NAVSIM_DATASETS (separated by ":" on Linux, ";" on Windows)
    2. ~/.navsim/datasets (or $NAVSIM_HOME/datasets): where ``navsim.setup`` downloads to
    3. Kaggle inputs: /kaggle/input/<anything>/<name> and /kaggle/input/<anything>/*/<name>
    4. the demos inside the package: demo-svg, demo-grid

Folder layout (schema 2.0, the format hm3denv builds):

    manifest.json                     name, env ("svg" | "grid"), robots, splits, sha256 of every file
    tasks/<robot>/<map_id>.json       {"split", "success_radius", "tasks": [{"id", "start", "goal",
                                       "geodesic_m", "euclidean_m", "labels"}, ...]}
    maps/<height_class>/<map_id>.svg  continuous maps (svg datasets)
    grids/<map_id>.npz + .json        grids (grid datasets)
    robots/<robot>.json               the robots as they were when the dataset was built

Map ids look like ``isb-S001_s0``: storey 0 of building (scene) S001. The splits are by
building, so the test maps are buildings never seen in training.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..robots import Robot, load as load_robot

SCHEMA_VERSION = "2.0"
SPLITS = ("train", "val", "test")
DEMO_DATASETS = Path(__file__).resolve().parent / "demo"
KAGGLE_INPUT = Path("/kaggle/input")


# ------------------------------------------------------------------ where datasets are

def data_home() -> Path:
    """$NAVSIM_HOME, else ~/.navsim."""
    value = os.environ.get("NAVSIM_HOME")
    return Path(value).expanduser().resolve() if value else Path.home() / ".navsim"


def download_dir() -> Path:
    return data_home() / "datasets"


def kaggle_dirs(root: Path | None = None) -> list[Path]:
    """Folders under the Kaggle inputs that may hold datasets (two levels deep)."""
    root = KAGGLE_INPUT if root is None else Path(root)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        out.append(d)
        out += sorted(p for p in d.iterdir() if p.is_dir() and not (p / "manifest.json").exists())
    return out


def search_dirs(extra: str | os.PathLike | None = None) -> list[Path]:
    """The folders searched for datasets, in priority order (``extra`` first)."""
    dirs = [Path(extra).expanduser()] if extra else []
    dirs += [Path(p).expanduser() for p in os.environ.get("NAVSIM_DATASETS", "").split(os.pathsep) if p]
    dirs += [download_dir(), *kaggle_dirs(), DEMO_DATASETS]
    out: list[Path] = []
    for d in dirs:
        d = d.resolve()
        if d not in out:
            out.append(d)
    return out


def find_dataset(name_or_path: str | os.PathLike, search=None) -> Path | None:
    """The folder of a dataset given by path or name, or None if it is nowhere."""
    path = Path(name_or_path).expanduser()
    if (path / "manifest.json").exists():
        return path.resolve()
    for d in search_dirs(search):
        if (d / str(name_or_path) / "manifest.json").exists():
            return (d / str(name_or_path)).resolve()
    return None


# ------------------------------------------------------------------ the dataset

class Dataset:
    """One dataset folder (see the module docstring)."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{self.root}: manifest schema {self.manifest.get('schema_version')}")
        self._task_files: dict = {}
        self._grid_meta: dict = {}

    def __repr__(self):
        return f"Dataset({self.name!r}, env={self.env_type!r}, robots={self.robots})"

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def env_type(self) -> str:
        """"svg" (continuous maps: NavSim/Nav-v0) or "grid" (NavSim/GridNav-v0)."""
        return self.manifest["env"]

    @property
    def robots(self) -> list[str]:
        return sorted(r for r, v in self.manifest["robots"].items() if v.get("n_tasks"))

    @property
    def splits(self) -> dict[str, list[str]]:
        """split -> its buildings (scenes)."""
        return self.manifest["splits"]

    def robot(self, robot_id: str) -> Robot:
        """The robot as it was when the dataset was built."""
        return load_robot(robot_id, extra=self.root / "robots")

    def check_robot(self, robot_id: str) -> None:
        if robot_id not in self.manifest["robots"]:
            raise KeyError(f"robot '{robot_id}' not in dataset {self.name}; available: {self.robots}")

    def height_class(self, robot_id: str) -> str:
        """The map set of a robot (robots of one height class share the same maps)."""
        self.check_robot(robot_id)
        return self.manifest["robots"][robot_id]["height_class"]

    def default_robot(self, robot: str | None = None) -> str:
        """``robot``, else the only robot, else turtlebot4, else an error listing the choices."""
        if robot is not None:
            self.check_robot(robot)
            return robot
        if len(self.robots) == 1:
            return self.robots[0]
        if "turtlebot4" in self.robots:
            return "turtlebot4"
        raise ValueError(f"dataset {self.name} has several robots, choose one: {self.robots}")

    def maps(self, robot: str, split: str | None = None) -> list[str]:
        """Map ids with tasks for ``robot`` (in ``split``, or all)."""
        self.check_robot(robot)
        ids = sorted(p.stem for p in (self.root / "tasks" / robot).glob("*.json"))
        if split is None:
            return ids
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None, got {split!r}")
        return [m for m in ids if scene_of(m) in self.splits[split]]

    def task_file(self, robot: str, map_id: str) -> dict:
        key = (robot, map_id)
        if key not in self._task_files:
            self._task_files[key] = read_tasks(self.root / "tasks" / robot / f"{map_id}.json")
        return self._task_files[key]

    def tasks(self, robot: str, map_id: str) -> list[dict]:
        return self.task_file(robot, map_id)["tasks"]

    def svg_path(self, robot: str, map_id: str) -> Path:
        return self.root / "maps" / self.height_class(robot) / f"{map_id}.svg"

    def grid_arrays(self, map_id: str) -> dict:
        with np.load(self.root / "grids" / f"{map_id}.npz") as z:
            return {k: z[k] for k in z.files}

    def grid_meta(self, map_id: str) -> dict:
        if map_id not in self._grid_meta:
            path = self.root / "grids" / f"{map_id}.json"
            self._grid_meta[map_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._grid_meta[map_id]

    def summary(self) -> dict:
        """Number of maps and tasks per robot and split (``navsim info``)."""
        out = {"name": self.name, "env": self.env_type, "path": str(self.root), "robots": {}}
        for r in self.robots:
            out["robots"][r] = {s: {"maps": len(ms), "tasks": sum(len(self.tasks(r, m)) for m in ms)}
                                for s in SPLITS for ms in [self.maps(r, s)]}
        return out

    def validate(self) -> list[str]:
        """Problems found in the folder (empty = valid)."""
        return validate_dataset(self.root)


def load_dataset(name_or_path, search=None) -> Dataset:
    """A dataset by name (see the search order in the module docstring) or by path."""
    if isinstance(name_or_path, Dataset):
        return name_or_path
    found = find_dataset(name_or_path, search)
    if found is None:
        names = sorted({q.parent.name for d in search_dirs(search) if d.is_dir()
                        for q in d.glob("*/manifest.json")})
        raise FileNotFoundError(
            f"dataset '{name_or_path}' not found. Available: {names or 'none'}. "
            f"Download it with navsim.setup('{name_or_path}') (or `navsim setup {name_or_path}`), "
            "or point NAVSIM_DATASETS at the folder that contains it.")
    return Dataset(found)


def list_datasets(search=None) -> list[Dataset]:
    """Every dataset on this machine (the first one found of each name)."""
    seen, out = set(), []
    for d in search_dirs(search):
        for manifest in sorted(d.glob("*/manifest.json")) if d.is_dir() else []:
            if manifest.parent.name in seen:
                continue
            seen.add(manifest.parent.name)
            try:
                out.append(Dataset(manifest.parent))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return out


# ------------------------------------------------------------------ file format

def scene_of(map_id: str) -> str:
    """The building of a map: ``"isb-S001_s0"`` (storey 0 of S001) -> ``"isb-S001"``."""
    return map_id.rsplit("_s", 1)[0]


def read_tasks(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: task schema {data.get('schema_version')} (expected {SCHEMA_VERSION})")
    return data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(root) -> list[str]:
    """Problems of a dataset folder: missing, extra or modified files (against the sha256
    in the manifest) and unreadable task files. Empty list = valid."""
    root = Path(root)
    if not (root / "manifest.json").exists():
        return ["manifest.json missing"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"manifest schema {manifest.get('schema_version')}")
    recorded = manifest.get("files", {})
    actual = {p.relative_to(root).as_posix(): sha256(p)
              for d in ("maps", "grids", "tasks", "robots") if (root / d).exists()
              for p in sorted((root / d).rglob("*")) if p.is_file()}
    problems += [f"missing file: {f}" for f in sorted(set(recorded) - set(actual))]
    problems += [f"file not in manifest: {f}" for f in sorted(set(actual) - set(recorded))]
    problems += [f"modified file: {f}" for f in sorted(set(recorded) & set(actual))
                 if recorded[f] != actual[f]]
    for p in sorted((root / "tasks").rglob("*.json")) if (root / "tasks").exists() else []:
        try:
            read_tasks(p)
        except ValueError as e:
            problems.append(str(e))
    return problems
