"""How every environment picks the map and the task (start, goal) of an episode.

At ``reset`` (the first rule that applies wins):

1. ``reset(options={"map_id": m, "start": ..., "goal": ...})``: your own start and goal on
   map m (checked: the start must be free and the goal reachable, else ValueError).
   ``Nav-v0``: start (x, y[, theta]) and goal (x, y) in metres; ``GridNav-v0``: (row, col)
   cells. The episode's task_id is -1.
2. ``reset(options={"map_id": m, "task_idx": i})``: task i of map m.
3. the ``map_id`` / ``task_idx`` given when the environment was made.
4. otherwise a random map of the split, then a random task of that map (seeded by
   ``reset(seed=...)``).

Curriculum: ``set_task_filter`` restricts the random choice (rule 4) to the tasks that
match every criterion, and to the maps that have one (a map chosen with ``map_id`` that
has no matching task raises ValueError at reset, unless ``task_idx`` is given too):

    env.unwrapped.set_task_filter(geodesic_m=(1.0, 5.0))      # range, inclusive (None = no bound)
    env.unwrapped.set_task_filter(geodesic_m={"min": 1.0})   # the same as a dict
    env.unwrapped.set_task_filter(level=["easy", "medium"])  # list = allowed values
    env.unwrapped.set_task_filter()                          # no filter
    venv.call("set_task_filter", geodesic_m=(1.0, 5.0))      # in every env of a vector env

A criterion is looked up in the task (``geodesic_m``, ``euclidean_m``), then in its
``labels`` (``Nav-v0``: ``gdr``; ``GridNav-v0``: ``d_bfs``, ``level``, ...). A task without
it does not match.

Loaded maps are kept in memory (the last ``map_cache`` ones). ``map_repeat = K`` keeps a
random map for K episodes in a row (with a new task each time): K times fewer map loads,
but consecutive episodes of one environment are then alike. ``reset(seed=...)`` starts a
new run, so a seed always gives the same episode.
"""

from __future__ import annotations

from collections import OrderedDict

_MISSING = object()


class EpisodeSource:
    """Mixin of the environments: the maps of the split, the maps in memory, and the
    choice of the episode at reset. Needs ``self.np_random``."""

    def _init_episodes(self, map_ids, loader, map_id=None, task_idx=None, map_cache=8,
                       map_repeat=1, task_filter=None):
        """``loader(map_id)`` loads one map (with its ``tasks``)."""
        if not map_ids:
            raise ValueError("no maps with tasks for this robot/split")
        if int(map_repeat) < 1:
            raise ValueError("map_repeat must be >= 1")
        self._map_ids = list(map_ids)
        self._loader = loader
        self._fixed_map, self._fixed_task = map_id, task_idx
        self._loaded: OrderedDict = OrderedDict()          # map id -> map, most recent last
        self._map_cache = max(1, int(map_cache))
        self._map_repeat = int(map_repeat)
        self._repeat_map, self._repeats_left = None, 0
        self._criteria, self._eligible = None, None        # task filter, map -> task indices
        if task_filter:
            self.set_task_filter(**task_filter)

    @property
    def map_ids(self) -> list[str]:
        """Every map this environment draws from."""
        return list(self._map_ids)

    def tasks_of(self, map_id) -> list:
        """The tasks of a map (environments made from a dataset read them without loading
        the map)."""
        return self._load_map(map_id).tasks

    def set_task_filter(self, **criteria) -> dict:
        """Restrict random episodes to the matching tasks (see the module docstring). Returns
        how many maps and tasks are left; ValueError if none."""
        self._repeat_map, self._repeats_left = None, 0
        if not criteria:
            self._criteria, self._eligible = None, None
            return {"maps": len(self._map_ids),
                    "tasks": sum(len(self.tasks_of(m)) for m in self._map_ids)}
        criteria = {k: _as_range(k, v) if isinstance(v, dict) else v for k, v in criteria.items()}
        for k, v in criteria.items():
            if isinstance(v, tuple) and (len(v) != 2 or not all(
                    b is None or isinstance(b, (int, float)) for b in v)):
                raise ValueError(f"{k}: a range is (lo, hi) with numbers or None, got {v!r}")
        eligible = {}
        for m in self._map_ids:
            matching = [i for i, t in enumerate(self.tasks_of(m)) if task_matches(t, criteria)]
            if matching:
                eligible[m] = matching
        if not eligible:
            raise ValueError(f"no task matches {criteria}")
        self._criteria, self._eligible = dict(criteria), eligible
        return {"maps": len(eligible), "tasks": sum(map(len, eligible.values()))}

    def _on_seed(self, seed) -> None:
        """A new seed starts the sequence of episodes over, the ``map_repeat`` run too:
        ``reset(seed=s)`` then always gives the same episode."""
        if seed is not None:
            self._repeat_map, self._repeats_left = None, 0

    def _load_map(self, map_id):
        if map_id in self._loaded:
            self._loaded.move_to_end(map_id)
            return self._loaded[map_id]
        loaded = self._loader(map_id)
        self._loaded[map_id] = loaded
        if len(self._loaded) > self._map_cache:
            self._loaded.popitem(last=False)
        return loaded

    def _choose_map(self, options, custom: bool = False) -> str:
        options = options or {}
        map_id = options.get("map_id", self._fixed_map)
        if custom and map_id is None:
            if len(self._map_ids) > 1:
                raise ValueError("a custom start/goal needs map_id in the reset options "
                                 f"(maps: {self._map_ids[:5]}{' ...' if len(self._map_ids) > 5 else ''})")
            return self._map_ids[0]
        if map_id is None and self._repeats_left > 0:
            self._repeats_left -= 1
            return self._repeat_map
        if map_id is None:
            maps = self._map_ids if self._eligible is None else list(self._eligible)
            map_id = maps[int(self.np_random.integers(len(maps)))]
            self._repeat_map, self._repeats_left = map_id, self._map_repeat - 1
        return map_id

    def _task_pool(self, map_id, loaded) -> list[int]:
        """Indices of the tasks a random task is drawn from on this map (the task filter
        applied); ValueError if the filter leaves none."""
        every = list(range(len(loaded.tasks)))
        if self._criteria is None:
            return every
        pool = (self._eligible.get(map_id) or
                [i for i in every if task_matches(loaded.tasks[i], self._criteria)])
        if not pool:
            raise ValueError(f"map {map_id} has no task that matches the task filter {self._criteria}")
        return pool

    def _choose(self, options):
        """(map id, loaded map, task index) of a one-robot episode."""
        options = options or {}
        map_id = self._choose_map(options, custom_endpoints(options) is not None)
        loaded = self._load_map(map_id)
        idx = options.get("task_idx", self._fixed_task)
        if idx is None:
            pool = self._task_pool(map_id, loaded)
            idx = pool[int(self.np_random.integers(len(pool)))]
        return map_id, loaded, int(idx) % len(loaded.tasks)


def custom_endpoints(options) -> tuple | None:
    """(start, goal) given in the reset options, or None. Both are needed."""
    if not options or ("start" not in options and "goal" not in options):
        return None
    if options.get("start") is None or options.get("goal") is None:
        raise ValueError("a custom episode needs both 'start' and 'goal'")
    return options["start"], options["goal"]


def as_point(value, names, n_required) -> list[float]:
    """[a, b, ...] from a sequence, or from a dict with the keys ``names`` (the keys after
    the first ``n_required`` are optional)."""
    if isinstance(value, dict):
        value = [value[k] for k in names if k in value]
    value = [float(x) for x in value]
    if not n_required <= len(value) <= len(names):
        optional = f"[, {', '.join(names[n_required:])}]" if len(names) > n_required else ""
        raise ValueError(f"expected {', '.join(names[:n_required])}{optional}, got {value}")
    return value


def _as_range(key, value: dict) -> tuple:
    """{"min": a, "max": b} (either optional) -> (a, b)."""
    unknown = set(value) - {"min", "max"}
    if unknown:
        raise ValueError(f"{key}: a range dict takes 'min' and 'max', got {sorted(unknown)}")
    return (value.get("min"), value.get("max"))


def task_matches(task: dict, criteria: dict) -> bool:
    """Whether a task meets every criterion: a (lo, hi) range (None = no bound), a list
    of allowed values, or one value."""
    labels = task.get("labels") or {}
    for key, wanted in criteria.items():
        value = task.get(key, labels.get(key, _MISSING))
        if value is _MISSING:
            return False
        if isinstance(wanted, tuple):
            lo, hi = wanted
            if value is None or (lo is not None and value < lo) or (hi is not None and value > hi):
                return False
        elif isinstance(wanted, (list, set, frozenset)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True
