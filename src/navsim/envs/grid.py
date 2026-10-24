"""``NavSim/GridNav-v0``: one robot moves cell by cell from a start to a goal on a grid.

    env = gym.make("NavSim/GridNav-v0", dataset="demo-grid", goal_obs=True)

One cell is one robot body. The grid comes from a real floor plan.

Action ``Discrete(4)``: 0 = up, 1 = down, 2 = left, 3 = right. A move into a wall or off
the grid leaves the robot where it is (counted as a collision).

Observation: ``lidar_rays`` LiDAR distances in cells, up to ``lidar_range``. The goal is
NOT observed by default: the robot has to search for it. To drive TO the goal use
``goal_obs=True``: the observation becomes ``Dict(lidar, goal)``, goal = (rows, columns)
from the robot to the goal. ``normalize_obs=True`` divides both by ``lidar_range``.
By default the LiDAR is the one of the curriculum-learning work: its rays start at the
top-left corner of the robot's cell, so it reads up to ~1 cell more down/right than
up/left. ``lidar_centered=True`` starts them at the centre of the cell (symmetric:
opposite rays read the same in a symmetric room); see :func:`lidar_scan`.

Reward: always 0.0. Write yours with ``navsim.wrappers.Reward(env, fn)`` from
``env.unwrapped.robot_state()`` (cell, goal, trajectory, shortest path, ...) and the step
info. ``terminated`` at the goal; ``truncated`` after ``budget`` steps.
``difficulty="easy" | "medium" | "hard" | "starved"`` keeps only the tasks of one level
(and the maps that have one): a task filter ``level=[difficulty]`` that stays in force
when you call ``set_task_filter``.

``info`` at every step: steps, d_bfs (shortest path from the start, in steps), collided,
goal_geodesic (shortest path from here to the goal), progress (how much it shrank). At the
end of an episode also: map_id, task_id, split, success, spl, path_length (m), time (steps),
n_collisions, termination, geodesic_m.

Choosing the map and task, curriculum: see :mod:`navsim.envs.episodes`.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .._sim.world import spl
from .episodes import EpisodeSource, as_point, custom_endpoints

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))      # up, down, left, right as (d_row, d_col)
LABELS = ("p0", "sa_uniform", "sa_persist", "starved", "level")


@dataclass
class GridMap:
    map_id: str
    grid: np.ndarray                 # 0 = free, anything else = blocked
    tasks: list
    header: dict                     # the task file
    cell_m: float                    # size of a cell (m)
    lidar: dict = field(default_factory=dict)    # (rays, range, centered) -> LiDAR of every cell
    goal_steps: OrderedDict = field(default_factory=OrderedDict)   # task -> steps to its goal


class GridNavEnv(EpisodeSource, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, dataset="demo-grid", robot=None, split=None, map_id=None, task_idx=None,
                 goal_obs=False, normalize_obs=False, budget=200, lidar_rays=22, lidar_range=10.0,
                 lidar_centered=False, difficulty=None, task_filter=None, render_mode=None, map_cache=256, map_repeat=1,
                 grid=None, tasks=None, cell_m=1.0):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"render_mode must be None or one of {self.metadata['render_modes']}")
        self.split, self.difficulty = split, difficulty
        if difficulty is not None and task_filter and "level" in task_filter:
            raise ValueError("give either difficulty or a level task filter, not both")
        episode_args = (map_id, task_idx, map_cache, map_repeat, task_filter)
        if tasks is not None:                       # one grid given in memory (tests)
            self.dataset = None
            self.robot_id = robot
            name = tasks.get("map_id", "map")
            # your grid may mark obstacles with any non-zero code (e.g. 100 or 255 of an
            # occupancy map): make them all 1, the code the LiDAR stops at
            grid = (np.asarray(grid) != 0).astype(np.int8)
            data = GridMap(name, grid, tasks["tasks"], tasks, cell_m)
            self._init_episodes([name], lambda m: data, *episode_args)
        else:
            from ..data import load_dataset
            self.dataset = ds = load_dataset(dataset)
            if ds.env_type != "grid":
                raise ValueError(f"dataset {ds.name} is a {ds.env_type} dataset: use NavSim/Nav-v0")
            self.robot_id = robot = ds.default_robot(robot)
            maps = [map_id] if map_id else ds.maps(robot, split)
            if not maps:
                raise ValueError(f"no {split} maps for {robot} in {ds.name}")
            self._init_episodes(maps, lambda m: GridMap(
                m, ds.grid_arrays(m)["grid"], ds.tasks(robot, m), ds.task_file(robot, m),
                ds.grid_meta(m)["cell_m"]), *episode_args)
        if difficulty is not None and not task_filter:
            self.set_task_filter()                  # the level of the difficulty alone
        self.budget = int(budget)
        self.lidar_rays, self.lidar_range = int(lidar_rays), float(lidar_range)
        self.lidar_centered = bool(lidar_centered)
        self.goal_obs, self.normalize_obs = goal_obs, normalize_obs
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        lidar = spaces.Box(0.0, 1.0 if normalize_obs else self.lidar_range, (self.lidar_rays,), np.float32)
        self.observation_space = (spaces.Dict({"lidar": lidar, "goal": spaces.Box(
            -np.inf, np.inf, (2,), np.float32)}) if goal_obs else lidar)
        self.agent_pos = None
        self.map_id = None

    def tasks_of(self, map_id) -> list:
        if self.dataset is None:
            return super().tasks_of(map_id)
        return self.dataset.tasks(self.robot_id, map_id)

    def set_task_filter(self, **criteria) -> dict:
        """See :mod:`navsim.envs.episodes`; with ``difficulty`` the filter also keeps
        ``level=[difficulty]``."""
        if self.difficulty is not None:
            if "level" in criteria:
                raise ValueError(f"this environment has difficulty={self.difficulty!r}: "
                                 "filter on other criteria than level")
            criteria["level"] = [self.difficulty]
        return super().set_task_filter(**criteria)

    @property
    def grid(self) -> np.ndarray:
        """The current grid: 0 = free, anything else = blocked."""
        return self.map_data.grid

    @property
    def tasks(self) -> list:
        return self.map_data.tasks

    # ------------------------------------------------------------ Gymnasium API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._on_seed(seed)
        self.map_id, self.map_data, self.task_idx = self._choose(options)
        custom = custom_endpoints(options)
        if custom:
            task = self._custom_task(*custom)
        else:
            task = self.tasks[self.task_idx]
            self.start, self.goal = list(task["start"]["cell"]), list(task["goal"]["cell"])
            self._goal_key = self.task_idx
            self.d_bfs = task["labels"]["d_bfs"]
        self.agent_pos = list(self.start)
        self.traj = [list(self.start)]
        self.steps = self.moves = self.bumps = 0
        self.collided = False
        self._steps_to_goal = self._goal_steps()
        self._prev_steps = float(self._steps_to_goal[self.agent_pos[0], self.agent_pos[1]])
        info = {"start": self.start, "goal": self.goal, "d_bfs": self.d_bfs,
                **{k: task["labels"].get(k) for k in LABELS},
                "map_id": self.map_id, "task_id": self.task_idx,
                "split": self.map_data.header.get("split", self.split)}
        if self.render_mode == "human":
            self.render()
        return self._observe(), info

    def step(self, action):
        if self.agent_pos is None:
            raise gym.error.ResetNeeded("call reset() before step()")
        dr, dc = MOVES[int(action)]
        row, col = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        H, W = self.grid.shape
        blocked = not (0 <= row < H and 0 <= col < W and self.grid[row, col] == 0)
        if blocked:
            self.bumps += 1
        else:
            self.agent_pos = [row, col]
            self.moves += 1
        self.collided = blocked
        self.traj.append(list(self.agent_pos))
        self.steps += 1
        terminated = self.agent_pos == self.goal
        truncated = (not terminated) and self.steps >= self.budget
        steps_left = float(self._steps_to_goal[self.agent_pos[0], self.agent_pos[1]])
        info = {"steps": self.steps, "d_bfs": self.d_bfs, "collided": blocked,
                "goal_geodesic": steps_left, "progress": self._prev_steps - steps_left}
        self._prev_steps = steps_left
        if terminated or truncated:
            cell_m = self.map_data.cell_m
            info.update({"map_id": self.map_id, "task_id": self.task_idx,
                         "split": self.map_data.header.get("split", self.split),
                         "success": bool(terminated), "spl": spl(terminated, self.d_bfs, self.moves),
                         "path_length": self.moves * cell_m, "time": self.steps,
                         "n_collisions": self.bumps,
                         "termination": "success" if terminated else "time_limit",
                         "geodesic_m": self.d_bfs * cell_m})
        if self.render_mode == "human":
            self.render()
        return self._observe(), 0.0, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.rgb_frame()
        if self.render_mode == "human":
            from ..visualize import show_frame
            show_frame(self, self.rgb_frame())
        return None

    def close(self):
        from ..visualize import close_window
        close_window(self)

    def robot_state(self) -> "GridRobotState":
        """Everything about the robot now: cell, goal, trajectory, shortest path, distances.
        Use it to write a reward."""
        return GridRobotState(self)

    # ------------------------------------------------------------ helpers

    def _custom_task(self, start, goal) -> dict:
        """Your own start and goal cells on the current map, checked: inside the grid,
        free, different and connected."""
        H, W = self.grid.shape
        cells = []
        for which, value in (("start", start), ("goal", goal)):
            if isinstance(value, dict) and "cell" in value:
                value = value["cell"]
            row, col = (int(round(x)) for x in as_point(value, ("row", "col"), 2))
            if not (0 <= row < H and 0 <= col < W):
                raise ValueError(f"{which} cell ({row}, {col}) is outside the {H} x {W} grid "
                                 f"of {self.map_id}")
            if self.grid[row, col] != 0:
                raise ValueError(f"{which} cell ({row}, {col}) of {self.map_id} is blocked")
            cells.append([row, col])
        if cells[0] == cells[1]:
            raise ValueError("start and goal are the same cell")
        self.start, self.goal = cells
        self.task_idx, self._goal_key = -1, ("goal", *self.goal)
        steps = self._goal_steps()[self.start[0], self.start[1]]
        if not np.isfinite(steps):
            raise ValueError(f"goal {tuple(self.goal)} cannot be reached from start {tuple(self.start)} "
                             f"on {self.map_id}")
        self.d_bfs = int(steps)
        return {"id": -1, "start": {"cell": self.start}, "goal": {"cell": self.goal},
                "labels": {"d_bfs": self.d_bfs}}

    def _goal_steps(self) -> np.ndarray:
        """Fewest steps from every cell to the goal (inf = unreachable); the last 8 are kept."""
        cache, key = self.map_data.goal_steps, self._goal_key
        steps = cache.get(key)
        if steps is None:
            raw = bfs(self.grid == 0, self.goal)
            steps = cache[key] = np.where(raw < 0, np.inf, raw).astype(float)
            while len(cache) > 8:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return steps

    def lidar_ranges(self) -> np.ndarray:
        """The LiDAR of the current cell. It depends on the cell only, so each cell is
        scanned once and the result kept."""
        key = (self.lidar_rays, self.lidar_range, self.lidar_centered)
        table = self.map_data.lidar.get(key)
        if table is None:
            table = self.map_data.lidar[key] = np.full((*self.grid.shape, self.lidar_rays), np.nan,
                                                       np.float32)
        ranges = table[self.agent_pos[0], self.agent_pos[1]]
        if np.isnan(ranges[0]):
            ranges[:] = lidar_scan(self.grid, self.agent_pos, self.lidar_rays, self.lidar_range,
                                   self.lidar_centered)
        return ranges.copy()

    def _observe(self):
        lidar = self.lidar_ranges()
        scale = np.float32(self.lidar_range if self.normalize_obs else 1.0)
        if self.normalize_obs:
            lidar = lidar / scale
        if not self.goal_obs:
            return lidar
        goal = np.array([self.goal[0] - self.agent_pos[0], self.goal[1] - self.agent_pos[1]],
                        np.float32) / scale
        return {"lidar": lidar, "goal": goal}

    def rgb_frame(self, cell=8) -> np.ndarray:
        """Image of the grid (free white, walls black), goal red, robot green."""
        img = np.where(self.grid[..., None] == 0, 255, 0).astype(np.uint8).repeat(3, 2)
        for (row, col), color in ((self.goal, (220, 0, 0)), (self.agent_pos, (0, 160, 0))):
            img[row, col] = color
        return img.repeat(cell, 0).repeat(cell, 1)


class GridRobotState:
    """Everything about the robot in the current episode of ``GridNav-v0``, to write a reward
    or analyse a run: ``env.unwrapped.robot_state()``. Positions are cells (row, col),
    distances are in cells (one move = 1); a ``cell`` argument defaults to the robot's cell.
    It always shows the latest state."""

    def __init__(self, env: GridNavEnv):
        if env.agent_pos is None:
            raise RuntimeError("reset the environment first")
        self._env = env

    def __repr__(self):
        e = self._env
        return f"GridRobotState(position={tuple(e.agent_pos)}, goal={tuple(e.goal)}, steps={e.steps})"

    # -------------------------------------------------- now
    @property
    def position(self) -> tuple:
        """(row, col) of the robot."""
        return tuple(int(v) for v in self._env.agent_pos)

    @property
    def collided(self) -> bool:
        """Whether the last move was blocked (wall or edge of the grid)."""
        return self._env.collided

    @property
    def reached_goal(self) -> bool:
        return list(self._env.agent_pos) == list(self._env.goal)

    @property
    def done(self) -> bool:
        """Whether the episode is over (goal reached or budget used up)."""
        return self.reached_goal or self._env.steps >= self._env.budget

    # -------------------------------------------------- the episode
    @property
    def start(self) -> tuple:
        return tuple(int(v) for v in self._env.start)

    @property
    def goal(self) -> tuple:
        return tuple(int(v) for v in self._env.goal)

    @property
    def start_geodesic(self) -> int:
        """Fewest moves from the start to the goal."""
        return int(self._env.d_bfs)

    @property
    def trajectory(self) -> np.ndarray:
        """(N, 2) cells (row, col), one per step since reset (row 0 = the start; a blocked
        move repeats the cell)."""
        return np.array(self._env.traj, int)

    @property
    def path_length(self) -> int:
        """Moves made since reset (blocked moves not counted)."""
        return self._env.moves

    @property
    def steps(self) -> int:
        return self._env.steps

    @property
    def budget(self) -> int:
        return self._env.budget

    @property
    def n_collisions(self) -> int:
        """Blocked moves since reset."""
        return self._env.bumps

    @property
    def cell_m(self) -> float:
        """Size of a cell in metres."""
        return self._env.map_data.cell_m

    # -------------------------------------------------- questions
    def _cell(self, cell):
        return self.position if cell is None else (int(cell[0]), int(cell[1]))

    def goal_distance(self, cell=None) -> float:
        """Straight-line distance from ``cell`` to the goal (cells)."""
        r, c = self._cell(cell)
        return float(np.hypot(self._env.goal[0] - r, self._env.goal[1] - c))

    def geodesic_distance(self, cell=None) -> float:
        """Fewest moves from ``cell`` to the goal (inf if unreachable or blocked)."""
        r, c = self._cell(cell)
        H, W = self._env.grid.shape
        if not (0 <= r < H and 0 <= c < W):
            return float("inf")
        return float(self._env._steps_to_goal[r, c])

    def shortest_path(self, cell=None) -> np.ndarray:
        """A shortest path from ``cell`` to the goal: (M, 2) cells, both ends included
        (empty if unreachable)."""
        steps = self._env._steps_to_goal
        r, c = self._cell(cell)
        if not np.isfinite(self.geodesic_distance((r, c))):
            return np.zeros((0, 2), int)
        H, W = steps.shape
        path = [(r, c)]
        while steps[r, c] > 0:
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and steps[nr, nc] == steps[r, c] - 1:
                    r, c = nr, nc
                    break
            path.append((r, c))
        return np.array(path, int)

    def lidar(self) -> np.ndarray:
        """LiDAR distances in cells (lidar_rays,), not normalised."""
        return self._env.lidar_ranges()

    def is_free(self, cell) -> bool:
        """Whether a cell is inside the grid and not blocked."""
        r, c = int(cell[0]), int(cell[1])
        H, W = self._env.grid.shape
        return 0 <= r < H and 0 <= c < W and self._env.grid[r, c] == 0


def bfs(free: np.ndarray, start) -> np.ndarray:
    """Fewest steps (4 moves) from ``start`` to every cell; -1 = unreachable."""
    H, W = free.shape
    dist = np.full((H, W), -1, np.int32)
    r0, c0 = int(start[0]), int(start[1])
    if not free[r0, c0]:
        return dist
    dist[r0, c0] = 0
    queue = deque([(r0, c0)])
    while queue:
        r, c = queue.popleft()
        d = dist[r, c] + 1
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = d
                queue.append((nr, nc))
    return dist


def lidar_scan(grid: np.ndarray, position, n_rays: int = 22, max_range: float = 10.0,
               centered: bool = False) -> np.ndarray:
    """Grid LiDAR in CELLS: the ``lidar_scan`` of the curriculum-learning work, vectorised
    (tests/test_grid_env.py checks it against the original loop).

    ``n_rays`` rays at angles linspace(0, 2 pi, n_rays, endpoint=False) start from the cell
    index (row, col) and go row += t cos(a), col += t sin(a). Each ray is sampled every 0.1
    cell (int() truncation) and stops at the edge of the grid or at a cell equal to 1; a ray
    that hits nothing reads ``max_range``. Starting at the corner of the cell and truncating
    toward zero make it read more down/right than up/left (10.0 vs 9.1 in the middle of a
    square room). ``centered=True`` corrects both: the rays start at the centre of the cell
    (row + 0.5, col + 0.5), the cell of a sample is its floor, and each 0.1 step is sampled
    in its middle, so opposite rays read the same distance in a symmetric room."""
    H, W = grid.shape
    row, col = position
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    t = np.arange(0, max_range, 0.1)
    if centered:
        mid = t[None, :] + 0.05
        rows = np.floor(row + 0.5 + mid * np.cos(angles)[:, None]).astype(np.int64)
        cols = np.floor(col + 0.5 + mid * np.sin(angles)[:, None]).astype(np.int64)
    else:
        rows = (row + t[None, :] * np.cos(angles)[:, None]).astype(np.int64)
        cols = (col + t[None, :] * np.sin(angles)[:, None]).astype(np.int64)
    outside = (rows < 0) | (rows >= H) | (cols < 0) | (cols >= W)
    hit = outside.copy()
    hit[~outside] = grid[rows[~outside], cols[~outside]] == 1
    first = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), t[first], max_range).astype(np.float32)
