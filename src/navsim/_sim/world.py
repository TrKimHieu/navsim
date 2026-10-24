"""The simulator behind ``NavSim/Nav-v0`` (one robot) and ``navsim.multiagent.nav_v0`` (several).

A ``World`` holds N >= 1 robots on one map. Each step:

1. every robot's action (v, [vy], w) is scaled by its speed limits;
2. the motion over ``dt`` is split into small sub-steps (<= 2 cm, <= 2 degrees), so a fast
   robot cannot jump through a thin wall;
3. a sub-step that would hit a wall or another robot stops that robot where it was
   (a collision); the goal is checked after every sub-step;
4. each robot gets its observation (LiDAR, goal, velocity), step info and episode end.

There is no reward: the environments return 0.0 and you write your own reward from
``RobotState`` (``robot_state()``) and the step info.

Units are metres, seconds and radians; angles are counter-clockwise from +x (ROS REP-103).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np
from gymnasium import spaces

from ..robots import Robot
from .geometry import Collider, place, polygon_segments, polygons_overlap, segments_raycast
from .planning import TURN_SMOOTH, Planner, simplify, turning
from .svgmap import SvgMap

SUBSTEP_DISTANCE = 0.02                 # longest move of one sub-step (m)
SUBSTEP_ANGLE = math.radians(2.0)       # largest turn of one sub-step (rad)
# robot colours in renders (RGB); with one robot: green robot, red goal
COLORS = [(0, 160, 0), (31, 119, 180), (255, 127, 14), (148, 103, 189), (140, 86, 75),
          (227, 119, 194), (23, 190, 207), (188, 189, 34), (127, 127, 127), (214, 39, 40)]


def spl(success: bool, shortest: float, taken: float) -> float:
    """Success weighted by Path Length (Anderson et al. 2018): success x shortest / max(taken, shortest)."""
    return float(success) * shortest / max(taken, shortest) if shortest > 0 else float(success)


# ------------------------------------------------------------------ motion

def wrap_angle(a: float) -> float:
    """Angle in [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def integrate(x, y, theta, vx, vy, w, dt):
    """Exact pose after driving at a constant body velocity (vx, vy, w) for dt.
    A differential drive has vy = 0 (it drives on a circular arc)."""
    theta1 = theta + w * dt
    if abs(w) < 1e-9:
        c, s = math.cos(theta) * dt, math.sin(theta) * dt
    else:
        c = (math.sin(theta1) - math.sin(theta)) / w          # integral of cos
        s = -(math.cos(theta1) - math.cos(theta)) / w         # integral of sin
    return (x + vx * c - vy * s, y + vx * s + vy * c, wrap_angle(theta1))


def bodies_overlap(pose_a, robot_a: Robot, pose_b, robot_b: Robot) -> bool:
    """Whether two robot footprints overlap (the cheap bounding-circle test first)."""
    if math.hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1]) > robot_a.r_circ + robot_b.r_circ:
        return False
    return polygons_overlap(place(robot_a.footprint, *pose_a), place(robot_b.footprint, *pose_b))


# ------------------------------------------------------------------ maps

@dataclass
class MapData:
    """One loaded map: its geometry, collision checker, path planner (built on first use),
    tasks, and caches of shortest-path results."""

    map_id: str
    svgmap: SvgMap
    collider: Collider
    tasks: list
    header: dict                    # the task file (split, success_radius, ...)
    clearance: float                # extra margin of the planner around walls (m)
    robot_radius: float             # the planner keeps the robot centre this far from walls
    field_cache: int = 4            # how many goal distance fields to keep
    _planner: Planner | None = None
    goal_fields: OrderedDict = field(default_factory=OrderedDict)   # key -> distance field
    path_trees: OrderedDict = field(default_factory=OrderedDict)    # key -> shortest-path predecessors
    turns: dict = field(default_factory=dict)                       # key -> turning along the path (rad)
    renders: dict = field(default_factory=dict)                     # resolution -> background image

    @property
    def planner(self) -> Planner:
        if self._planner is None:
            self._planner = Planner(self.svgmap, self.robot_radius, self.clearance)
        return self._planner


def new_map_data(map_id, svgmap, header, clearance, robot_radius, field_cache=4,
                 edge_dist=None) -> MapData:
    return MapData(map_id, svgmap, Collider(svgmap, edge_dist=edge_dist), header["tasks"], header,
                   clearance if clearance is not None else header.get("clearance", 0.05),
                   robot_radius, max(1, field_cache))


class MapLoader:
    """``loader(map_id) -> MapData`` for one robot of a dataset, read from the disk cache
    when possible (see :mod:`navsim._sim.cache`)."""

    def __init__(self, dataset, robot_id: str, robot_radius: float, clearance=None,
                 field_cache=4, disk_cache=True):
        self.dataset, self.robot_id, self.robot_radius = dataset, robot_id, robot_radius
        self.clearance, self.field_cache = clearance, field_cache
        self.disk = None
        if disk_cache:
            from .cache import DatasetCache
            cache = DatasetCache(dataset)
            self.disk = cache if cache.enabled else None

    def __call__(self, map_id: str) -> MapData:
        ds, robot_id, disk = self.dataset, self.robot_id, self.disk
        paths = disk.paths(robot_id, map_id, self.robot_radius, self.clearance) if disk else None
        hit = disk.load(paths) if paths else None
        if hit is not None:
            data = new_map_data(map_id, hit["svgmap"], hit["header"], self.clearance,
                                self.robot_radius, self.field_cache, hit["edge_dist"])
            data._planner = hit["planner"]
            data.turns.update(hit["turns"])
            return data
        data = new_map_data(map_id, SvgMap.load(ds.svg_path(robot_id, map_id)),
                            ds.task_file(robot_id, map_id), self.clearance, self.robot_radius,
                            self.field_cache)
        if paths:
            disk.save(paths, data.svgmap, data.collider, data.header, data.planner)
        return data


# ------------------------------------------------------------------ robots

@dataclass
class Agent:
    """The state of one robot in the world."""

    robot: Robot
    beam_angles: np.ndarray          # LiDAR beam directions in the robot frame (rad)
    first_beam: float                # angle of beam 0
    beam_step: float                 # angle between two beams
    full_circle: bool                # 360 degree LiDAR
    speed_limits: np.ndarray         # (v, [vy], w) used to normalise the velocity
    noise_rng: np.random.Generator | None = None
    pose: tuple = (0.0, 0.0, 0.0)    # x, y, theta
    velocity: tuple = (0.0, 0.0)     # (v, w) or (v, vy, w)
    task: dict | None = None         # None: not placed on the map
    task_id: int = -1
    field_key: object = None         # cache key of the goal distance field
    turn_key: object = None          # cache key of the turning estimate
    success_radius: float = 0.2
    geodesic: float = 0.0            # shortest path length from start to goal (m)
    time_limit: float = float("inf")
    time: float = 0.0
    path_length: float = 0.0
    n_collisions: int = 0
    collided: bool = False           # in the last step
    reached_goal: bool = False
    steps: int = 0
    traj: list = field(default_factory=list)   # (time, x, y, theta) since reset
    done: bool = False
    present: bool = True             # False once removed from the map (on_done="remove")
    prev_geodesic: float = 0.0       # distance to the goal after the previous step


def new_agent(robot: Robot, n_beams: int) -> Agent:
    lidar = robot.lidar
    full_circle = lidar["fov"] >= 2 * math.pi - 1e-3
    first_beam = 0.0 if full_circle else -lidar["fov"] / 2
    beam_step = 2 * math.pi / n_beams if full_circle else lidar["fov"] / (n_beams - 1)
    limits = np.array([max(robot.v_max, -robot.v_min)] + ([robot.vy_max] if robot.omni else [])
                      + [robot.w_max], np.float32)
    return Agent(robot=robot, beam_angles=first_beam + beam_step * np.arange(n_beams),
                 first_beam=first_beam, beam_step=beam_step, full_circle=full_circle,
                 speed_limits=limits, velocity=(0.0,) * robot.action_dim)


# ------------------------------------------------------------------ world

class World:
    """N robots on one map: ``set_map``, then ``place`` every robot, then ``step``."""

    def __init__(self, robots: list[Robot], dt=0.1, n_beams=72, lidar_noise=0.0,
                 on_collision="stop", time_limit=None, time_factor=3.0, success_radius=None,
                 nav_info=True, on_done="park"):
        if not robots:
            raise ValueError("need at least one robot")
        _check_choice("on_collision", on_collision, ("stop", "terminate"))
        _check_choice("on_done", on_done, ("park", "remove"))
        if int(n_beams) < 2:
            raise ValueError("n_beams must be >= 2")
        self.robots = list(robots)
        self.n_robots = len(self.robots)
        self.dt, self.n_beams, self.lidar_noise = dt, int(n_beams), lidar_noise
        self.on_collision, self.on_done = on_collision, on_done
        self.fixed_time_limit, self.time_factor = time_limit, time_factor
        self.fixed_success_radius = success_radius
        self.nav_info = nav_info
        self.agents = [new_agent(r, self.n_beams) for r in self.robots]
        self.map_data: MapData | None = None

    # ------------------------------------------------------------ spaces

    def action_space(self, i: int) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (self.robots[i].action_dim,), np.float32)

    def observation_space(self, i: int, normalize: bool) -> spaces.Dict:
        agent, lidar = self.agents[i], self.robots[i].lidar
        goal_low = np.array([0, -1, -1], np.float32)
        if normalize:
            return spaces.Dict({
                "lidar": spaces.Box(0.0, 1.0, (self.n_beams,), np.float32),
                "goal": spaces.Box(goal_low, np.ones(3, np.float32)),
                "velocity": spaces.Box(-1.0, 1.0, agent.speed_limits.shape, np.float32)})
        return spaces.Dict({
            "lidar": spaces.Box(lidar["range_min"], lidar["range_max"], (self.n_beams,), np.float32),
            "goal": spaces.Box(goal_low, np.array([np.inf, 1, 1], np.float32)),
            "velocity": spaces.Box(-agent.speed_limits, agent.speed_limits)})

    # ------------------------------------------------------------ map and tasks

    def set_map(self, map_data: MapData) -> None:
        """Switch to a map; no robot is placed until ``place``."""
        self.map_data = map_data
        for agent in self.agents:
            agent.task, agent.done, agent.present = None, False, True

    def goal_field(self, agent: Agent):
        """Shortest-path distance from every free cell to the agent's goal (kept in a small
        cache per map). The first time a task is seen, the same Dijkstra run also gives the
        path whose turning sets the default time limit. Dataset tasks are keyed by their
        index; custom episodes by their goal (field) and start + goal (turning)."""
        m, field_key, turn_key = self.map_data, agent.field_key, agent.turn_key
        dist = m.goal_fields.get(field_key)
        need_turn = not agent.robot.omni and turn_key not in m.turns
        if dist is not None and not need_turn:
            m.goal_fields.move_to_end(field_key)
            return dist
        goal = agent.task["goal"]
        if need_turn:
            dist, predecessors = m.planner.field(goal["x"], goal["y"], predecessors=True)
            m.turns[turn_key] = _path_turning(m.planner, agent.task, predecessors)
        else:
            dist = m.planner.field(goal["x"], goal["y"])
        m.goal_fields[field_key] = dist
        m.goal_fields.move_to_end(field_key)
        while len(m.goal_fields) > m.field_cache:
            m.goal_fields.popitem(last=False)
        return dist

    def geodesic_to_goal(self, agent: Agent, x: float, y: float) -> float:
        return self.map_data.planner.lookup(self.goal_field(agent), x, y)

    def custom_task(self, i: int, start, goal) -> tuple[dict, tuple, tuple]:
        """(task, field key, turn key) for your own start and goal of robot ``i``, after
        checking that the robot fits at the start and can reach the goal."""
        from ..envs.episodes import as_point
        robot, m = self.robots[i], self.map_data
        planner = m.planner
        sx, sy, *theta = as_point(start, ("x", "y", "theta"), 2)
        gx, gy = as_point(goal, ("x", "y"), 2)
        theta = theta[0] if theta else math.atan2(gy - sy, gx - sx)      # default: face the goal
        who = f"{robot.id} on {m.map_id}" + (f" (robot {i})" if self.n_robots > 1 else "")
        if not m.collider.pose_valid(robot.footprint, sx, sy, theta):
            raise ValueError(f"start ({sx:g}, {sy:g}, {theta:.3f}) collides with the map ({who})")
        if not planner.in_free_space(gx, gy):
            need = planner.robot_radius + planner.clearance
            raise ValueError(f"goal ({gx:g}, {gy:g}) is too close to a wall or outside the free "
                             f"area for {who} (needs {need:.2f} m of clearance)")
        task = {"id": -1, "start": {"x": sx, "y": sy, "theta": theta}, "goal": {"x": gx, "y": gy},
                "euclidean_m": math.hypot(gx - sx, gy - sy), "labels": {}}
        agent = self.agents[i]
        keys = (("goal", gx, gy), ("task", sx, sy, theta, gx, gy))
        saved = (agent.task, agent.field_key, agent.turn_key)
        agent.task, agent.field_key, agent.turn_key = task, *keys
        try:
            dist = self.goal_field(agent)
        finally:
            agent.task, agent.field_key, agent.turn_key = saved
        row, col = planner.nearest_cell(sx, sy)
        if not np.isfinite(dist[row, col]):
            raise ValueError(f"goal ({gx:g}, {gy:g}) cannot be reached from the start ({sx:g}, "
                             f"{sy:g}) by {who}: another room, or a passage too narrow")
        task["geodesic_m"] = planner.lookup(dist, sx, sy)
        return task, *keys

    def overlapping(self, i: int, pose) -> int | None:
        """The first other placed robot whose body overlaps robot ``i`` at ``pose``."""
        for j, other in enumerate(self.agents):
            if j != i and other.task is not None and other.present \
                    and bodies_overlap(pose, self.robots[i], other.pose, other.robot):
                return j
        return None

    def place(self, i: int, task: dict, task_id: int, field_key, turn_key) -> dict:
        """Put robot ``i`` at the start of ``task`` and return the reset info. Raises
        ValueError if the start collides with the map or with a robot already placed
        (robot ``i`` then stays unplaced)."""
        agent, m = self.agents[i], self.map_data
        start = task["start"]
        pose = (start["x"], start["y"], start["theta"])
        if not m.collider.pose_valid(agent.robot.footprint, *pose):
            raise ValueError(f"{m.map_id} task {task_id}: the start pose collides with the map "
                             f"for {agent.robot.id}")
        agent.task = None
        j = self.overlapping(i, pose)
        if j is not None:
            raise ValueError(f"start ({pose[0]:g}, {pose[1]:g}) of robot {i} overlaps robot {j} "
                             f"on {m.map_id}: pick a start further away")
        agent.task, agent.task_id, agent.field_key, agent.turn_key = task, task_id, field_key, turn_key
        agent.success_radius = self.fixed_success_radius or m.header.get("success_radius", 0.2)
        agent.pose = pose
        agent.velocity = (0.0,) * agent.robot.action_dim
        agent.time, agent.path_length, agent.n_collisions, agent.steps = 0.0, 0.0, 0, 0
        agent.geodesic = float(task.get("geodesic_m", 0.0))
        agent.time_limit = self.fixed_time_limit or self._default_time_limit(agent)
        agent.done, agent.present, agent.collided, agent.reached_goal = False, True, False, False
        agent.traj = [(0.0, *pose)]
        agent.prev_geodesic = (self.geodesic_to_goal(agent, start["x"], start["y"])
                               if self.nav_info else 0.0)
        return {"map_id": m.map_id, "task_id": task_id, "split": m.header.get("split"),
                "geodesic_m": agent.geodesic, "time_limit": agent.time_limit}

    def shortest_path(self, agent: Agent, x: float, y: float) -> np.ndarray:
        """Shortest drivable path from (x, y) to the agent's goal: (M, 2) points in metres,
        from (x, y) to the goal, grid staircase removed. Empty when the goal is unreachable.
        (One Dijkstra run per goal; the last few are kept.)"""
        m = self.map_data
        tree = m.path_trees.get(agent.field_key)
        goal = agent.task["goal"]
        if tree is None:
            _, tree = m.planner.field(goal["x"], goal["y"], predecessors=True)
            m.path_trees[agent.field_key] = tree
            while len(m.path_trees) > m.field_cache:
                m.path_trees.popitem(last=False)
        else:
            m.path_trees.move_to_end(agent.field_key)
        if not math.isfinite(self.geodesic_to_goal(agent, x, y)):
            return np.zeros((0, 2))
        return simplify([(x, y)] + m.planner.path(tree, x, y) + [(goal["x"], goal["y"])], TURN_SMOOTH)

    def _default_time_limit(self, agent: Agent) -> float:
        """time_factor x the least time needed: drive the shortest path at top speed and
        turn through its corners at top turning speed (omnidirectional robots need not turn)."""
        turn = 0.0
        if not agent.robot.omni:
            if agent.turn_key not in self.map_data.turns:
                self.goal_field(agent)
            turn = self.map_data.turns[agent.turn_key]
        return self.time_factor * (agent.geodesic / agent.robot.v_max + turn / agent.robot.w_max)

    # ------------------------------------------------------------ sensing

    def scan(self, i: int) -> np.ndarray:
        """LiDAR ranges of robot ``i`` without noise (m, before clipping). Other robots
        on the map block the beams too."""
        agent, lidar = self.agents[i], self.robots[i].lidar
        x, y, theta = agent.pose
        ranges = self.map_data.collider.raycast_uniform(
            x, y, theta + agent.first_beam, agent.beam_step, self.n_beams, lidar["range_max"],
            agent.full_circle)
        for body in (self._bodies_in_reach(i, lidar["range_max"]) if self.n_robots > 1 else []):
            ranges = np.minimum(ranges, segments_raycast(x, y, theta + agent.beam_angles,
                                                         polygon_segments(body), lidar["range_max"]))
        return ranges

    def _bodies_in_reach(self, i: int, reach: float) -> list[np.ndarray]:
        """Footprints (map frame) of the other robots a beam of length ``reach`` can touch."""
        x, y = self.agents[i].pose[:2]
        return [place(b.robot.footprint, *b.pose) for j, b in enumerate(self.agents)
                if j != i and b.task is not None and b.present
                and math.hypot(b.pose[0] - x, b.pose[1] - y) <= reach + b.robot.r_circ]

    def observe(self, i: int, normalize: bool) -> dict:
        agent, lidar = self.agents[i], self.robots[i].lidar
        if agent.task is None:
            raw = {"lidar": np.full(self.n_beams, lidar["range_max"], np.float32),
                   "goal": np.array([0, 0, 1], np.float32),
                   "velocity": np.zeros(agent.robot.action_dim, np.float32)}
        else:
            x, y, theta = agent.pose
            ranges = self.scan(i)
            if self.lidar_noise > 0:
                ranges = ranges + agent.noise_rng.normal(0, self.lidar_noise, ranges.shape)
            goal = agent.task["goal"]
            dx, dy = goal["x"] - x, goal["y"] - y
            bearing = wrap_angle(math.atan2(dy, dx) - theta)
            raw = {"lidar": np.clip(ranges, lidar["range_min"], lidar["range_max"]).astype(np.float32),
                   "goal": np.array([math.hypot(dx, dy), math.sin(bearing), math.cos(bearing)],
                                    np.float32),
                   "velocity": np.array(agent.velocity, np.float32)}
        if not normalize:
            return raw
        range_max = np.float32(lidar["range_max"])
        goal = raw["goal"].copy()
        goal[0] = min(goal[0] / range_max, 1.0)
        return {"lidar": raw["lidar"] / range_max, "goal": goal,
                "velocity": np.clip(raw["velocity"] / agent.speed_limits, -1, 1).astype(np.float32)}

    # ------------------------------------------------------------ motion

    def step(self, actions, skip_done=True) -> list:
        """Move every placed robot for ``dt``. ``actions[i]`` is the action of robot i in
        [-1, 1] (ignored for robots not placed, or already done when ``skip_done``).
        Returns, per robot, ``(terminated, truncated, info)``, or None if it did
        not take part."""
        n = self.n_robots
        velocity: list = [None] * n         # (v, vy, w) in m/s and rad/s
        moving = [False] * n
        active = [False] * n
        collided = [False] * n
        success = [False] * n
        for i, agent in enumerate(self.agents):
            if agent.task is not None and not (skip_done and agent.done):
                velocity[i] = _scale_action(agent.robot, actions[i])
                moving[i] = active[i] = True
        n_sub = max([1] + [_substeps(*velocity[i], self.dt) for i in range(n) if moving[i]])
        h = self.dt / n_sub
        collider = self.map_data.collider
        for _ in range(n_sub):
            if not any(moving):
                break
            next_pose: dict = {}
            for i in range(n):
                if not moving[i]:
                    continue
                agent = self.agents[i]
                pose = integrate(*agent.pose, *velocity[i], h)
                if not collider.pose_valid(agent.robot.footprint, *pose):
                    collided[i], moving[i] = True, False
                    continue
                next_pose[i] = pose
            if n > 1 and next_pose:
                for i in self._robot_collisions(next_pose, moving):
                    collided[i], moving[i] = True, False
                    del next_pose[i]
            for i, pose in next_pose.items():
                agent = self.agents[i]
                agent.path_length += math.hypot(pose[0] - agent.pose[0], pose[1] - agent.pose[1])
                agent.pose = pose
                # the goal is checked at EVERY sub-step: a fast robot (Jackal: 0.2 m per step,
                # the size of the goal region) could otherwise drive over it
                goal = agent.task["goal"]
                if math.hypot(goal["x"] - pose[0], goal["y"] - pose[1]) <= agent.success_radius:
                    success[i], moving[i] = True, False
        return [self._end_step(self.agents[i], velocity[i], collided[i], success[i])
                if active[i] else None for i in range(n)]

    def _robot_collisions(self, next_pose: dict, moving: list) -> set:
        """Robots whose next pose overlaps a robot that does not move in this sub-step,
        or the next pose of another moving robot."""
        still = [(b.pose, b.robot) for j, b in enumerate(self.agents)
                 if not moving[j] and b.task is not None and b.present]
        hit = set()
        for i, pose in next_pose.items():
            if any(bodies_overlap(pose, self.robots[i], q, rq) for q, rq in still):
                hit.add(i)
        movers = [i for i in next_pose if i not in hit]
        for a in range(len(movers)):
            for b in range(a + 1, len(movers)):
                i, j = movers[a], movers[b]
                if bodies_overlap(next_pose[i], self.robots[i], next_pose[j], self.robots[j]):
                    hit.update((i, j))
        return hit

    def _end_step(self, agent: Agent, velocity, collided: bool, success: bool) -> tuple:
        """Clock, info and episode end of one robot after its motion."""
        v, vy, w = velocity
        agent.velocity = (0.0,) * agent.robot.action_dim if collided else (
            (v, vy, w) if agent.robot.omni else (v, w))
        agent.time += self.dt
        agent.steps += 1
        agent.n_collisions += collided
        agent.collided, agent.reached_goal = collided, success
        x, y, theta = agent.pose
        agent.traj.append((agent.time, x, y, theta))
        info = {"collided": collided, "n_collisions": agent.n_collisions, "time": agent.time,
                "pose": agent.pose, "path_length": agent.path_length}
        if self.nav_info:
            geodesic = self.geodesic_to_goal(agent, x, y)
            progress = agent.prev_geodesic - geodesic
            if not math.isfinite(progress):     # no finite distance (should not happen)
                progress = 0.0
            agent.prev_geodesic = geodesic
            goal = agent.task["goal"]
            info.update(goal_geodesic_m=geodesic, progress_m=progress,
                        goal_distance_m=math.hypot(goal["x"] - x, goal["y"] - y),
                        wall_distance_m=self.map_data.collider.wall_distance(x, y))
        terminated = success or (collided and self.on_collision == "terminate")
        truncated = (not terminated) and agent.time >= agent.time_limit - 1e-9
        if terminated or truncated:
            agent.done = True
            if self.on_done == "remove" and self.n_robots > 1:
                agent.present = False
            info.update(self.episode_metrics(agent, success, "success" if success else
                                             "collision" if terminated else "time_limit"))
        return bool(terminated), bool(truncated), info

    def episode_metrics(self, agent: Agent, success: bool, reason: str) -> dict:
        m = self.map_data
        return {"map_id": m.map_id, "task_id": agent.task_id, "split": m.header.get("split"),
                "success": bool(success), "spl": spl(success, agent.geodesic, agent.path_length),
                "path_length": agent.path_length, "time": agent.time,
                "n_collisions": agent.n_collisions, "termination": reason,
                "geodesic_m": agent.geodesic}

    # ------------------------------------------------------------ render

    def render_rgb(self, res=0.04) -> np.ndarray:
        """Image of the map (free space white) with each robot's path, goal and body."""
        m = self.map_data
        key = round(res, 6)
        if key not in m.renders:
            mask, x0, y1, _ = m.svgmap.raster(res)
            m.renders[key] = (np.where(mask[..., None], 255, 80).astype(np.uint8).repeat(3, 2),
                              x0, y1, res)
        background, x0, y1, r = m.renders[key]
        img = background.copy()

        def to_pixels(points):
            return np.stack([(points[:, 0] - x0) / r, (y1 - points[:, 1]) / r], 1).astype(np.int32)

        one = self.n_robots == 1
        for i, agent in enumerate(self.agents):
            if agent.task is None or not agent.present:
                continue
            body = COLORS[i % len(COLORS)]
            path = np.array([[p[1], p[2]] for p in agent.traj])
            cv2.polylines(img, [to_pixels(path)], False, (80, 140, 230) if one else body, 1)
            goal = agent.task["goal"]
            centre = tuple(int(v) for v in to_pixels(np.array([[goal["x"], goal["y"]]]))[0])
            radius = max(2, int(agent.success_radius / r))
            cv2.circle(img, centre, radius, (220, 0, 0) if one else body, -1 if one else 2)
            cv2.fillPoly(img, [to_pixels(place(agent.robot.footprint, *agent.pose))], body)
        return img


def _check_choice(name: str, value, choices: tuple) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _scale_action(robot: Robot, action) -> tuple:
    """Action in [-1, 1] -> (v, vy, w) in m/s and rad/s (a negative v uses the reverse limit)."""
    act = np.clip(np.asarray(action, float), -1, 1)
    v = act[0] * (robot.v_max if act[0] >= 0 else -robot.v_min)
    vy = act[1] * robot.vy_max if robot.omni else 0.0
    return v, vy, act[-1] * robot.w_max


def _substeps(v, vy, w, dt) -> int:
    return max(1, math.ceil(max(math.hypot(v, vy) * dt / SUBSTEP_DISTANCE,
                                abs(w) * dt / SUBSTEP_ANGLE)))


def _path_turning(planner: Planner, task: dict, predecessors) -> float:
    """Total turning (rad) along the smoothed shortest path of a task."""
    start, goal = task["start"], task["goal"]
    points = ([(start["x"], start["y"])] + planner.path(predecessors, start["x"], start["y"])
              + [(goal["x"], goal["y"])])
    return turning(simplify(points, TURN_SMOOTH), start["theta"])


# ------------------------------------------------------------------ what users can ask

class RobotState:
    """Everything about one robot in the current episode, to write a reward or analyse a run.
    Get it with ``env.unwrapped.robot_state()`` (Nav-v0) or ``env.robot_state(agent)``
    (nav_v0). Distances in metres, angles in radians, times in seconds; a ``point`` is
    (x, y) and defaults to the robot's position. It always shows the latest state."""

    def __init__(self, world: World, index: int):
        self._world, self._index = world, index

    @property
    def _agent(self) -> Agent:
        agent = self._world.agents[self._index]
        if agent.task is None:
            raise RuntimeError("the robot is not on the map: reset the environment first")
        return agent

    def __repr__(self):
        a = self._world.agents[self._index]
        if a.task is None:
            return "RobotState(not placed)"
        return (f"RobotState(pose=({a.pose[0]:.2f}, {a.pose[1]:.2f}, {a.pose[2]:.2f}), "
                f"goal=({a.task['goal']['x']:.2f}, {a.task['goal']['y']:.2f}), steps={a.steps})")

    # -------------------------------------------------- now
    @property
    def robot(self) -> Robot:
        """The robot model: footprint, speed limits, LiDAR."""
        return self._world.robots[self._index]

    @property
    def pose(self) -> tuple:
        """(x, y, theta) on the map."""
        return tuple(float(v) for v in self._agent.pose)

    @property
    def position(self) -> tuple:
        """(x, y) on the map."""
        return self.pose[:2]

    @property
    def heading(self) -> float:
        """theta: the direction the robot faces, counter-clockwise from +x."""
        return self.pose[2]

    @property
    def velocity(self) -> tuple:
        """(v, w), or (v, vy, w) for an omnidirectional robot: m/s and rad/s, after the last step."""
        return tuple(float(v) for v in self._agent.velocity)

    @property
    def collided(self) -> bool:
        """Whether the last step hit a wall or another robot."""
        return self._agent.collided

    @property
    def reached_goal(self) -> bool:
        return self._agent.reached_goal

    @property
    def done(self) -> bool:
        """Whether this robot's episode is over (goal, time limit, or collision with
        ``on_collision="terminate"``)."""
        return self._agent.done

    # -------------------------------------------------- the episode
    @property
    def start(self) -> tuple:
        """(x, y, theta) at reset."""
        s = self._agent.task["start"]
        return (float(s["x"]), float(s["y"]), float(s["theta"]))

    @property
    def goal(self) -> tuple:
        """(x, y) of the goal."""
        g = self._agent.task["goal"]
        return (float(g["x"]), float(g["y"]))

    @property
    def success_radius(self) -> float:
        """The goal is reached when the robot centre is this close to it."""
        return self._agent.success_radius

    @property
    def start_geodesic(self) -> float:
        """Length of the shortest path from the start to the goal."""
        return self._agent.geodesic

    @property
    def trajectory(self) -> np.ndarray:
        """(N, 4) rows (time, x, y, theta), one per step since reset (row 0 = the start)."""
        return np.array(self._agent.traj, float)

    @property
    def path_length(self) -> float:
        """Distance driven since reset."""
        return self._agent.path_length

    @property
    def time(self) -> float:
        return self._agent.time

    @property
    def time_limit(self) -> float:
        return self._agent.time_limit

    @property
    def steps(self) -> int:
        return self._agent.steps

    @property
    def n_collisions(self) -> int:
        """Steps with a collision since reset."""
        return self._agent.n_collisions

    # -------------------------------------------------- questions
    def _point(self, point):
        return self.position if point is None else (float(point[0]), float(point[1]))

    def goal_distance(self, point=None) -> float:
        """Straight-line distance from ``point`` to the goal."""
        x, y = self._point(point)
        gx, gy = self.goal
        return math.hypot(gx - x, gy - y)

    def geodesic_distance(self, point=None) -> float:
        """Length of the shortest drivable path from ``point`` to the goal (inf if unreachable)."""
        return self._world.geodesic_to_goal(self._agent, *self._point(point))

    def shortest_path(self, point=None) -> np.ndarray:
        """The shortest drivable path from ``point`` to the goal: (M, 2) points (x, y)."""
        return self._world.shortest_path(self._agent, *self._point(point))

    def wall_distance(self, point=None) -> float:
        """Distance from ``point`` to the nearest wall."""
        return self._world.map_data.collider.wall_distance(*self._point(point))

    def lidar(self) -> np.ndarray:
        """LiDAR distances without noise (n_beams,), in metres, from the current pose; the
        beam directions relative to the heading are ``beam_angles``."""
        lidar = self.robot.lidar
        return np.clip(self._world.scan(self._index), lidar["range_min"], lidar["range_max"])

    @property
    def beam_angles(self) -> np.ndarray:
        return self._world.agents[self._index].beam_angles

    def fits(self, x: float, y: float, theta: float) -> bool:
        """Whether the robot's body fits at (x, y, theta) without touching a wall."""
        return self._world.map_data.collider.pose_valid(self.robot.footprint, x, y, theta)
