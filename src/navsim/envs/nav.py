"""``NavSim/Nav-v0``: one robot drives from a start to a goal on a real floor plan.

    env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")

The robot has its real size (footprint), speed limits and LiDAR; distances are in metres.

Action ``Box(-1, 1, (2,))``: (forward speed, turning speed) as a fraction of the robot's
limits; ``(3,)`` = (forward, sideways, turning) for an omnidirectional robot (jetauto_pro).

Observation ``Dict``:
    lidar     ``n_beams`` distances measured by the LiDAR
    goal      (distance, sin, cos) of the goal direction, seen from the robot
    velocity  the current (v, [vy], w)
By default (``normalize_obs=True``) every value is scaled for learning: lidar and goal
distance divided by the LiDAR range (goal distance capped at 1), velocity divided by the
robot's limits. ``normalize_obs=False`` gives metres, m/s and rad/s.

Reward: always 0.0. The environment has no reward of its own: write yours with
``navsim.wrappers.Reward(env, fn)`` from ``env.unwrapped.robot_state()`` (pose, goal,
trajectory, shortest path, ...) and the step info.

Episode end: ``terminated`` when the robot centre is within ``success_radius`` of the goal
(or, with ``on_collision="terminate"``, at a collision; by default a collision only stops
the robot); ``truncated`` after ``time_factor`` x the least time the trip needs.

``info`` at every step: collided, n_collisions, time, pose, path_length, and (unless
``nav_info=False``) goal_geodesic_m, progress_m (how much goal_geodesic_m shrank in this
step), goal_distance_m, wall_distance_m. At the end of an episode also: map_id, task_id,
split, success, spl, termination ("success" / "collision" / "time_limit"), geodesic_m.

Choosing the map and task, curriculum: see :mod:`navsim.envs.episodes`.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .._sim.world import MapLoader, RobotState, World, new_map_data
from ..robots import Robot, load as load_robot
from .episodes import EpisodeSource, custom_endpoints


class NavEnv(EpisodeSource, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, dataset="demo-svg", robot=None, split=None, map_id=None, task_idx=None,
                 normalize_obs=True, dt=0.1, n_beams=72, lidar_noise=0.0,
                 on_collision="stop", time_limit=None, time_factor=3.0,
                 success_radius=None, task_filter=None, nav_info=True, render_mode=None,
                 map_cache=8, field_cache=4, map_repeat=1, disk_cache=True, clearance=None,
                 svgmap=None, tasks=None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"render_mode must be None or one of {self.metadata['render_modes']}")
        self.split, self.normalize_obs, self.render_mode = split, normalize_obs, render_mode
        self.metadata = {**self.metadata, "render_fps": max(1, round(1 / dt))}   # real time
        episode_args = (map_id, task_idx, map_cache, map_repeat, task_filter)
        if tasks is not None:                       # one map given in memory (tests)
            self.dataset = None
            self.robot = robot if isinstance(robot, Robot) else load_robot(robot or "turtlebot4")
            self.robot_id = self.robot.id
            name = tasks.get("map_id", "map")
            data = new_map_data(name, svgmap, tasks, clearance, self.robot.r_circ, field_cache)
            self._init_episodes([name], lambda m: data, *episode_args)
        else:
            from ..data import load_dataset
            self.dataset = ds = load_dataset(dataset)
            if ds.env_type != "svg":
                raise ValueError(f"dataset {ds.name} is a {ds.env_type} dataset: use NavSim/GridNav-v0")
            self.robot_id = ds.default_robot(robot.id if isinstance(robot, Robot) else robot)
            self.robot = robot if isinstance(robot, Robot) else ds.robot(self.robot_id)
            maps = [map_id] if map_id else ds.maps(self.robot_id, split)
            if not maps:
                raise ValueError(f"no {split} maps for {self.robot_id} in {ds.name}")
            loader = MapLoader(ds, self.robot_id, self.robot.r_circ, clearance, field_cache, disk_cache)
            self._init_episodes(maps, loader, *episode_args)
        self.world = World([self.robot], dt=dt, n_beams=n_beams, lidar_noise=lidar_noise,
                           on_collision=on_collision, time_limit=time_limit,
                           time_factor=time_factor, success_radius=success_radius,
                           nav_info=nav_info)
        self.action_space = self.world.action_space(0)
        self.observation_space = self.world.observation_space(0, normalize_obs)
        self.map_id = None

    def tasks_of(self, map_id) -> list:
        if self.dataset is None:
            return super().tasks_of(map_id)
        return self.dataset.tasks(self.robot_id, map_id)

    # ------------------------------------------------------------ current state

    def robot_state(self) -> RobotState:
        """Everything about the robot now: pose, goal, trajectory, shortest path, distances
        (see ``navsim._sim.world.RobotState``). Use it to write a reward."""
        return RobotState(self.world, 0)

    @property
    def agent(self):
        """The robot's state in the simulator (pose, velocity, task, path, ...)."""
        return self.world.agents[0]

    @property
    def pose(self) -> tuple:
        """(x, y, theta) of the robot on the map (m, rad)."""
        return self.agent.pose

    @property
    def task(self) -> dict:
        """The current task: {"start": {x, y, theta}, "goal": {x, y}, "geodesic_m", ...}."""
        return self.agent.task

    @property
    def dt(self) -> float:
        return self.world.dt

    @property
    def beam_angles(self) -> np.ndarray:
        """Direction of each LiDAR beam relative to the robot's heading (rad)."""
        return self.agent.beam_angles

    @property
    def map(self):
        """The current map (an SvgMap)."""
        return self.world.map_data.svgmap

    @property
    def tasks(self) -> list:
        """The tasks of the current map."""
        return self.world.map_data.tasks

    # ------------------------------------------------------------ Gymnasium API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._on_seed(seed)
        agent = self.agent
        if seed is not None or agent.noise_rng is None:
            # the LiDAR noise gets its own random stream, derived from the episode seed
            agent.noise_rng = np.random.default_rng(self.np_random.bit_generator.seed_seq.spawn(1)[0])
        self.map_id, data, task_idx = self._choose(options)
        self.world.set_map(data)
        custom = custom_endpoints(options)
        if custom:
            task, field_key, turn_key = self.world.custom_task(0, *custom)
            task_idx = -1
        else:
            task, field_key, turn_key = data.tasks[task_idx], task_idx, task_idx
        try:
            info = self.world.place(0, task, task_idx, field_key, turn_key)
        except ValueError as e:
            raise RuntimeError(str(e)) from None
        info["split"] = info["split"] or self.split
        if self.render_mode == "human":
            self.render()
        return self.world.observe(0, self.normalize_obs), info

    def step(self, action):
        if self.map_id is None:
            raise gym.error.ResetNeeded("call reset() before step()")
        terminated, truncated, info = self.world.step([action], skip_done=False)[0]
        if info.get("split", "") is None:
            info["split"] = self.split
        if self.render_mode == "human":
            self.render()
        return self.world.observe(0, self.normalize_obs), 0.0, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.world.render_rgb()
        if self.render_mode == "human":
            from ..visualize import show_frame
            show_frame(self, self.world.render_rgb())
        return None

    def close(self):
        from ..visualize import close_window
        close_window(self)
