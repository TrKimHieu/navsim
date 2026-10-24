"""Wrappers.

    env = navsim.wrappers.Reward(env, fn)              # your reward: fn(state, info) -> float
    env = navsim.wrappers.DiscreteActions(env)         # Nav-v0: forward 0.25 m, turn 15 deg, ...
    team = navsim.wrappers.ParallelReward(team, fn)    # nav_v0: fn(state, info) for every robot

Gymnasium's own wrappers work too, e.g. ``gymnasium.wrappers.FlattenObservation`` turns the
Dict observation of Nav-v0 into one flat vector.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


class DiscreteActions(gym.Wrapper):
    """Discrete actions for ``NavSim/Nav-v0`` (like Habitat): 0 wait one dt, 1 forward
    ``step_m``, 2 backward ``step_m``, 3 turn left ``turn_deg``, 4 turn right ``turn_deg``,
    and for omnidirectional robots 5 / 6 move left / right ``step_m``.

    Each action is driven at the robot's top speed over as many dt steps as it takes, and
    stops early at a collision or at the end of the episode. The reward and
    ``info["progress_m"]`` are the sums over those steps; ``info["macro_steps"]`` says how
    many there were."""

    def __init__(self, env, step_m=0.25, turn_deg=15.0):
        super().__init__(env)
        robot = env.unwrapped.robot
        self.step_m, self.turn = step_m, math.radians(turn_deg)
        self.action_space = gym.spaces.Discrete(7 if robot.omni else 5)

    def _plan(self, action) -> list:
        """The list of continuous actions (one per dt) that carry out a discrete action."""
        u = self.env.unwrapped
        robot, dt, n = u.robot, u.dt, u.robot.action_dim
        zero = np.zeros(n, np.float32)
        if action == 0:
            return [zero]
        sign = 1 if action in (1, 3, 5) else -1
        if action in (1, 2):         # forward / backward
            speed, amount, axis = (robot.v_max if action == 1 else -robot.v_min), self.step_m, 0
        elif action in (3, 4):       # turn left / right
            speed, amount, axis = robot.w_max, self.turn, n - 1
        else:                        # sideways left / right
            speed, amount, axis = robot.vy_max, self.step_m, 1
        per_step = speed * dt
        n_full, rest = int(amount // per_step), amount % per_step
        plan = []
        for fraction in [1.0] * n_full + ([rest / per_step] if rest > 1e-9 else []):
            a = zero.copy()
            a[axis] = sign * fraction
            plan.append(a)
        return plan

    def step(self, action):
        total, progress, n = 0.0, 0.0, 0
        for a in self._plan(int(action)):
            obs, reward, terminated, truncated, info = self.env.step(a)
            total += reward
            progress += info.get("progress_m", 0.0)
            n += 1
            if terminated or truncated or info["collided"]:
                break
        if "progress_m" in info:                 # absent with nav_info=False
            info["progress_m"] = progress
        info["macro_steps"] = n
        return obs, total, terminated, truncated, info


class Reward(gym.Wrapper):
    """Give the environment your reward: after each step, ``reward = fn(state, info)`` where
    ``state = env.unwrapped.robot_state()`` (pose, goal, trajectory, shortest path, ...) and
    ``info`` is the step info. Examples::

        # Nav-v0: metres gained along the shortest path, +10 at the goal, -0.1 per collision
        def reward(state, info):
            return info["progress_m"] + 10 * state.reached_goal - 0.1 * state.collided
        env = navsim.wrappers.Reward(gym.make("NavSim/Nav-v0"), reward)

        # GridNav-v0: +0.1 per move closer to the goal, +1 at the goal
        env = navsim.wrappers.Reward(grid_env, lambda s, i: 0.1 * i["progress"] + s.reached_goal)

    Put it outside ``DiscreteActions`` to get one reward per discrete action.
    """

    def __init__(self, env, fn):
        super().__init__(env)
        self.fn = fn

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = float(self.fn(self.env.unwrapped.robot_state(), info))
        return obs, reward, terminated, truncated, info


def __getattr__(name):
    if name == "ParallelReward":                 # needs PettingZoo: imported on first use
        from .multiagent.nav_v0 import ParallelReward
        return ParallelReward
    raise AttributeError(f"module 'navsim.wrappers' has no attribute {name!r}")
