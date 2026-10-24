"""Measure how well a policy navigates.

    navsim.evaluate(model, "NavSim/Nav-v0", dataset="isb-svg-v1", robot="turtlebot4")  # every test task
    navsim.evaluate(model, env, episodes=100)          # an env you built yourself (wrappers kept)
    navsim.evaluate(policy, nav_v0.parallel_env(n_agents=4, split="test"), episodes=50)

``policy`` is one of:
    - a function ``policy(obs) -> action``
    - a model with ``predict(obs, deterministic=...)`` (Stable-Baselines3)
    - ``"random"``
With several robots the same policy drives every robot, or pass ``{agent: policy}``.
A policy with a ``reset()`` method is reset at the start of every episode.

``deterministic`` (models only): None = the most likely action, except on
``NavSim/GridNav-v0`` where actions are sampled. On the grid the observation depends only on
the cell, so "always the most likely action" means one fixed move per cell, and two cells
pointing at each other trap the robot until the budget runs out: the score can be far below
that of the policy you trained. Sampled results vary a little from run to run.

Which episodes: by default every task of every map of the environment, in order
(``per_map`` caps the tasks per map); or ``episodes`` random episodes (seeds seed, seed + 1,
...). Several robots: ``episodes`` random episodes (default 20).

The summary: ``success`` (share of episodes that reached the goal), ``spl`` (success
weighted by how short the path was), ``collision_episodes`` (share with a collision),
``mean_progress`` (share of the shortest path covered: 1 on success, negative if the robot
ended farther away), ``mean_time``, ``mean_path_m``; with several robots also
``team_success`` (every robot reached its goal).
"""

from __future__ import annotations

import sys
import time

import gymnasium as gym
import numpy as np


def evaluate(policy, env="NavSim/Nav-v0", *, split="test", episodes: int | None = None,
             per_map: int | None = None, seed: int = 0, deterministic: bool | None = None,
             progress: bool = False, return_episodes: bool = False, **env_kwargs) -> dict:
    """Run ``policy`` and return the summary. ``env`` is an env id (made with ``split`` and
    ``env_kwargs``), a Gymnasium env, or a PettingZoo ParallelEnv. ``return_episodes``: also
    return the end-of-episode infos (``"episode_infos"``). ``progress``: print progress."""
    if isinstance(env, str):
        env = gym.make(env, split=split, **env_kwargs)
    elif env_kwargs:
        raise TypeError(f"environment arguments {sorted(env_kwargs)} need an env id, not an env")
    if deterministic is None:
        deterministic = default_deterministic(env)
    if hasattr(env, "possible_agents"):
        out = _evaluate_team(policy, env, episodes or 20, seed, deterministic, progress,
                             return_episodes)
        return _with_mode(out, policy, deterministic)
    act = as_policy(policy, env.action_space, deterministic)
    env.action_space.seed(seed)
    u = env.unwrapped
    if episodes is None:
        plan = [{"map_id": m, "task_idx": i} for m in u.map_ids
                for i in range(min(len(u.tasks_of(m)), per_map or len(u.tasks_of(m))))]
    else:
        plan = [None] * int(episodes)
    ends, t0 = [], time.perf_counter()
    for e, options in enumerate(plan):
        obs, _ = env.reset(seed=seed + (0 if options else e), options=options)
        reset_policy(policy)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(act(obs))
            done = terminated or truncated
        ends.append(info)
        if progress and (e + 1) % 20 == 0:
            print(f"  {e + 1}/{len(plan)} episodes, success {np.mean([r['success'] for r in ends]):.2f} "
                  f"({time.perf_counter() - t0:.0f} s)", file=sys.stderr, flush=True)
    out = {"split": getattr(u, "split", None) or "all", **summarize(ends)}
    if return_episodes:
        out["episode_infos"] = ends
    return _with_mode(out, policy, deterministic)


def summarize(ends: list[dict]) -> dict:
    """The summary of a list of end-of-episode infos."""
    if not ends:
        raise ValueError("no episodes (empty split?)")
    return {"episodes": len(ends),
            "success": float(np.mean([r["success"] for r in ends])),
            "spl": float(np.mean([r["spl"] for r in ends])),
            "collision_episodes": float(np.mean([r["n_collisions"] > 0 for r in ends])),
            "mean_progress": _mean([_progress(r) for r in ends]),
            "mean_time": float(np.mean([r["time"] for r in ends])),
            "mean_path_m": float(np.mean([r["path_length"] for r in ends]))}


def as_policy(policy, action_space, deterministic):
    """``policy`` (function, model with ``predict``, or "random") as a function obs -> action."""
    if isinstance(policy, str):
        if policy != "random":
            raise ValueError("policy must be a callable, an object with predict(), or 'random'")
        return lambda obs: action_space.sample()
    if hasattr(policy, "predict"):
        return lambda obs: policy.predict(obs, deterministic=deterministic)[0]
    if callable(policy):
        return policy
    raise TypeError(f"cannot use {type(policy).__name__} as a policy")


def default_deterministic(env) -> bool:
    """The default of ``deterministic``: True, except on the grid (see the module docstring)."""
    from .envs.grid import GridNavEnv
    return not isinstance(getattr(env, "unwrapped", env), GridNavEnv)


def reset_policy(policy) -> None:
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset()


def _with_mode(out: dict, policy, deterministic: bool) -> dict:
    """Add ``deterministic`` when it applied: a model (or a dict of them) with ``predict``."""
    policies = policy.values() if isinstance(policy, dict) else [policy]
    if any(hasattr(p, "predict") for p in policies):
        out["deterministic"] = bool(deterministic)
    return out


def _progress(info: dict) -> float:
    """Share of the start's shortest-path distance covered at the end of an episode."""
    if info.get("success"):
        return 1.0
    if "goal_geodesic" in info:                           # grid: steps
        start, end = info.get("d_bfs"), info["goal_geodesic"]
    else:                                                 # continuous (absent with nav_info=False)
        start, end = info.get("geodesic_m"), info.get("goal_geodesic_m")
    if not start or end is None or not np.isfinite(end):
        return float("nan")
    return 1.0 - end / start


def _mean(values) -> float | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def _evaluate_team(policy, env, episodes, seed, deterministic, progress, return_episodes):
    agents = env.possible_agents
    policies = {a: as_policy(policy[a] if isinstance(policy, dict) else policy, env.action_space(a),
                             deterministic) for a in agents}
    for a in agents:
        env.action_space(a).seed(seed + agents.index(a))
    ends, team, t0 = [], [], time.perf_counter()
    for e in range(int(episodes)):
        obs, _ = env.reset(seed=seed + e)
        for p in (policy.values() if isinstance(policy, dict) else [policy]):
            reset_policy(p)
        last = {}
        while env.agents:
            obs, _, terms, truncs, infos = env.step({a: policies[a](obs[a]) for a in env.agents})
            for a in infos:
                if terms[a] or truncs[a]:
                    last[a] = infos[a]
        ends += [last[a] for a in agents]
        team.append(all(last[a]["success"] for a in agents))
        if progress and (e + 1) % 5 == 0:
            print(f"  {e + 1}/{episodes} episodes ({time.perf_counter() - t0:.0f} s)",
                  file=sys.stderr, flush=True)
    out = {"split": getattr(env, "split", None) or "all", "robots": len(agents),
           "team_success": float(np.mean(team)), **summarize(ends)}
    out["robot_episodes"] = out.pop("episodes")
    out["episodes"] = int(episodes)
    if return_episodes:
        out["episode_infos"] = ends
    return out
