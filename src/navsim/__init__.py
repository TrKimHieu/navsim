"""NavSim: robot navigation environments for reinforcement learning, on real floor plans.

One robot (Gymnasium)::

    import gymnasium as gym
    import navsim

    env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
    env = gym.make("NavSim/GridNav-v0", dataset="demo-grid", goal_obs=True)
    navsim.setup("isb-svg-v1")              # a full dataset: download once

Several robots on one map (PettingZoo)::

    from navsim.multiagent import nav_v0
    env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=4)

Evaluate and watch::

    navsim.evaluate(model, "NavSim/Nav-v0", dataset="demo-svg")      # every test task
    navsim.record_video(env, model)         # GIF
    navsim.plot_episode(env)                # map, path, robot, LiDAR

Where things are::

    envs/          the environments (nav.py: Nav-v0, grid.py: GridNav-v0, episodes.py: map/task choice)
    multiagent/    several robots (nav_v0.py)
    data/          datasets: find and read (dataset.py), download (download.py), Colab/Kaggle (platform.py)
    robots/        robot models (presets/*.json)
    evaluation.py  evaluate()
    visualize.py   plot_episode(), render_frame(), save_gif(), record_video()
    wrappers.py    Reward (your reward), ParallelReward, DiscreteActions
    cli.py         the `navsim` command
    _sim/          the simulator itself (internal)
"""

import multiprocessing
from functools import partial

from gymnasium.envs.registration import register

__version__ = "0.1.0"

register(id="NavSim/Nav-v0", entry_point="navsim.envs.nav:NavEnv")
register(id="NavSim/GridNav-v0", entry_point="navsim.envs.grid:GridNavEnv")

from . import envs, wrappers  # noqa: E402,F401
from .data import Dataset, list_datasets, load_dataset, setup  # noqa: E402,F401
from .evaluation import evaluate  # noqa: E402,F401
from .robots import list_robots, load as load_robot  # noqa: E402,F401
from .visualize import plot_episode, record_video, render_frame, save_gif  # noqa: E402,F401

__all__ = ["setup", "load_dataset", "list_datasets", "Dataset", "list_robots", "load_robot",
           "evaluate", "plot_episode", "render_frame", "save_gif", "record_video", "env_fn", "wrappers",
           "__version__"]


def env_fn(env_id: str = "NavSim/Nav-v0", wrappers=(), **kwargs):
    """A function that makes the environment, for libraries that create it in worker
    processes, e.g. Stable-Baselines3::

        venv = make_vec_env(navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg"), n_envs=8,
                            vec_env_cls=SubprocVecEnv)

    It is ``() -> gym.make(env_id, **kwargs)`` with each of ``wrappers`` applied, and it
    imports navsim in the worker first: a worker started fresh (Windows, macOS) does not
    know the NavSim ids otherwise. On Linux (Colab, Kaggle) plain ids work too, see
    ``_preload_in_forkserver``."""
    return partial(_make_env, env_id, tuple(wrappers), kwargs)


def _make_env(env_id, wrappers, kwargs):
    import gymnasium as gym
    env = gym.make(env_id, **kwargs)
    for wrapper in wrappers:
        env = wrapper(env)
    return env


def _preload_in_forkserver() -> None:
    """Stable-Baselines3's SubprocVecEnv on Linux starts its workers from a "forkserver"
    process that never ran your ``import navsim``, so ``gym.make("NavSim/Nav-v0")`` would
    fail in them. Ask the forkserver to import navsim too (keeping its other modules).
    No effect where there is no forkserver, or once it is running."""
    if "forkserver" not in multiprocessing.get_all_start_methods():
        return
    from multiprocessing import forkserver
    modules = list(getattr(forkserver._forkserver, "_preload_modules", ["__main__"]))
    if "navsim" not in modules:
        forkserver.set_forkserver_preload(modules + ["navsim"])


_preload_in_forkserver()
