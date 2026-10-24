<div align="center">

# NavSim

**Robot navigation environments for reinforcement learning, on real building floor plans.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Gymnasium](https://img.shields.io/badge/API-Gymnasium-0081A5)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/multi--agent-PettingZoo-orange)](https://pettingzoo.farama.org/)
[![CPU only](https://img.shields.io/badge/runs%20on-CPU-lightgrey)](#install)

English · [Tiếng Việt](README.vi.md)

</div>

A robot drives from a start to a goal in a real floor plan, with the real size, speed limits and
LiDAR of a real robot. One robot uses the [Gymnasium](https://gymnasium.farama.org/) API; several
robots on one map use [PettingZoo](https://pettingzoo.farama.org/).

NavSim is **only the environment**: it has no reward and no training code. `step()` returns a reward
of `0.0`; you write your own reward from the robot's state (pose, goal, trajectory, shortest path,
distances, ...), then train with any library.

## Highlights

- **Three environments**: continuous floor plans (`Nav-v0`), grid worlds (`GridNav-v0`), several robots on one map (`nav_v0`).
- **Real robots**: nine models (TurtleBot 3/4, Jackal, TIAGo, Kobuki, ...) with footprint, speed limits and LiDAR; differential and omnidirectional drives.
- **Real maps**: HM3D-based buildings, split into train / val / test by building. Two small demos ship inside the package.
- **Bring your own reward**: wrappers and a rich `RobotState` (geodesic distance, shortest path, wall distance, trajectory, ...).
- **Curriculum**: filter tasks by path length or difficulty labels at any time.
- **Batteries included**: `evaluate()` (success, SPL, collisions), pictures and GIFs, parallel environments, a `navsim` command line.
- **Fast and deterministic**: CPU only, optional numba kernels (about 3× faster, identical results), disk cache of loaded maps.

## Contents

[Install](#install) · [Quick start](#quick-start) · [Environments](#environments) ·
[Choosing episodes](#choosing-episodes-and-curriculum) · [Rewards](#robot-state-and-rewards) ·
[Evaluation](#evaluation) · [Pictures and GIFs](#pictures-and-gifs) ·
[Parallel environments](#parallel-environments) · [Datasets](#datasets) · [Robots](#robots) ·
[Command line](#command-line) · [Documentation](#documentation) · [Code layout](#code-layout) ·
[Development](#development) · [License](#license)

## Install

NavSim is one wheel file, `navsim-<version>-py3-none-any.whl` (Python ≥ 3.10).

| Where | How |
|---|---|
| **Computer** | `pip install "navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Colab** | put the `navsim` folder (wheel + README) at the root of your Google Drive, then in a cell: `from google.colab import drive; drive.mount("/content/drive")` and `!pip install -q "/content/drive/MyDrive/navsim/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Kaggle** | *Add Input → Upload* the wheel as a private Dataset, then `!pip install -q "/kaggle/input/<name>/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |

| Extra | Adds | For |
|---|---|---|
| `hub` | `huggingface_hub` | downloading datasets |
| `marl` | `pettingzoo` | several robots |
| `fast` | `numba` | about 3× faster, same results |
| `plot` | `matplotlib` | pictures and GIFs in `"plot"` style |
| `all` | all of the above | |

> **Notes**
> - If the notebook has numpy 1.x, add `"numpy==<its version>"` to the same `pip install`, otherwise pip upgrades numpy (recent OpenCV needs numpy 2) and the kernel must restart.
> - Install only from the wheel: `pip install navsim` installs an unrelated PyPI package with the same name.

More: [Getting started](docs/guide/getting-started.md) (Vietnamese).

## Quick start

```python
import gymnasium as gym
import navsim

env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
env = navsim.wrappers.Reward(env, lambda state, info: info["progress_m"] + 10 * state.reached_goal)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

state = env.unwrapped.robot_state()
state.pose, state.goal, state.geodesic_distance(), state.shortest_path()
navsim.plot_episode(env, shortest_path=True, save="episode.png")
```

`import navsim` is required: it registers the `NavSim/...` ids with Gymnasium.

## Environments

| | [`NavSim/Nav-v0`](docs/environments/nav_v0.md) | [`NavSim/GridNav-v0`](docs/environments/grid_nav_v0.md) | [`navsim.multiagent.nav_v0`](docs/environments/multiagent_nav_v0.md) |
|---|---|---|---|
| World | floor plan, metres | grid, one cell = one robot body | floor plan, several robots |
| API | Gymnasium | Gymnasium | PettingZoo `ParallelEnv` (and AEC) |
| Make | `gym.make("NavSim/Nav-v0", ...)` | `gym.make("NavSim/GridNav-v0", ...)` | `nav_v0.parallel_env(...)` |
| Action | `Box(-1, 1, (2,))`, `(3,)` if omnidirectional | `Discrete(4)` | per robot, as `Nav-v0` |
| Observation | `Dict(lidar, goal, velocity)` | LiDAR (+ goal with `goal_obs=True`) | per robot, as `Nav-v0` |
| Reward | 0.0 | 0.0 | 0.0 for every robot |

`step()` returns `(obs, reward, terminated, truncated, info)` (PettingZoo: one dict per item, keyed by
agent). `reset(seed=None, options=None)` returns `(obs, info)`. Each environment has its own page in
the [documentation](docs/environments/README.md), with its action and observation spaces, episode
end, `info` keys and every argument.

```python
# several robots on one map (needs the `marl` extra)
from navsim.multiagent import nav_v0

env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=4, split="train")
obs, infos = env.reset(seed=0)
while env.agents:
    obs, rewards, terminations, truncations, infos = env.step(
        {a: env.action_space(a).sample() for a in env.agents})
```

## Choosing episodes and curriculum

| Call | Effect |
|---|---|
| `env.reset(seed=s)` | a random map of the split, then a random task of it (seeded) |
| `env.reset(options={"map_id": m, "task_idx": i})` | task `i` of map `m` |
| `env.reset(options={"map_id": m, "start": (x, y[, theta]), "goal": (x, y)})` | your own start and goal (checked; grid: `(row, col)` cells) |
| `env.unwrapped.set_task_filter(geodesic_m=(1.0, 5.0))` | only tasks matching every criterion; no argument removes the filter |

Criteria are looked up in the task (`geodesic_m`, `euclidean_m`), then in its `labels` (svg: `gdr`;
grid: `d_bfs`, `level`, ...). In a vector environment use `venv.call("set_task_filter", ...)`.
Details: [Choosing episodes and curriculum](docs/concepts/episodes.md).

## Robot state and rewards

`state = env.unwrapped.robot_state()` (`Nav-v0`, `GridNav-v0`) or `state = env.robot_state(agent)`
(`nav_v0`) always shows the latest state, so you can keep it.

| Member | Meaning |
|---|---|
| `pose`, `position`, `heading`, `velocity` | where the robot is and how it moves |
| `start`, `goal`, `trajectory`, `path_length` | the episode so far |
| `collided`, `n_collisions`, `reached_goal`, `done` | what happened |
| `goal_distance()`, `geodesic_distance()`, `shortest_path()` | straight line, shortest drivable distance and path to the goal (from any point) |
| `wall_distance()`, `lidar()`, `fits(x, y, theta)` | surroundings |

Full table (metres and radians vs grid cells): [Robot state](docs/concepts/robot_state.md).

```python
# progress along the shortest path, +10 at the goal, -0.1 per collision, -0.01 per step
def dense(state, info):
    return info["progress_m"] + 10 * state.reached_goal - 0.1 * state.collided - 0.01

env = navsim.wrappers.Reward(gym.make("NavSim/Nav-v0"), dense)
team = navsim.wrappers.ParallelReward(nav_v0.parallel_env(n_agents=4), dense)   # per robot
```

| Wrapper | What |
|---|---|
| `navsim.wrappers.Reward(env, fn)` | reward = `fn(state, info)` |
| `navsim.wrappers.ParallelReward(env, fn)` | the same for `nav_v0`, per robot |
| `navsim.wrappers.DiscreteActions(env, step_m=0.25, turn_deg=15.0)` | `Nav-v0` with `Discrete(5)`: wait, forward, backward, turn left, turn right |
| `gymnasium.wrappers.FlattenObservation(env)` | one flat vector instead of the `Dict` |

More: [Wrappers](docs/api/wrappers.md).

## Evaluation

```python
navsim.evaluate(model, "NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4")   # every test task
navsim.evaluate("random", nav_v0.parallel_env(n_agents=4, split="test"), episodes=50)
```

`policy` is a Stable-Baselines3 model, a function `obs -> action`, `"random"`, or (several robots) a dict
`{agent: policy}`. It returns `success`, `spl`, `collision_episodes`, `mean_progress`, `mean_time`,
`mean_path_m` (several robots: also `team_success`). More: [Evaluation](docs/api/evaluation.md).

## Pictures and GIFs

| Function | What |
|---|---|
| `navsim.plot_episode(env, ..., save="x.png")` | map, path, start, goal, robot, LiDAR in a matplotlib `Axes` |
| `navsim.render_frame(env, style="plot" \| "fast")` | `(H, W, 3)` image of the current step |
| `navsim.save_gif(frames, path, fps=10)` | a list of images → animated GIF |
| `navsim.record_video(env, policy, path="videos/episode.gif", ...)` | run a policy and save a GIF |

They work with all three environments, at any step, and never change the episode. A `"plot"` frame
takes about 0.1 s: for long episodes use `every=5` or `style="fast"`. More:
[Pictures and GIFs](docs/api/visualization.md).

## Parallel environments

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

venv = make_vec_env(navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg",
                                  wrappers=[lambda e: navsim.wrappers.Reward(e, dense)]),
                    n_envs=8, vec_env_cls=SubprocVecEnv)
```

`navsim.env_fn` works everywhere. A plain id (`make_vec_env("NavSim/Nav-v0", ...)`) also works on Linux
but fails on Windows and macOS with `Namespace NavSim not found`. In a `.py` script, put this code under
`if __name__ == "__main__":`. More: [Parallel environments](docs/api/parallel.md).

## Datasets

| Call | Returns |
|---|---|
| `navsim.setup(datasets="demo-svg", token=None, data_dir=None, force=False)` | a `Dataset`; downloads once from Hugging Face if it is not on the machine |
| `navsim.load_dataset(name_or_path)` | a `Dataset` |
| `navsim.list_datasets()` | the datasets on this machine |

Two demos are inside the package: `demo-svg` (floor plans: turtlebot4, clearpath_jackal) and
`demo-grid` (grids: jetauto_pro), each with 3 train / 2 val / 1 test buildings. Full datasets are
downloaded from Hugging Face:

1. Request access on [the dataset page](https://huggingface.co/datasets/PhysicalSI/2D_Navigation_Sim).
2. Save a *Read* token as a secret named `HF_TOKEN` (Colab: key icon, *Notebook access* on; Kaggle: *Add-ons → Secrets*, *Internet* on; computer: `export HF_TOKEN=...`).
3. `navsim.setup("isb-svg-v1")`.

A dataset name is looked for in: `$NAVSIM_DATASETS`, `~/.navsim/datasets`, Kaggle inputs
(`/kaggle/input/*/<name>`), the demos. More: [Datasets](docs/api/datasets.md).

## Robots

`navsim.list_robots()` → ids; `navsim.load_robot(id_or_json_path)` → `Robot` (`footprint`, `r_circ`,
`v_max`, `v_min`, `vy_max`, `w_max`, `lidar`, `drive`, `height_class`, ...).

| Robot | Drive | Robot | Drive |
|---|---|---|---|
| `agilex_limo` | differential | `turtlebot3_burger` | differential |
| `clearpath_jackal` | differential | `turtlebot3_waffle_pi` | differential |
| `jetauto_pro` | omnidirectional | `turtlebot4` | differential |
| `kobuki` | differential | `turtlebot4_lite` | differential |
| `pal_tiago` | differential | | |

`navsim robots` prints their sizes and limits. Add your own robot with a JSON file:
[Robots](docs/api/robots.md).

## Command line

```bash
navsim setup isb-svg-v1        # download a dataset
navsim datasets                # datasets on this machine
navsim robots                  # robot models
navsim info demo-svg           # maps and tasks per robot and split
navsim check                   # Gymnasium / PettingZoo API checks and speed
navsim cache --clear           # remove the map cache
```

Also `python -m navsim ...`. Environment variables (`HF_TOKEN`, `NAVSIM_HOME`, `NAVSIM_DATASETS`,
`NAVSIM_ROBOTS`, `NAVSIM_CACHE`, `NAVSIM_ACCEL`): [Command line](docs/api/cli.md),
[Environment variables](docs/api/environment-variables.md).

## Documentation

The full documentation is in [`docs/`](docs/README.md) (Vietnamese). Start from the page that matches
what you want to do:

| I want to... | Read |
|---|---|
| install and run a first example | [Getting started](docs/guide/getting-started.md) |
| pick and use an environment | [Environments](docs/environments/README.md): [Nav-v0](docs/environments/nav_v0.md), [GridNav-v0](docs/environments/grid_nav_v0.md), [nav_v0](docs/environments/multiagent_nav_v0.md) |
| choose maps and tasks, use a curriculum | [Choosing episodes](docs/concepts/episodes.md) |
| write a reward | [Robot state](docs/concepts/robot_state.md), [Wrappers](docs/api/wrappers.md) |
| evaluate, draw, record | [Evaluation](docs/api/evaluation.md), [Pictures and GIFs](docs/api/visualization.md) |
| train in parallel | [Parallel environments](docs/api/parallel.md), [Recipes](docs/guide/recipes.md) |
| fix an error | [Troubleshooting](docs/guide/troubleshooting.md) |
| browse every command as a tree | [`docs/navsim-api-tree.html`](docs/navsim-api-tree.html) (open in a browser) |
| understand how it works inside | [Overview](docs/internals/overview.md), [Dataset format](docs/internals/dataset-format.md), [LiDAR](docs/internals/lidar.md), [A `step()` and planning](docs/internals/step-and-planning.md), [Architecture](docs/internals/architecture.md), [Simulator](docs/internals/simulator.md) |

## Code layout

```text
src/navsim/
  envs/          nav.py (Nav-v0), grid.py (GridNav-v0, GridRobotState), episodes.py (map and task choice)
  multiagent/    nav_v0.py (several robots, ParallelReward)
  data/          dataset.py (find, read), download.py (setup), platform.py (Colab / Kaggle token)
  robots/        robot models (presets/*.json)
  evaluation.py  evaluate()
  visualize.py   plot_episode(), render_frame(), save_gif(), record_video()
  wrappers.py    Reward, ParallelReward, DiscreteActions
  cli.py         the navsim command
  _sim/          the simulator: world.py (motion, collisions, RobotState), geometry, planning, ...
docs/            documentation (see docs/internals/code-layout.md for the full tree)
tests/           pytest suite
scripts/         smoke_wheel.py (build and test the wheel), make_drive_folder.py
```

## Development

```bash
pip install -e ".[test,all]"
pytest                                   # tests (the regression test runs if hm3denv is installed)
python scripts/smoke_wheel.py            # build dist/navsim-<version>-py3-none-any.whl, test it in a clean venv
python scripts/make_drive_folder.py      # GGDrive/navsim/ = wheel + README.md + README.vi.md, for Google Drive
```

After every change run `scripts/make_drive_folder.py`, so `GGDrive/navsim/` matches the project. The
datasets are built with [hm3denv](https://github.com/TrKimHieu/2D_Navigation_Sim) (same format).
More: [For maintainers](docs/internals/maintainers.md).

## License

[MIT](LICENSE). Datasets built from HM3D are subject to Matterport's HM3D terms of use.
