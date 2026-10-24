# NavSim

Robot navigation environments for reinforcement learning: a robot drives from a start to a goal
in real building floor plans, with real robots (size, speed limits, LiDAR). One robot uses the
[Gymnasium](https://gymnasium.farama.org/) API, several robots use [PettingZoo](https://pettingzoo.farama.org/).
CPU only. [Tiếng Việt](README.vi.md)

NavSim is **only the environment**: it has no reward and no training code. `step()` returns a
reward of 0.0; you write your own reward from the robot's state (pose, goal, trajectory,
shortest path, distances, ...), then train with any library.

**Contents:** [Install](#install) · [Quick start](#quick-start) · [Environments](#environments) ·
[Choosing episodes](#choosing-episodes-and-curriculum) · [Robot state](#robot-state-to-write-a-reward) ·
[Writing a reward](#writing-a-reward) · [Wrappers](#wrappers) · [Evaluation](#evaluation) ·
[Pictures and GIFs](#pictures-and-gifs) · [Parallel environments](#parallel-environments) ·
[Datasets](#datasets) · [Robots](#robots) · [Command line](#command-line) ·
[Environment variables](#environment-variables) · [Code layout](#code-layout)

## Install

NavSim is one wheel file, `navsim-<version>-py3-none-any.whl` (Python ≥ 3.10).

| Where | How |
|---|---|
| **Colab** | put the `navsim` folder (wheel + README) at the root of your Google Drive, then in a cell: `from google.colab import drive; drive.mount("/content/drive")` and `!pip install -q "/content/drive/MyDrive/navsim/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Kaggle** | *Add Input → Upload* the wheel as a private Dataset, then `!pip install -q "/kaggle/input/<name>/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Computer** | `pip install "navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |

Extras: `hub` downloads datasets, `marl` adds PettingZoo (several robots), `fast` adds numba
(about 3× faster, same results), `plot` adds matplotlib (pictures, GIFs in "plot" style).

- If the notebook has numpy 1.x, add `"numpy==<its version>"` to the same `pip install`,
  otherwise pip upgrades numpy (recent opencv needs numpy 2) and the kernel must restart.
- Install only from the wheel: `pip install navsim` installs an unrelated PyPI package with
  the same name.

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

## Environments

| | `NavSim/Nav-v0` | `NavSim/GridNav-v0` | `navsim.multiagent.nav_v0` |
|---|---|---|---|
| World | floor plan, metres | grid, one cell = one robot body | floor plan, several robots |
| API | Gymnasium | Gymnasium | PettingZoo `ParallelEnv` (and AEC) |
| Make | `gym.make("NavSim/Nav-v0", ...)` | `gym.make("NavSim/GridNav-v0", ...)` | `nav_v0.parallel_env(...)` |
| Reward | 0.0 | 0.0 | 0.0 for every robot |

`step()` returns `(obs, reward, terminated, truncated, info)` (PettingZoo: one dict per item,
keyed by agent). `reset(seed=None, options=None)` returns `(obs, info)`.

### `NavSim/Nav-v0`

| Argument | Default | Meaning |
|---|---|---|
| `dataset` | `"demo-svg"` | dataset name or folder (an svg dataset) |
| `robot` | the only robot, else `"turtlebot4"` | robot id (`navsim robots`) |
| `split` | `None` (all maps) | `"train"`, `"val"` or `"test"` (split by building) |
| `map_id`, `task_idx` | `None` | always use this map / task |
| `normalize_obs` | `True` | observation scaled to [0, 1] / [-1, 1]; `False`: metres, m/s, rad/s |
| `dt` | `0.1` | seconds per step |
| `n_beams` | `72` | LiDAR beams in the observation |
| `lidar_noise` | `0.0` | standard deviation of Gaussian LiDAR noise (m) |
| `on_collision` | `"stop"` | `"stop"`: the robot stops, the episode goes on; `"terminate"`: the episode ends |
| `time_limit` | `None` | episode length in seconds; `None`: `time_factor` × the least time the trip needs |
| `time_factor` | `3.0` | see `time_limit` |
| `success_radius` | from the task file (0.2) | the goal is reached within this distance (m) |
| `task_filter` | `None` | curriculum filter, see [Choosing episodes](#choosing-episodes-and-curriculum) |
| `nav_info` | `True` | add `goal_geodesic_m`, `progress_m`, `goal_distance_m`, `wall_distance_m` to the step info |
| `render_mode` | `None` | `"rgb_array"`: `env.render()` returns an image |
| `map_repeat` | `1` | keep a random map for this many episodes in a row (fewer map loads) |
| `map_cache`, `field_cache`, `disk_cache`, `clearance` | `8`, `4`, `True`, `None` | memory / disk caches, planner clearance (m) |

| Space | Value |
|---|---|
| action | `Box(-1, 1, (2,))`: (forward speed, turning speed) as a share of the robot's limits; `(3,)` = (forward, sideways, turning) for an omnidirectional robot (`jetauto_pro`). A negative forward speed uses the reverse limit. |
| observation | `Dict("lidar": (n_beams,), "goal": (3,), "velocity": (2,) or (3,))`. lidar: distances / range_max; goal: (distance / range_max capped at 1, sin, cos of the goal direction seen from the robot); velocity: current velocity / limits |

| Episode end | When |
|---|---|
| `terminated` | the robot centre is within `success_radius` of the goal (checked during the motion), or a collision with `on_collision="terminate"` |
| `truncated` | the time limit is reached |

| `info` | When | Meaning |
|---|---|---|
| `map_id`, `task_id`, `split`, `geodesic_m`, `time_limit` | reset | episode; `geodesic_m` = shortest path start → goal (m) |
| `collided`, `n_collisions` | every step | collision in this step; steps with a collision so far |
| `time`, `pose`, `path_length` | every step | seconds; (x, y, theta); metres driven |
| `goal_geodesic_m`, `progress_m` | every step (`nav_info`) | shortest path to the goal now; how much it shrank in this step |
| `goal_distance_m`, `wall_distance_m` | every step (`nav_info`) | straight-line distance to the goal; distance to the nearest wall |
| `success`, `spl`, `termination`, `geodesic_m`, `map_id`, `task_id`, `split`, `path_length`, `time`, `n_collisions` | last step | `termination`: `"success"`, `"collision"` or `"time_limit"`; `spl` = success × shortest / max(driven, shortest) |

### `NavSim/GridNav-v0`

| Argument | Default | Meaning |
|---|---|---|
| `dataset` | `"demo-grid"` | a grid dataset |
| `robot`, `split`, `map_id`, `task_idx`, `task_filter`, `render_mode`, `map_repeat` | | as in `Nav-v0` |
| `goal_obs` | `False` | add the goal to the observation (needed to drive *to* the goal) |
| `normalize_obs` | `False` | divide LiDAR and goal by `lidar_range` |
| `budget` | `200` | steps before `truncated` |
| `lidar_rays`, `lidar_range` | `22`, `10.0` | LiDAR rays, range in cells |
| `difficulty` | `None` | `"easy"`, `"medium"`, `"hard"` or `"starved"`: only tasks of this level |

| Space / end | Value |
|---|---|
| action | `Discrete(4)`: 0 up, 1 down, 2 left, 3 right; a blocked move stays in place (a collision) |
| observation | `Box(0, lidar_range, (lidar_rays,))`: LiDAR distances in cells; with `goal_obs=True`: `Dict("lidar", "goal": (2,))`, goal = (rows, columns) to the goal |
| `terminated` / `truncated` | on the goal cell / after `budget` steps |

| `info` | When | Meaning |
|---|---|---|
| `start`, `goal`, `d_bfs`, `level`, `p0`, `sa_uniform`, `sa_persist`, `starved`, `map_id`, `task_id`, `split` | reset | `d_bfs` = fewest moves start → goal; the others are the task's labels |
| `steps`, `collided`, `goal_geodesic`, `progress`, `d_bfs` | every step | moves to the goal now; how much it shrank in this step |
| `success`, `spl`, `termination`, `path_length` (m), `time` (steps), `n_collisions`, `geodesic_m`, ... | last step | as in `Nav-v0` |

### `navsim.multiagent.nav_v0`

```python
from navsim.multiagent import nav_v0
env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=4, split="train")
obs, infos = env.reset(seed=0)
while env.agents:
    obs, rewards, terminations, truncations, infos = env.step({a: policy(obs[a]) for a in env.agents})
```

Arguments: those of `Nav-v0` (except `task_idx`), plus:

| Argument | Default | Meaning |
|---|---|---|
| `n_agents` | `2` | robots, all of model `robot` |
| `robots` | `None` | a mixed team instead, e.g. `["turtlebot4", "clearpath_jackal"]` (same height class) |
| `flatten_obs` | `False` | one flat `Box` per robot instead of the `Dict` (SuperSuit, BenchMARL) |
| `on_done` | `"park"` | a finished robot stays on the map as an obstacle; `"remove"`: it disappears |
| `separation` | `0.1` | extra distance (m) between the starts and goals of different robots |
| `max_tries` | `50` | random draws per robot before trying another map |

- Agents are `"robot_0" ... "robot_{n-1}"`. A robot that is done leaves `env.agents`; the
  episode is over when `env.agents` is empty. Per-robot spaces, `info` and episode end: as `Nav-v0`.
- Each robot sees the others in its LiDAR and collides with them.
- `env.state()`: global state (10 values per robot: x, y, cos, sin, v, vy, w, goal x, goal y,
  active), for centralised critics; `env.state_space`.
- `nav_v0.env(...)`: the same environment with the AEC API; `nav_v0.raw_env`: the class.

## Choosing episodes and curriculum

| Call | Effect |
|---|---|
| `env.reset(seed=s)` | a random map of the split, then a random task of it (seeded) |
| `env.reset(options={"map_id": m, "task_idx": i})` | task `i` of map `m` |
| `env.reset(options={"map_id": m, "start": (x, y[, theta]), "goal": (x, y)})` | your own start and goal (grid: `(row, col)` cells); checked, `task_id` = -1 |
| nav_v0: `options={"task_idx": [i0, None, ...]}` / `{"start": [...], "goal": [...]}` | one entry per robot |
| `env.unwrapped.map_ids` | the maps of the split |
| `env.unwrapped.tasks_of(map_id)` | the tasks of a map (list of dicts, see [Datasets](#datasets)) |
| `env.unwrapped.set_task_filter(**criteria)` | only tasks matching every criterion; returns `{"maps": n, "tasks": n}` |
| `env.unwrapped.set_task_filter()` | no filter |

Criteria are looked up in the task (`geodesic_m`, `euclidean_m`), then in its `labels`
(svg: `gdr`; grid: `d_bfs`, `level`, ...): `geodesic_m=(1.0, 5.0)` (range, `None` = no bound),
`geodesic_m={"min": 1.0}`, `level=["easy", "medium"]` (allowed values), `level="easy"`.
In a vector environment: `venv.call("set_task_filter", ...)` (Gymnasium) or
`venv.env_method("set_task_filter", ...)` (Stable-Baselines3).

## Robot state (to write a reward)

`state = env.unwrapped.robot_state()` (`Nav-v0`, `GridNav-v0`), `state = env.robot_state(agent)`
(`nav_v0`). The object always shows the latest state, so you can keep it. `point` / `cell`
arguments default to the robot's position.

| Member | `Nav-v0`, `nav_v0` (metres, radians, seconds) | `GridNav-v0` (cells) |
|---|---|---|
| `position` | (x, y) | (row, col) |
| `pose`, `heading` | (x, y, theta); theta | — |
| `velocity` | (v, w) or (v, vy, w) after the last step | — |
| `start`, `goal` | (x, y, theta); (x, y) | (row, col); (row, col) |
| `start_geodesic` | shortest path start → goal (m) | fewest moves start → goal |
| `trajectory` | `(N, 4)` array: time, x, y, theta per step since reset | `(N, 2)` array: row, col per step |
| `path_length` | metres driven | moves made |
| `steps`; `time`, `time_limit` | steps; seconds | steps; `budget` |
| `collided`, `n_collisions` | collision in the last step; steps with a collision | blocked last move; blocked moves |
| `reached_goal`, `done` | on the goal; episode over for this robot | same |
| `goal_distance(point)` | straight line to the goal | straight line to the goal (cells) |
| `geodesic_distance(point)` | shortest drivable path to the goal (inf if unreachable) | fewest moves to the goal |
| `shortest_path(point)` | `(M, 2)` points (x, y) from `point` to the goal | `(M, 2)` cells from `cell` to the goal |
| `wall_distance(point)` | distance to the nearest wall | — |
| `lidar()` | LiDAR distances (m), noise-free; directions: `beam_angles` | LiDAR distances (cells) |
| `fits(x, y, theta)` / `is_free(cell)` | the robot's body fits there | the cell is free |
| `robot` / `success_radius` / `cell_m` | robot model; goal radius | cell size (m) |

Other attributes of `env.unwrapped`: `map` (`SvgMap`: the free area, `map.raster(res)`),
`robot` (model), `tasks` (current map's tasks), `world` (the simulator); grid: `grid` (0 = free).

## Writing a reward

`navsim.wrappers.Reward(env, fn)` calls `fn(state, info)` after every step and returns its value
as the reward (`navsim.wrappers.ParallelReward(env, fn)` for `nav_v0`, per robot). Examples:

```python
# progress along the shortest path, +10 at the goal, -0.1 per collision, -0.01 per step
def dense(state, info):
    return info["progress_m"] + 10 * state.reached_goal - 0.1 * state.collided - 0.01

# sparse: 1 at the goal
sparse = lambda state, info: float(state.reached_goal)

# keep away from walls
def careful(state, info):
    return info["progress_m"] - 0.05 * (state.wall_distance() < 0.3) + 10 * state.reached_goal

# grid: +0.1 per move closer to the goal, +1 at the goal
grid_reward = lambda state, info: 0.1 * info["progress"] + state.reached_goal

env = navsim.wrappers.Reward(gym.make("NavSim/Nav-v0"), dense)
team = navsim.wrappers.ParallelReward(nav_v0.parallel_env(n_agents=4), dense)
```

Your own variables between steps: use a class with `__call__` (reset them when `state.steps == 1`).

## Wrappers

| Wrapper | What |
|---|---|
| `navsim.wrappers.Reward(env, fn)` | reward = `fn(state, info)` (above) |
| `navsim.wrappers.ParallelReward(env, fn)` | the same for `nav_v0`, per robot (PettingZoo wrapper) |
| `navsim.wrappers.DiscreteActions(env, step_m=0.25, turn_deg=15.0)` | `Nav-v0` with `Discrete(5)`: 0 wait, 1 forward, 2 backward, 3 turn left, 4 turn right (`Discrete(7)`: + 5 / 6 sideways for omnidirectional robots); each action runs at top speed over several steps, `info["macro_steps"]` |
| `gymnasium.wrappers.FlattenObservation(env)` | one flat vector instead of the `Dict` |

## Evaluation

```text
navsim.evaluate(policy, env="NavSim/Nav-v0", *, split="test", episodes=None, per_map=None, seed=0,
                deterministic=None, progress=False, return_episodes=False, **env_kwargs) -> dict
```

| Argument | Meaning |
|---|---|
| `policy` | a model with `predict` (Stable-Baselines3), a function `obs -> action`, `"random"`, or for `nav_v0` a dict `{agent: policy}` |
| `env` | an env id (made with `split` and `env_kwargs`), a Gymnasium env (wrappers kept) or a `nav_v0` env |
| `episodes` | `None`: every task of every map once, in order (`per_map` caps the tasks per map); a number: that many random episodes. `nav_v0`: default 20 |
| `deterministic` | models: `None` = most likely action, except on `GridNav-v0` where actions are sampled (a greedy grid policy gets stuck between two cells) |
| `return_episodes` | also return `"episode_infos"`: the last `info` of every episode |

Returns `{"split", "episodes", "success", "spl", "collision_episodes", "mean_progress", "mean_time",
"mean_path_m"}` (+ `"deterministic"` for a model). `mean_progress`: share of the shortest path
covered at the end. `nav_v0` adds `"robots"`, `"robot_episodes"`, `"team_success"`.

## Pictures and GIFs

| Function | Returns | What |
|---|---|---|
| `navsim.plot_episode(env, ax=None, lidar=True, path=True, shortest_path=False, title=True, legend=True, figsize=(8, 8), save=None, dpi=150)` | matplotlib `Axes` | map, path since reset, start, goal, robot, LiDAR; `shortest_path=True` adds the shortest path to the goal; `save="x.png"` (or .pdf, .svg) writes the figure |
| `navsim.render_frame(env, style="plot", size=480, lidar=True, shortest_path=False)` | `(H, W, 3)` uint8 | image of the current step; `"plot"`: like `plot_episode`; `"fast"`: the environment's own image (no matplotlib, much faster) |
| `navsim.save_gif(frames, path, fps=10)` | `Path` | a list of images → animated GIF |
| `navsim.record_video(env, policy=None, path="videos/episode.gif", episodes=1, seed=0, options=None, fps=None, max_steps=1000, every=1, size=480, style="plot", deterministic=None, show=True)` | `Path` | runs `policy` (None = random) and saves a GIF (every `every`-th step); shown in a notebook |
| `env.render()` | `(H, W, 3)` uint8 | with `render_mode="rgb_array"` |

All of them work with the three environments, after `reset` and at any step; drawing does not
change the episode. A "plot" frame takes about 0.1 s: for long episodes record every few steps
(`every=5`) or use `style="fast"`. GIF of your own loop:

```python
frames = [navsim.render_frame(env)]
while not done:
    obs, reward, terminated, truncated, info = env.step(policy(obs))
    done = terminated or truncated
    frames.append(navsim.render_frame(env))
navsim.save_gif(frames, "run.gif", fps=10)
```

## Parallel environments

| Call | What |
|---|---|
| `gym.make_vec("NavSim/Nav-v0", num_envs=8, vectorization_mode="async", wrappers=[...], **kwargs)` | Gymnasium vector environment, one process per copy |
| `navsim.env_fn(env_id, wrappers=(), **kwargs)` | a picklable `() -> env` for libraries that make the environments in worker processes |

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

venv = make_vec_env(navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg",
                                  wrappers=[lambda e: navsim.wrappers.Reward(e, dense)]),
                    n_envs=8, vec_env_cls=SubprocVecEnv)
```

`navsim.env_fn` works everywhere. A plain id (`make_vec_env("NavSim/Nav-v0", ...)`) also works
on Linux (Colab, Kaggle) but fails on Windows and macOS with `Namespace NavSim not found`.
In a `.py` script, put this code under `if __name__ == "__main__":`.

## Datasets

| Call | Returns |
|---|---|
| `navsim.setup(datasets="demo-svg", token=None, data_dir=None, force=False, quiet=False)` | a `Dataset` (a list for a list); downloads once from Hugging Face if it is not on the machine |
| `navsim.load_dataset(name_or_path)` | a `Dataset` |
| `navsim.list_datasets()` | the datasets on this machine |

| `Dataset` member | Value |
|---|---|
| `name`, `env_type`, `root`, `robots`, `splits` | name; `"svg"` or `"grid"`; folder; robot ids; split → buildings |
| `maps(robot, split=None)` | map ids |
| `tasks(robot, map_id)` | list of tasks: `{"id", "start": {"x", "y", "theta"}, "goal": {"x", "y"}, "geodesic_m", "euclidean_m", "labels"}` (grid: `start` / `goal` also have `"cell": [row, col]`) |
| `robot(robot_id)`, `height_class(robot_id)`, `default_robot()` | robot model; map set; the default robot |
| `svg_path(robot, map_id)`, `grid_arrays(map_id)`, `grid_meta(map_id)` | map file; grid arrays; grid metadata (`cell_m`, ...) |
| `summary()`, `validate()` | maps and tasks per robot and split; list of problems (empty = valid) |

Two demos are in the package: `demo-svg` (floor plans: turtlebot4, clearpath_jackal) and
`demo-grid` (grids: jetauto_pro), each 3 train / 2 val / 1 test buildings. Full datasets are
downloaded from Hugging Face: request access on
[the dataset page](https://huggingface.co/datasets/PhysicalSI/2D_Navigation_Sim), save a *Read*
token as a secret named `HF_TOKEN` (Colab: key icon, *Notebook access* on; Kaggle: *Add-ons →
Secrets*, *Internet* on; computer: `export HF_TOKEN=...`), then `navsim.setup("isb-svg-v1")`.

A dataset name is looked for in: `$NAVSIM_DATASETS`, `~/.navsim/datasets`, Kaggle inputs
(`/kaggle/input/*/<name>`), the demos.

## Robots

`navsim.list_robots()` → ids; `navsim.load_robot(id_or_json_path)` → `Robot`:

| `Robot` member | Value |
|---|---|
| `id`, `name`, `drive` | id; full name; `"differential"` or `"omnidirectional"` (`omni`) |
| `footprint` | `(M, 2)` polygon in the robot frame (m), x forward, y left |
| `r_circ` | radius of the smallest circle around the centre that holds the robot (m) |
| `v_max`, `v_min`, `vy_max`, `w_max` | speed limits: forward, reverse (≤ 0), sideways (m/s); turning (rad/s) |
| `lidar` | `{"range_min", "range_max", "fov", "beams", ...}` (m, rad) |
| `height`, `height_class`, `action_dim`, `raw` | m; map set; 2 or 3; the JSON file |

Robots: agilex_limo, clearpath_jackal, jetauto_pro, kobuki, pal_tiago, turtlebot3_burger,
turtlebot3_waffle_pi, turtlebot4, turtlebot4_lite (`navsim robots` prints their numbers).

## Command line

```bash
navsim setup isb-svg-v1        # download a dataset
navsim datasets                # datasets on this machine
navsim robots                  # robot models
navsim info demo-svg           # maps and tasks per robot and split
navsim check                   # Gymnasium / PettingZoo API checks and speed
navsim cache --clear           # remove the map cache
```

## Environment variables

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Hugging Face token (also read from the Colab / Kaggle secret `HF_TOKEN`) |
| `NAVSIM_HOME` | data folder (default `~/.navsim`; datasets in `<it>/datasets`) |
| `NAVSIM_DATASETS` | extra folders searched for datasets (`:` or `;` separated) |
| `NAVSIM_ROBOTS` | extra folders searched for robot JSON files |
| `NAVSIM_CACHE` | map cache folder, or `off` |
| `NAVSIM_ACCEL` | `0` disables the numba kernels |

## Code layout

```
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
```

## For maintainers

```bash
pip install -e ".[test,all]"
pytest                                   # tests (the regression test runs if hm3denv is installed)
python scripts/smoke_wheel.py            # build dist/navsim-<version>-py3-none-any.whl, test it in a clean venv
python scripts/make_drive_folder.py      # GGDrive/navsim/ = wheel + README.md + README.vi.md, for Google Drive
```

After every change run `scripts/make_drive_folder.py`, so `GGDrive/navsim/` matches the
project. The datasets are built with [hm3denv](https://github.com/TrKimHieu/2D_Navigation_Sim)
(same format). Design notes: [docs/specs](docs/specs/2026-10-12-navsim-design.md).

MIT License. Datasets built from HM3D are subject to Matterport's HM3D terms of use.
#   n a v s i m  
 