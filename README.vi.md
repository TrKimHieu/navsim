<div align="center">

# NavSim

**Môi trường điều hướng robot cho học tăng cường, trên mặt bằng tòa nhà thật.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Gymnasium](https://img.shields.io/badge/API-Gymnasium-0081A5)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/multi--agent-PettingZoo-orange)](https://pettingzoo.farama.org/)
[![CPU only](https://img.shields.io/badge/runs%20on-CPU-lightgrey)](#cài-đặt)

[English](README.md) · Tiếng Việt

</div>

Robot đi từ điểm xuất phát (start) tới điểm đích (goal) trên mặt bằng tầng nhà thật, với kích thước,
giới hạn tốc độ và LiDAR đúng theo thông số của robot thật. Một robot dùng API
[Gymnasium](https://gymnasium.farama.org/); nhiều robot trên cùng một bản đồ dùng
[PettingZoo](https://pettingzoo.farama.org/).

NavSim **chỉ là môi trường**: không có reward và không có code huấn luyện. `step()` luôn trả reward
`0.0`; bạn tự viết reward từ trạng thái robot (tư thế, đích, quỹ đạo, đường ngắn nhất, khoảng cách, ...)
rồi huấn luyện bằng thư viện nào cũng được.

## Điểm nổi bật

- **Ba môi trường**: mặt bằng liên tục (`Nav-v0`), lưới ô (`GridNav-v0`), nhiều robot trên một bản đồ (`nav_v0`).
- **Robot thật**: chín mẫu (TurtleBot 3/4, Jackal, TIAGo, Kobuki, ...) với footprint, giới hạn tốc độ và LiDAR; lái vi sai và đa hướng.
- **Bản đồ thật**: các tòa nhà từ HM3D, chia train / val / test theo tòa nhà. Hai dataset demo nhỏ nằm sẵn trong package.
- **Tự viết reward**: wrapper và `RobotState` đầy đủ (khoảng cách geodesic, đường ngắn nhất, khoảng cách tới tường, quỹ đạo, ...).
- **Curriculum**: lọc task theo độ dài đường đi hoặc nhãn độ khó bất cứ lúc nào.
- **Đủ đồ nghề**: `evaluate()` (success, SPL, collision), ảnh và GIF, môi trường song song, lệnh `navsim`.
- **Nhanh và tất định**: chỉ cần CPU, kernel numba tùy chọn (nhanh khoảng 3 lần, kết quả giống hệt), cache bản đồ trên đĩa.

## Mục lục

[Cài đặt](#cài-đặt) · [Chạy thử](#chạy-thử) · [Môi trường](#môi-trường) ·
[Chọn episode](#chọn-episode-và-curriculum) · [Reward](#trạng-thái-robot-và-reward) ·
[Đánh giá](#đánh-giá-policy) · [Ảnh và GIF](#ảnh-và-gif) · [Môi trường song song](#môi-trường-song-song) ·
[Dataset](#dataset) · [Robot](#robot) · [Dòng lệnh](#dòng-lệnh) · [Tài liệu](#tài-liệu) ·
[Cấu trúc code](#cấu-trúc-code) · [Phát triển](#phát-triển) · [Giấy phép](#giấy-phép)

## Cài đặt

NavSim là một file wheel, `navsim-<phiên bản>-py3-none-any.whl` (Python ≥ 3.10).

| Ở đâu | Cách làm |
|---|---|
| **Máy cá nhân** | `pip install "navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Colab** | đặt thư mục `navsim` (wheel + README) ở gốc Google Drive, rồi trong một ô: `from google.colab import drive; drive.mount("/content/drive")` và `!pip install -q "/content/drive/MyDrive/navsim/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Kaggle** | *Add Input → Upload* wheel thành một Dataset riêng tư, rồi `!pip install -q "/kaggle/input/<tên>/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |

| Extra | Cài thêm | Để làm gì |
|---|---|---|
| `hub` | `huggingface_hub` | tải dataset |
| `marl` | `pettingzoo` | nhiều robot |
| `fast` | `numba` | nhanh khoảng 3 lần, kết quả giống hệt |
| `plot` | `matplotlib` | ảnh và GIF kiểu `"plot"` |
| `all` | tất cả trên | |

> **Lưu ý**
> - Nếu notebook đang có numpy 1.x, thêm `"numpy==<bản đang có>"` vào cùng lệnh `pip install`, nếu không pip sẽ nâng numpy (OpenCV mới cần numpy 2) và phải khởi động lại kernel.
> - Chỉ cài từ file wheel: `pip install navsim` cài một package *khác* trên PyPI trùng tên.

Thêm: [Bắt đầu](docs/guide/getting-started.md).

## Chạy thử

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

`import navsim` là bắt buộc: nó đăng ký các id `NavSim/...` với Gymnasium. Hai dataset demo (`demo-svg`,
`demo-grid`) nằm sẵn trong package, không cần tải.

## Môi trường

| | [`NavSim/Nav-v0`](docs/environments/nav_v0.md) | [`NavSim/GridNav-v0`](docs/environments/grid_nav_v0.md) | [`navsim.multiagent.nav_v0`](docs/environments/multiagent_nav_v0.md) |
|---|---|---|---|
| Không gian | mặt bằng, đơn vị mét | lưới ô, mỗi ô bằng một thân robot | mặt bằng, nhiều robot |
| API | Gymnasium | Gymnasium | PettingZoo `ParallelEnv` (và AEC) |
| Tạo | `gym.make("NavSim/Nav-v0", ...)` | `gym.make("NavSim/GridNav-v0", ...)` | `nav_v0.parallel_env(...)` |
| Action | `Box(-1, 1, (2,))`, `(3,)` nếu đa hướng | `Discrete(4)` | mỗi robot như `Nav-v0` |
| Observation | `Dict(lidar, goal, velocity)` | LiDAR (+ goal với `goal_obs=True`) | mỗi robot như `Nav-v0` |
| Reward | 0.0 | 0.0 | 0.0 cho mỗi robot |
| Khi nào dùng | cần mô phỏng sát robot thật (kích thước, tốc độ) | chạy nhanh, thuật toán rời rạc, curriculum theo độ khó | nghiên cứu multi-agent |

`step()` trả `(obs, reward, terminated, truncated, info)` (PettingZoo: mỗi thành phần là một dict theo
agent). `reset(seed=None, options=None)` trả `(obs, info)`. Mỗi môi trường có một trang riêng trong
[tài liệu](docs/environments/README.md): không gian action và observation, điều kiện kết thúc episode, các
khoá `info` và mọi tham số.

```python
# nhiều robot trên một bản đồ (cần extra `marl`)
from navsim.multiagent import nav_v0

env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=4, split="train")
obs, infos = env.reset(seed=0)
while env.agents:
    obs, rewards, terminations, truncations, infos = env.step(
        {a: env.action_space(a).sample() for a in env.agents})
```

## Chọn episode và curriculum

| Lệnh | Tác dụng |
|---|---|
| `env.reset(seed=s)` | một map ngẫu nhiên của split, rồi một task ngẫu nhiên của map đó (có seed) |
| `env.reset(options={"map_id": m, "task_idx": i})` | task `i` của map `m` |
| `env.reset(options={"map_id": m, "start": (x, y[, theta]), "goal": (x, y)})` | start và goal tự chọn (có kiểm tra; grid: ô `(hàng, cột)`) |
| `env.unwrapped.set_task_filter(geodesic_m=(1.0, 5.0))` | chỉ các task khớp mọi tiêu chí; không đối số = bỏ lọc |

Tiêu chí được tìm trong task (`geodesic_m`, `euclidean_m`), rồi trong `labels` của nó (svg: `gdr`; grid:
`d_bfs`, `level`, ...). Với vector env dùng `venv.call("set_task_filter", ...)`. Chi tiết:
[Chọn episode và curriculum](docs/concepts/episodes.md).

## Trạng thái robot và reward

`state = env.unwrapped.robot_state()` (`Nav-v0`, `GridNav-v0`) hoặc `state = env.robot_state(agent)`
(`nav_v0`) luôn phản ánh trạng thái mới nhất, nên có thể giữ lại dùng tiếp.

| Thành phần | Ý nghĩa |
|---|---|
| `pose`, `position`, `heading`, `velocity` | robot đang ở đâu và di chuyển thế nào |
| `start`, `goal`, `trajectory`, `path_length` | episode cho tới lúc này |
| `collided`, `n_collisions`, `reached_goal`, `done` | điều đã xảy ra |
| `goal_distance()`, `geodesic_distance()`, `shortest_path()` | khoảng cách đường thẳng, đường đi được ngắn nhất và đường đó tới goal (từ điểm bất kỳ) |
| `wall_distance()`, `lidar()`, `fits(x, y, theta)` | môi trường xung quanh |

Bảng đầy đủ (mét và radian so với ô grid): [Trạng thái robot](docs/concepts/robot_state.md).

```python
# tiến độ theo đường ngắn nhất, +10 khi tới goal, -0.1 mỗi collision, -0.01 mỗi step
def dense(state, info):
    return info["progress_m"] + 10 * state.reached_goal - 0.1 * state.collided - 0.01

env = navsim.wrappers.Reward(gym.make("NavSim/Nav-v0"), dense)
team = navsim.wrappers.ParallelReward(nav_v0.parallel_env(n_agents=4), dense)   # theo từng robot
```

| Wrapper | Tác dụng |
|---|---|
| `navsim.wrappers.Reward(env, fn)` | reward = `fn(state, info)` |
| `navsim.wrappers.ParallelReward(env, fn)` | như trên cho `nav_v0`, theo từng robot |
| `navsim.wrappers.DiscreteActions(env, step_m=0.25, turn_deg=15.0)` | `Nav-v0` với `Discrete(5)`: đứng yên, tiến, lùi, quay trái, quay phải |
| `gymnasium.wrappers.FlattenObservation(env)` | một vector phẳng thay cho `Dict` |

Thêm: [Wrappers](docs/api/wrappers.md).

## Đánh giá policy

```python
navsim.evaluate(model, "NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4")   # mọi task của split test
navsim.evaluate("random", nav_v0.parallel_env(n_agents=4, split="test"), episodes=50)
```

`policy` là model Stable-Baselines3, hàm `obs -> action`, `"random"`, hoặc (nhiều robot) dict
`{agent: policy}`. Kết quả gồm `success`, `spl`, `collision_episodes`, `mean_progress`, `mean_time`,
`mean_path_m` (nhiều robot: thêm `team_success`). Thêm: [Đánh giá](docs/api/evaluation.md).

## Ảnh và GIF

| Hàm | Tác dụng |
|---|---|
| `navsim.plot_episode(env, ..., save="x.png")` | vẽ map, lộ trình, start, goal, robot, LiDAR vào một `Axes` matplotlib |
| `navsim.render_frame(env, style="plot" \| "fast")` | ảnh `(H, W, 3)` của step hiện tại |
| `navsim.save_gif(frames, path, fps=10)` | list ảnh → GIF động |
| `navsim.record_video(env, policy, path="videos/episode.gif", ...)` | chạy policy và lưu GIF |

Dùng được với cả ba môi trường, ở mọi step, và không làm thay đổi episode. Một khung `"plot"` mất khoảng
0,1 giây: với episode dài hãy dùng `every=5` hoặc `style="fast"`. Thêm: [Ảnh và GIF](docs/api/visualization.md).

## Môi trường song song

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

venv = make_vec_env(navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg",
                                  wrappers=[lambda e: navsim.wrappers.Reward(e, dense)]),
                    n_envs=8, vec_env_cls=SubprocVecEnv)
```

`navsim.env_fn` chạy được ở mọi nơi. Dùng thẳng id (`make_vec_env("NavSim/Nav-v0", ...)`) cũng được trên
Linux nhưng báo `Namespace NavSim not found` trên Windows và macOS. Trong file `.py`, đặt đoạn này dưới
`if __name__ == "__main__":`. Thêm: [Môi trường song song](docs/api/parallel.md).

## Dataset

| Lệnh | Trả về |
|---|---|
| `navsim.setup(datasets="demo-svg", token=None, data_dir=None, force=False)` | một `Dataset`; tải một lần từ Hugging Face nếu chưa có trên máy |
| `navsim.load_dataset(name_or_path)` | một `Dataset` |
| `navsim.list_datasets()` | các dataset có trên máy |

Hai dataset demo nằm trong package: `demo-svg` (mặt bằng: turtlebot4, clearpath_jackal) và `demo-grid`
(lưới: jetauto_pro), mỗi dataset 3 tòa nhà train / 2 val / 1 test. Dataset đầy đủ tải từ Hugging Face:

1. Xin quyền ở [trang dataset](https://huggingface.co/datasets/PhysicalSI/2D_Navigation_Sim).
2. Lưu một token *Read* thành secret `HF_TOKEN` (Colab: biểu tượng chìa khóa, bật *Notebook access*; Kaggle: *Add-ons → Secrets*, bật *Internet*; máy cá nhân: `export HF_TOKEN=...`).
3. `navsim.setup("isb-svg-v1")`.

Tên dataset được tìm trong: `$NAVSIM_DATASETS`, `~/.navsim/datasets`, input của Kaggle
(`/kaggle/input/*/<tên>`), các demo. Thêm: [Dataset](docs/api/datasets.md); định dạng file `.svg`, `.npz`:
[Định dạng dataset](docs/internals/dataset-format.md).

## Robot

`navsim.list_robots()` → các id; `navsim.load_robot(id_hoặc_đường_dẫn_json)` → `Robot` (`footprint`,
`r_circ`, `v_max`, `v_min`, `vy_max`, `w_max`, `lidar`, `drive`, `height_class`, ...).

| Robot | Kiểu lái | Robot | Kiểu lái |
|---|---|---|---|
| `agilex_limo` | vi sai | `turtlebot3_burger` | vi sai |
| `clearpath_jackal` | vi sai | `turtlebot3_waffle_pi` | vi sai |
| `jetauto_pro` | đa hướng | `turtlebot4` | vi sai |
| `kobuki` | vi sai | `turtlebot4_lite` | vi sai |
| `pal_tiago` | vi sai | | |

`navsim robots` in kích thước và giới hạn của chúng. Thêm robot riêng bằng một file JSON:
[Robot](docs/api/robots.md); chi tiết thông số: [Robot và task](docs/internals/robots-and-tasks.md).

## Dòng lệnh

```bash
navsim setup isb-svg-v1        # tải một dataset
navsim datasets                # dataset có trên máy
navsim robots                  # các mẫu robot
navsim info demo-svg           # số map và task theo robot và split
navsim check                   # kiểm tra API Gymnasium / PettingZoo và đo tốc độ
navsim cache --clear           # xóa cache map
```

Cũng chạy được bằng `python -m navsim ...`. Biến môi trường (`HF_TOKEN`, `NAVSIM_HOME`,
`NAVSIM_DATASETS`, `NAVSIM_ROBOTS`, `NAVSIM_CACHE`, `NAVSIM_ACCEL`): [Dòng lệnh](docs/api/cli.md),
[Biến môi trường](docs/api/environment-variables.md).

## Tài liệu

Toàn bộ tài liệu nằm ở [`docs/`](docs/README.md). Bắt đầu từ trang hợp với việc bạn muốn làm:

| Tôi muốn... | Đọc |
|---|---|
| cài đặt và chạy ví dụ đầu tiên | [Bắt đầu](docs/guide/getting-started.md) |
| chọn và dùng một môi trường | [Môi trường](docs/environments/README.md): [Nav-v0](docs/environments/nav_v0.md), [GridNav-v0](docs/environments/grid_nav_v0.md), [nav_v0](docs/environments/multiagent_nav_v0.md) |
| chọn map và task, học theo curriculum | [Chọn episode](docs/concepts/episodes.md) |
| viết reward | [Trạng thái robot](docs/concepts/robot_state.md), [Wrappers](docs/api/wrappers.md) |
| đánh giá, vẽ, ghi hình | [Đánh giá](docs/api/evaluation.md), [Ảnh và GIF](docs/api/visualization.md) |
| huấn luyện song song | [Môi trường song song](docs/api/parallel.md), [Công thức hay dùng](docs/guide/recipes.md) |
| sửa lỗi | [Lỗi thường gặp](docs/guide/troubleshooting.md) |
| xem mọi lệnh dạng cây | [`docs/navsim-api-tree.html`](docs/navsim-api-tree.html) (mở bằng trình duyệt) |
| hiểu dữ liệu và cơ chế bên trong | [Tổng quan luồng hoạt động](docs/internals/overview.md), [Định dạng dataset](docs/internals/dataset-format.md), [LiDAR](docs/internals/lidar.md), [Một lần `step()` và shortest path](docs/internals/step-and-planning.md), [Kiến trúc](docs/internals/architecture.md), [Simulator](docs/internals/simulator.md) |

## Cấu trúc code

```text
src/navsim/
  envs/          nav.py (Nav-v0), grid.py (GridNav-v0, GridRobotState), episodes.py (chọn map và task)
  multiagent/    nav_v0.py (nhiều robot, ParallelReward)
  data/          dataset.py (tìm, đọc), download.py (setup), platform.py (token Colab / Kaggle)
  robots/        mô hình robot (presets/*.json)
  evaluation.py  evaluate()
  visualize.py   plot_episode(), render_frame(), save_gif(), record_video()
  wrappers.py    Reward, ParallelReward, DiscreteActions
  cli.py         lệnh navsim
  _sim/          simulator bên trong: world.py (chuyển động, collision, RobotState), geometry, planning, ...
docs/            tài liệu (cây đầy đủ: docs/internals/code-layout.md)
tests/           pytest
scripts/         smoke_wheel.py (build và thử wheel), make_drive_folder.py
```

## Phát triển

```bash
pip install -e ".[test,all]"
pytest                                   # chạy test (test hồi quy chạy nếu có cài hm3denv)
python scripts/smoke_wheel.py            # build dist/navsim-<phiên bản>-py3-none-any.whl, thử trong venv sạch
python scripts/make_drive_folder.py      # GGDrive/navsim/ = wheel + README.md + README.vi.md, để tải lên Drive
```

Sau mỗi thay đổi hãy chạy `scripts/make_drive_folder.py` để `GGDrive/navsim/` khớp với project. Dataset
được build bằng [hm3denv](https://github.com/TrKimHieu/2D_Navigation_Sim) (cùng định dạng). Thêm:
[Cho người bảo trì](docs/internals/maintainers.md).

## Giấy phép

[MIT](LICENSE). Các dataset build từ HM3D tuân theo điều khoản sử dụng HM3D của Matterport.
