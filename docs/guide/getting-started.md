# Bắt đầu

## Cài đặt

NavSim là một file wheel `navsim-<phiên bản>-py3-none-any.whl` (Python ≥ 3.10).

| Ở đâu | Cách làm |
|---|---|
| **Máy cá nhân** | `pip install "navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Colab** | đặt thư mục `navsim` (wheel + README) ở gốc Google Drive, rồi trong một ô: `from google.colab import drive; drive.mount("/content/drive")` và `!pip install -q "/content/drive/MyDrive/navsim/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |
| **Kaggle** | *Add Input → Upload* wheel thành một Dataset riêng tư, rồi `!pip install -q "/kaggle/input/<tên>/navsim-0.1.0-py3-none-any.whl[hub,marl,fast]"` |

| Extra | Cài thêm | Để làm gì |
|---|---|---|
| `hub` | `huggingface_hub` | tải dataset đầy đủ ([`navsim.setup`](../api/datasets.md)) |
| `marl` | `pettingzoo` | môi trường nhiều robot |
| `fast` | `numba` | LiDAR và dò collision nhanh hơn ~3 lần, **kết quả y hệt** (tắt bằng `NAVSIM_ACCEL=0`) |
| `plot` | `matplotlib` | ảnh/GIF kiểu `"plot"` |
| `all` | tất cả trên | |
| `test` | `pytest`, `pettingzoo` | chạy test |

Thư viện bắt buộc (tự cài): `gymnasium>=1.0`, `numpy>=1.25`, `scipy>=1.10`,
`opencv-python-headless>=4.5`, `pillow>=9`.

- Nếu notebook đang có numpy 1.x, thêm `"numpy==<bản đang có>"` vào cùng lệnh `pip install`, nếu
  không pip sẽ nâng numpy (opencv mới cần numpy 2) và phải khởi động lại kernel.
- Chỉ cài từ file wheel: `pip install navsim` sẽ cài một gói *khác* trên PyPI trùng tên.

## Bắt đầu nhanh

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

`import navsim` là bắt buộc: nó đăng ký `NavSim/Nav-v0` và `NavSim/GridNav-v0` với Gymnasium.

**Danh sách mọi thứ công khai ở mức `navsim.*`:**

| Tên | Loại | Tác dụng | Mục |
|---|---|---|---|
| `navsim.setup(...)` | hàm | làm dataset sẵn sàng (tải nếu cần) | [14](../api/datasets.md) |
| `navsim.load_dataset(name_or_path)` | hàm | nạp dataset thành `Dataset` | [14](../api/datasets.md) |
| `navsim.list_datasets()` | hàm | các dataset có trên máy | [14](../api/datasets.md) |
| `navsim.Dataset` | class | một dataset | [14](../api/datasets.md) |
| `navsim.list_robots()` | hàm | id các robot | [15](../api/robots.md) |
| `navsim.load_robot(id_or_path)` | hàm | nạp `Robot` | [15](../api/robots.md) |
| `navsim.evaluate(...)` | hàm | đo policy trên cả split | [11](../api/evaluation.md) |
| `navsim.plot_episode(...)`, `render_frame(...)`, `save_gif(...)`, `record_video(...)` | hàm | ảnh, GIF | [12](../api/visualization.md) |
| `navsim.env_fn(...)` | hàm | tạo env trong tiến trình con | [13](../api/parallel.md) |
| `navsim.wrappers` | module | `Reward`, `DiscreteActions`, `ParallelReward` | [10](../api/wrappers.md) |
| `navsim.__version__` | str | phiên bản (`"0.1.0"`) | |
