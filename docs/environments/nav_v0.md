# Nav

Thuộc nhóm môi trường NavSim: robot điều hướng trên mặt bằng liên tục.

| | |
|---|---|
| Action Space | `Box(-1.0, 1.0, (2,), float32)`; robot đa hướng: `(3,)` |
| Observation Space | `Dict("lidar": Box(0.0, 1.0, (n_beams,)), "goal": Box([0, -1, -1], 1.0, (3,)), "velocity": Box(-1.0, 1.0, (2,)))` |
| import | `gymnasium.make("NavSim/Nav-v0")` |

## Description

Một robot thật (có kích thước thân, giới hạn tốc độ và LiDAR của mẫu robot đó) lái từ điểm xuất phát tới
điểm đích trên **mặt bằng tòa nhà thật** (dữ liệu loại `svg`, đơn vị mét). Mỗi bước, robot nhận tốc độ
tiến và tốc độ quay (đa hướng: thêm tốc độ ngang), mô phỏng chạy trong `dt` giây, và môi trường trả về
quan sát LiDAR, hướng tới đích và vận tốc hiện tại.

Môi trường **chỉ là môi trường**: không có reward và không có code huấn luyện. Bạn tự viết reward từ trạng
thái của robot (xem [Rewards](#rewards)).

## Action Space

Action là một vector `float32` trong `[-1, 1]`: **tỉ lệ so với giới hạn tốc độ của robot**.

| Chỉ số | Hành động | Nhân với | Ghi chú |
|---|---|---|---|
| 0 | tốc độ tiến | `v_max` nếu ≥ 0, `-v_min` nếu < 0 | giá trị âm là đi lùi |
| 1 | tốc độ quay (vi sai) / tốc độ ngang (đa hướng) | `w_max` / `vy_max` | quay trái là dương |
| 2 | tốc độ quay (chỉ robot đa hướng, ví dụ `jetauto_pro`) | `w_max` | |

**Ghi chú:** giá trị ngoài `[-1, 1]` được kẹp lại. Một `step` kéo dài `dt` giây và được chia thành các
bước nhỏ (tối đa 2 cm hoặc 2° mỗi bước nhỏ), nên robot nhanh không xuyên qua tường mỏng được.

## Observation Space

Quan sát là một `Dict` với ba khoá. Mặc định (`normalize_obs=True`) mọi giá trị được chuẩn hoá để học.

| Khoá | Hình dạng | `normalize_obs=True` | `normalize_obs=False` |
|---|---|---|---|
| `lidar` | `(n_beams,)` | khoảng cách / `range_max`, trong `[0, 1]` | mét, trong `[range_min, range_max]` |
| `goal` | `(3,)` | (`min(d / range_max, 1)`, `sin`, `cos`) | (`d` mét, `sin`, `cos`) |
| `velocity` | `(2,)` hoặc `(3,)` | vận tốc / giới hạn, trong `[-1, 1]` | m/s và rad/s |

`d` là khoảng cách đường thẳng tới đích; `sin`, `cos` là của góc từ mũi robot tới đích: `cos = 1` đích ngay
phía trước, `cos = -1` ngay sau lưng, `sin > 0` đích ở bên trái. `velocity` là vận tốc **sau step vừa rồi**
(về 0 nếu step đó va chạm).

**Ghi chú:** LiDAR quét `n_beams` tia đều nhau (360° nếu `fov ≥ 2π`, ngược lại trải đều trong `fov`). Tầm
và `fov` lấy từ mẫu robot; nhiễu Gauss (`lidar_noise`) được cộng sau khi quét rồi kẹp lại vào
`[range_min, range_max]`.

## Rewards

Môi trường **luôn trả `0.0`**. Dùng wrapper `navsim.wrappers.Reward(env, fn)` để tự viết, `fn(state, info)`
được gọi sau mỗi step với `state = env.unwrapped.robot_state()` (xem [RobotState](../concepts/robot_state.md)):

```python
def dense(state, info):
    # tiến độ theo đường ngắn nhất, +10 khi tới đích, -0.1 mỗi va chạm, -0.01 mỗi step
    return info["progress_m"] + 10 * state.reached_goal - 0.1 * state.collided - 0.01

env = navsim.wrappers.Reward(gym.make("NavSim/Nav-v0"), dense)
```

`info["progress_m"]` là mức giảm của đường ngắn nhất tới đích trong step này (âm nếu đi xa ra) và cần
`nav_info=True` (mặc định).

## Starting State

Mỗi episode lấy một **task** (điểm xuất phát `(x, y, θ)` và đích `(x, y)`) từ file dữ liệu: chọn một bản đồ
ngẫu nhiên của `split`, rồi một task ngẫu nhiên của bản đồ đó (có seed). Robot đứng yên (vận tốc 0) tại
điểm xuất phát. Bạn có thể chọn episode cụ thể bằng `reset(options=...)`:

| Lệnh | Tác dụng |
|---|---|
| `env.reset(seed=s)` | map và task ngẫu nhiên; cùng seed luôn ra cùng episode |
| `env.reset(options={"map_id": m, "task_idx": i})` | task `i` của map `m` |
| `env.reset(options={"map_id": m, "start": (x, y[, θ]), "goal": (x, y)})` | start/goal tự chọn (có kiểm tra, `task_id = -1`) |

## Episode End

1. **Terminated (thành công):** tâm robot cách đích ≤ `success_radius` (kiểm tra sau *mỗi bước nhỏ*).
2. **Terminated (va chạm):** chỉ khi `on_collision="terminate"`. Mặc định (`"stop"`) robot chỉ dừng lại, episode
   tiếp tục.
3. **Truncated:** hết `time_limit`. Mặc định `time_limit = time_factor × (thời gian ít nhất để đi đường ngắn
   nhất ở tốc độ tối đa + thời gian quay qua các góc cua)`.

Ở step cuối, `info` có thêm `success`, `spl`, `termination` (`"success"`, `"collision"` hoặc `"time_limit"`),
`path_length`, `time`, `n_collisions`, `geodesic_m`, `map_id`, `task_id`, `split`. `spl = success × ngắn nhất /
max(đã đi, ngắn nhất)`.

### Info

| Khoá | Khi nào | Ý nghĩa |
|---|---|---|
| `map_id`, `task_id`, `split`, `geodesic_m`, `time_limit` | `reset` | thông tin episode; `geodesic_m` = đường ngắn nhất start → goal |
| `collided`, `n_collisions` | mọi step | step này có va chạm không; số step có va chạm từ đầu |
| `time`, `pose`, `path_length` | mọi step | giây đã trôi; `(x, y, θ)`; mét đã đi |
| `goal_geodesic_m`, `progress_m` | mọi step (`nav_info`) | đường ngắn nhất tới đích bây giờ; mức giảm trong step này |
| `goal_distance_m`, `wall_distance_m` | mọi step (`nav_info`) | khoảng cách đường thẳng tới đích; tới tường gần nhất |

## Arguments

```python
import gymnasium as gym
import navsim

env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
```

| Tham số | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `dataset` | `str` | `"demo-svg"` | tên hoặc thư mục dataset loại `svg` |
| `robot` | `str` / `Robot` | robot duy nhất của dataset, nếu có nhiều thì `"turtlebot4"` | id robot (`navsim robots`) |
| `split` | `str` | `None` | `"train"`, `"val"`, `"test"` (chia theo tòa nhà); `None`: mọi bản đồ |
| `map_id`, `task_idx` | `str`, `int` | `None` | luôn dùng map / task này |
| `normalize_obs` | `bool` | `True` | `False`: quan sát theo mét, m/s, rad/s |
| `dt` | `float` | `0.1` | số giây mô phỏng mỗi `step()` |
| `n_beams` | `int` | `72` | số tia LiDAR (≥ 2) |
| `lidar_noise` | `float` | `0.0` | độ lệch chuẩn nhiễu Gauss của LiDAR (m) |
| `on_collision` | `str` | `"stop"` | `"stop"` hoặc `"terminate"` |
| `time_limit` | `float` | `None` | độ dài episode (s); `None`: `time_factor × thời gian tối thiểu` |
| `time_factor` | `float` | `3.0` | hệ số cho `time_limit` mặc định |
| `success_radius` | `float` | theo file task (0.2) | bán kính "tới đích" (m) |
| `task_filter` | `dict` | `None` | bộ lọc curriculum, xem [Chọn episode](../concepts/episodes.md) |
| `nav_info` | `bool` | `True` | thêm `goal_geodesic_m`, `progress_m`, `goal_distance_m`, `wall_distance_m` vào `info`; `False` nhanh hơn |
| `render_mode` | `str` | `None` | `"rgb_array"` hoặc `"human"` |
| `map_repeat` | `int` | `1` | giữ một map ngẫu nhiên cho bấy nhiêu episode liên tiếp |
| `map_cache`, `field_cache` | `int` | `8`, `4` | số bản đồ / bảng khoảng cách giữ trong RAM |
| `disk_cache` | `bool` | `True` | dùng cache bản đồ trên đĩa |
| `clearance` | `float` | `None` (theo file task, 0.05) | khoảng an toàn thêm quanh tường cho planner (m) |

## Vectorized environment

Môi trường tạo bằng `gym.make_vec`, mỗi bản sao một tiến trình. Với Stable-Baselines3 hoặc thư viện tạo env
trong tiến trình con, dùng `navsim.env_fn` (chạy được cả trên Windows và macOS):

```python
envs = gym.make_vec("NavSim/Nav-v0", num_envs=8, vectorization_mode="async", dataset="demo-svg")

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
venv = make_vec_env(navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg"),
                    n_envs=8, vec_env_cls=SubprocVecEnv)
```

Đổi bộ lọc curriculum trên mọi bản sao: `envs.call("set_task_filter", geodesic_m=(1.0, 5.0))`.

## Version History

- v0 (navsim 0.1.0): phát hành đầu tiên.

> Cơ chế chạy bên trong (sub-step, va chạm, LiDAR, planner): [internals/simulator.md](../internals/simulator.md#navsimnav-v0).
