# Multi-robot Nav

Thuộc nhóm môi trường NavSim: nhiều robot điều hướng trên cùng một mặt bằng (PettingZoo).

| | |
|---|---|
| Actions | liên tục, `Box(-1.0, 1.0, (2,), float32)` mỗi robot; đa hướng: `(3,)` |
| Parallel API | có |
| Manual Control | không |
| Agents | `n_agents` (mặc định 2), tên `robot_0 … robot_{n-1}` |
| Observation Shape | mỗi robot: `Dict("lidar", "goal", "velocity")`; `flatten_obs=True`: một `Box` phẳng |
| import | `from navsim.multiagent import nav_v0` |

Cần cài `pip install "navsim[marl]"`.

## Description

Nhiều robot cùng lái trên **một bản đồ** của dữ liệu `svg`, mỗi robot có start và đích riêng. Mỗi robot quan
sát và hành động đúng như trong [`Nav-v0`](nav_v0.md); thêm vào đó **các robot nhìn thấy nhau trong
LiDAR và va chạm được với nhau**. Mọi robot hành động cùng lúc (Parallel API); `nav_v0.env(...)` cung cấp
cùng môi trường theo API AEC.

Mỗi lần `reset` chọn một bản đồ của split, rồi một task cho mỗi robot sao cho start/goal của hai robot
không gần hơn `r_i + r_j + separation` mét. Task được chọn theo robot **lớn nhất** nên hợp với cả đội.

```python
from navsim.multiagent import nav_v0

env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=4, split="train")
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
```

## Action Space

Như `Nav-v0` cho từng robot: `env.action_space(agent)`. Robot thiếu trong dict `actions` nhận action `0`
(đứng yên).

## Observation Space

`env.observation_space(agent)`: `Dict` như `Nav-v0` (`lidar`, `goal`, `velocity`), kể cả khi đội gồm các mẫu
robot khác nhau (mỗi robot theo giới hạn của nó). Với `flatten_obs=True` mỗi robot nhận **một `Box` phẳng**
(ghép theo thứ tự khoá `goal`, `lidar`, `velocity`), dùng cho SuperSuit hoặc BenchMARL.

### Global state

`env.state()` trả vector cho critic tập trung (MAPPO…), kích thước `10 × số robot`; `env.state_space` là
không gian của nó. Mỗi robot đóng góp 10 giá trị:

| Chỉ số | Giá trị |
|---|---|
| 0, 1 | `x`, `y` (chuẩn hoá theo khung bản đồ về `[0, 1]`) |
| 2, 3 | `cos θ`, `sin θ` |
| 4, 5, 6 | `v`, `vy`, `w` (chia cho giới hạn; `vy = 0` với robot vi sai) |
| 7, 8 | `x`, `y` của đích (chuẩn hoá như `x`, `y`) |
| 9 | còn hoạt động (1) hay đã xong (0) |

## Rewards

Luôn `0.0` cho mọi robot. Dùng `navsim.wrappers.ParallelReward(env, fn)` (hoặc `nav_v0.ParallelReward`):
`rewards[a] = fn(env.robot_state(a), infos[a])` cho từng robot.

```python
team = navsim.wrappers.ParallelReward(nav_v0.parallel_env(n_agents=4),
                                      lambda state, info: info["progress_m"] + 10 * state.reached_goal)
```

## Starting State

| Lệnh | Tác dụng |
|---|---|
| `env.reset(seed=s)` | map ngẫu nhiên; mỗi robot một task ngẫu nhiên thỏa điều kiện khoảng cách |
| `options={"map_id": m}` | cố định bản đồ |
| `options={"task_idx": [i0, None, ...]}` | mỗi robot một chỉ số task (`None` = ngẫu nhiên); danh sách phải đủ `n` phần tử |
| `options={"start": [...], "goal": [...]}` | mỗi robot một `(x, y[, θ])` và `(x, y)`; đủ `n` phần tử |

Với mỗi robot ngẫu nhiên, thử tối đa `max_tries` lần; không đặt được thì thử bản đồ khác (tối đa 10 lần nếu
bản đồ không cố định), cuối cùng `RuntimeError` (gợi ý: ít robot hơn, `separation` nhỏ hơn, bản đồ khác).

## Episode End

Mỗi robot kết thúc như trong `Nav-v0` (tới đích, hết `time_limit`, hoặc va chạm với
`on_collision="terminate"`). Robot đã xong **rời `env.agents`**:

- `on_done="park"` (mặc định): thân robot vẫn đứng đó làm chướng ngại cho robot khác.
- `on_done="remove"`: robot biến mất khỏi bản đồ.

Cả episode kết thúc khi `env.agents` rỗng. `info` của từng robot như `Nav-v0`.

## Arguments

| Tham số | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| (như `Nav-v0`, trừ `task_idx`) | | | `dataset`, `robot`, `split`, `map_id`, `normalize_obs`, `dt`, `n_beams`, `lidar_noise`, `on_collision`, `time_limit`, `time_factor`, `success_radius`, `task_filter`, `nav_info`, `render_mode`, `map_repeat`, `map_cache`, `disk_cache`, `clearance` |
| `n_agents` | `int` | `2` | số robot, cùng mẫu `robot` (≥ 1) |
| `robots` | `list` | `None` | đội hỗn hợp, ví dụ `["turtlebot4", "clearpath_jackal"]`; phải **cùng `height_class`** |
| `flatten_obs` | `bool` | `False` | một `Box` phẳng mỗi robot |
| `on_done` | `str` | `"park"` | `"park"` hoặc `"remove"` |
| `separation` | `float` | `0.1` | khoảng cách thêm (m) giữa start/goal của các robot khác nhau |
| `max_tries` | `int` | `50` | số lần bốc ngẫu nhiên mỗi robot trước khi đổi bản đồ |
| `field_cache` | `int` | tự động | số bảng khoảng cách giữ trong RAM |

Các hàm của đối tượng: `reset`, `step`, `robot_state(agent)`, `state()`, `observation_space(agent)`,
`action_space(agent)`, `tasks_of(map_id)`, `map_ids`, `render()`, `close()`. `nav_v0.env(...)` trả bản AEC;
`nav_v0.raw_env` là lớp môi trường.

## Version History

- v0 (navsim 0.1.0): phát hành đầu tiên.

> Cơ chế chạy bên trong (sub-step, va chạm, LiDAR, planner): [internals/simulator.md](../internals/simulator.md#navsimmultiagentnav_v0).
