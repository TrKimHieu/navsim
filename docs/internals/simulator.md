# Cơ chế bên trong của ba môi trường

Trang này giải thích simulator chạy thế nào. Nếu chỉ cần *dùng* môi trường, đọc các trang trong [environments/](../environments/README.md).

## `NavSim/Nav-v0`

*(`envs/nav.py` + `_sim/world.py`)* Một robot, bản đồ liên tục.

### `gym.make("NavSim/Nav-v0", **kwargs)`

Việc xảy ra khi khởi tạo:
1. Kiểm tra `render_mode ∈ {None, "human", "rgb_array"}`; `render_fps` = `round(1/dt)`.
2. Nạp dataset, từ chối dataset grid. Chọn robot (`default_robot`). Danh sách bản đồ = `ds.maps(robot, split)`;
   rỗng → `ValueError: no <split> maps for …`.
3. Tạo `MapLoader` (đọc cache đĩa nếu có), `World([robot], …)`.
4. Dựng `action_space` và `observation_space` (xem dưới).

**Không gian action:** `Box(-1,1,(2,))` = (tiến, quay), hoặc `(3,)` với robot omni. Hàm `_scale_action`
kẹp về `[-1,1]`, nhân với giới hạn: tiến ≥ 0 dùng `v_max`, tiến < 0 dùng `-v_min`.

**Không gian observation** (`Dict`):

| Khoá | `normalize_obs=True` | `False` |
|---|---|---|
| `lidar (n_beams,)` | khoảng cách / `range_max` ∈ [0,1] | mét ∈ [`range_min`,`range_max`] |
| `goal (3,)` | (min(d/range_max,1), sin, cos) | (d, sin, cos) |
| `velocity` | v / giới hạn ∈ [-1,1] | m/s, rad/s |

`sin`, `cos` là của góc `atan2(dy,dx) − theta` (hướng tới đích nhìn từ mũi robot): cos=1 đích ngay
trước mặt, sin>0 đích bên trái.

### `env.reset(*, seed=None, options=None)`

1. `super().reset(seed)`; nếu có `seed` thì sinh luồng ngẫu nhiên **riêng cho nhiễu LiDAR** từ seed đó
   (nên cùng seed → cùng nhiễu).
2. `_choose(options)` chọn bản đồ + task (xem [chọn episode](../concepts/episodes.md)).
3. `world.set_map` rồi `world.place`: kiểm tra tư thế xuất phát hợp lệ (`pose_valid`; sai →
   `RuntimeError`), đặt robot, đặt `time=0`, `steps=0`, ghi điểm đầu vào `traj`, tính khoảng cách tới đích
   ban đầu (`prev_geodesic`).
4. **Giới hạn thời gian** nếu không đặt `time_limit`:
   `time_factor × (geodesic/v_max + tổng góc quay dọc đường ngắn nhất / w_max)`. Robot omni không cần quay
   nên số hạng quay = 0. Đường ngắn nhất được làm mượt bằng Douglas–Peucker (dung sai 5 cm) trước khi
   đếm góc quay.
5. Trả `(obs, info)`; `info = {map_id, task_id, split, geodesic_m, time_limit}`.

### `env.step(action)`

Gọi `World.step`. Thứ tự:

1. Action → `(v, vy, w)` thật (m/s, rad/s).
2. **Chia nhỏ thời gian** `dt` thành các *sub-step* sao cho mỗi sub-step đi ≤ 2 cm và quay ≤ 2°
   (`n = ceil(max(|v|·dt/0.02, |w|·dt/2°))`). Nhờ vậy robot nhanh không "nhảy xuyên" qua tường mỏng.
3. Mỗi sub-step: tích phân **chính xác** tư thế với vận tốc không đổi (robot vi sai đi theo cung tròn).
4. Nếu tư thế mới không hợp lệ (đụng tường hoặc robot khác) → robot **đứng lại ở tư thế cũ** trong
   phần còn lại của step; `collided=True`.
5. **Kiểm tra đích sau mỗi sub-step** (không chỉ cuối step): Jackal đi 0,2 m/step, bằng cỡ vùng đích,
   nếu chỉ kiểm cuối step thì có thể lướt qua đích.
6. Cập nhật đồng hồ, `n_collisions`, `traj`; nếu `nav_info`: tính `goal_geodesic_m`,
   `progress_m = khoảng_cách_trước − khoảng_cách_sau`, `goal_distance_m`, `wall_distance_m`.
7. `terminated = tới_đích or (va_chạm and on_collision=="terminate")`;
   `truncated = (not terminated) and time ≥ time_limit`. Khi kết thúc, `info` thêm
   `success, spl, termination, path_length, time, n_collisions, geodesic_m, map_id, task_id, split`.

Reward trả về luôn **0.0**. Gọi `step` trước `reset` → `gymnasium.error.ResetNeeded`.

**Chú ý khi va chạm:** với `on_collision="stop"` (mặc định), velocity trong observation bị đặt về 0 ở
step va chạm, nhưng episode tiếp tục; robot có thể quay đi và đi tiếp.

**SPL:** `spl = success × geodesic / max(path_length, geodesic)` (0 nếu thất bại).

### Các hàm/thuộc tính khác

| Lệnh | Cơ chế |
|---|---|
| `env.render()` | `"rgb_array"`: ảnh bản đồ rasterise (0,04 m/px, nền được cache), đường đi, đích (hình tròn đỏ), robot (đa giác xanh). `"human"`: mở cửa sổ matplotlib và `plt.pause(1/render_fps)` |
| `env.close()` | đóng cửa sổ `human` nếu có |
| `env.unwrapped.robot_state()` | tạo `RobotState` ([RobotState](../concepts/robot_state.md)) |
| `env.unwrapped.map_ids / tasks_of / tasks / set_task_filter` | [chọn episode](../concepts/episodes.md) |
| `env.unwrapped.map` | `SvgMap` hiện tại (`.rings`, `.segments`, `.bbox`, `.raster(res)`) |
| `env.unwrapped.world`, `.agent` | nội bộ, có thể đổi giữa các phiên bản — nên dùng `robot_state()` |

### LiDAR (Nav-v0)

Tia đều quanh robot: nếu `fov` ≥ 2π thì 360° với bước `2π/n_beams`; ngược lại từ `-fov/2` với bước
`fov/(n_beams-1)`. Mỗi tia trả khoảng cách tới cạnh free gần nhất (hoặc `range_max`). Chỉ test mỗi
cạnh với các tia nằm trong góc nó che, nên rẻ hơn nhiều so với `n_beams × số_cạnh`. Nhiễu Gauss
(`lidar_noise`) được cộng **sau** khi quét, rồi kẹp về `[range_min, range_max]`. `state.lidar()` luôn là
bản **không nhiễu**.

## `NavSim/GridNav-v0`

*(`envs/grid.py`)* Một robot, bản đồ lưới; mỗi ô vừa một thân robot.

### `gym.make("NavSim/GridNav-v0", **kwargs)`

- Dataset phải là `grid`. Lưới: `0` = trống, khác 0 = chặn (khi truyền `grid=` tự cấp, mọi giá trị khác 0
  đều đổi thành 1).
- `difficulty` = `task_filter` ngầm `level=[difficulty]`; không được vừa đặt `difficulty` vừa lọc theo
  `level` (`ValueError`), và `set_task_filter(level=…)` bị từ chối khi đã có `difficulty`.
- Không gian: action `Discrete(4)`; observation = `Box(0, lidar_range, (lidar_rays,))`, hoặc
  `Dict(lidar, goal:(2,))` khi `goal_obs=True`. `normalize_obs=True` chia cả hai cho `lidar_range`.

### `reset` / `step`

- `reset`: chọn bản đồ/task ([chọn episode](../concepts/episodes.md)); với start/goal tự chọn, kiểm tra
  trong lưới, ô trống, khác nhau, và nối được bằng BFS (không thì `ValueError`). Tính bảng BFS "số nước
  còn lại tới đích" (giữ 8 bảng gần nhất). `info` có `start, goal, d_bfs, p0, sa_uniform, sa_persist,
  starved, level, map_id, task_id, split`.
- `step(action)`: ánh xạ `0↑ 1↓ 2← 3→` thành `(Δhàng, Δcột)`. Nếu ô đích ra ngoài lưới hoặc bị chặn thì
  robot **đứng yên**, `collided=True`, `n_collisions += 1` (và **không** tính vào `path_length`). Ngược
  lại di chuyển và `moves += 1`.
  `terminated = (đang ở ô đích)`; `truncated = (không terminated) and steps ≥ budget`.
  `progress = số_nước_còn_lại_trước − số_nước_còn_lại_sau`.
  Step cuối: `path_length = moves × cell_m` (mét), `time = steps` (step), `geodesic_m = d_bfs × cell_m`.

### LiDAR lưới (`lidar_scan`)

`lidar_rays` tia ở góc `linspace(0, 2π, n, endpoint=False)`, đi từ chỉ số ô `(row, col)`, lấy mẫu mỗi 0,1
ô, dừng ở biên lưới hoặc ô `==1`; không chạm gì thì trả `max_range`.

- **Mặc định** (`lidar_centered=False`) tái hiện đúng LiDAR của công trình curriculum gốc: tia xuất phát
  từ **góc trên-trái** của ô và cắt về 0, nên giữa phòng vuông đọc 10,0 ô theo hướng xuống/phải nhưng 9,1 ô
  theo hướng lên/trái (không đối xứng).
- `lidar_centered=True`: xuất phát từ tâm ô, lấy sàn, mỗi bước lấy mẫu ở giữa → đối xứng.
- Kết quả phụ thuộc **chỉ vào ô**, nên mỗi ô được quét một lần rồi lưu (bảng cache theo
  `(rays, range, centered)`).

Hàm dùng độc lập: `navsim.envs.grid.lidar_scan(grid, position, n_rays, max_range, centered)` và
`navsim.envs.grid.bfs(free, start)` (số nước ít nhất tới mọi ô, `-1` = không tới được).

> **Lưu ý đánh giá:** policy "greedy" trên grid có thể kẹt giữa hai ô chỉ vào nhau, vì observation chỉ
> phụ thuộc ô. `evaluate` vì thế **lấy mẫu** action trên `GridNav-v0` theo mặc định.

## `navsim.multiagent.nav_v0`

*(`multiagent/nav_v0.py`)* Nhiều robot trên một bản đồ; cần `navsim[marl]`.

### `nav_v0.parallel_env(**kwargs)` · `nav_v0.env(**kwargs)` · `nav_v0.raw_env`

- `parallel_env` trả `NavParallelEnv` (PettingZoo `ParallelEnv`, mọi robot hành động cùng lúc).
- `env` bọc nó bằng `parallel_to_aec` (từng robot lần lượt). `raw_env` là tên khác của lớp.

**Khởi tạo:** nhận tham số của `Nav-v0` (trừ `task_idx`) cùng `n_agents, robots, flatten_obs, on_done,
separation, max_tries`.
- Đội hình: `robots=[…]` hoặc `n_agents` bản sao của `robot`. **Mọi robot phải cùng `height_class`**,
  nếu không `ValueError` (vì khác bộ bản đồ).
- **Task được chọn theo robot có `r_circ` lớn nhất**, nên hợp với mọi robot trong đội. Planner và cache
  cũng dùng bán kính lớn nhất.
- Tên agent: `robot_0 … robot_{n-1}`; `possible_agents` là tất cả, `agents` là những robot còn hoạt động.

### `reset(seed=None, options=None)`

1. Seed: tạo luồng nhiễu LiDAR riêng cho từng robot (`seed_seq.spawn(n)`).
2. `options` hợp lệ: `map_id`; `task_idx` = list **đúng n phần tử** (`None` = ngẫu nhiên); hoặc `start` +
   `goal` = list đúng n phần tử. Sai độ dài → `ValueError`.
3. Đặt từng robot: với task ngẫu nhiên, thử tối đa `max_tries` lần mỗi robot; một ứng viên bị loại nếu
   start/goal cách start/goal của robot đã đặt **< `r_i + r_j + separation`**, tư thế xuất phát không
   hợp lệ, hoặc thân chồng lên robot khác. Không đặt được → thử bản đồ khác (tối đa 10 lần nếu bản đồ
   không cố định) → cuối cùng `RuntimeError: could not place n robots…` (gợi ý: ít robot hơn, `separation`
   nhỏ hơn, bản đồ khác). Với task/start/goal chỉ định thì kiểm tra nghiêm ngặt (`ValueError`).
4. Trả `({agent: obs}, {agent: info})`.

### `step(actions)`

- `actions` là dict `{agent: action}`; agent thiếu trong dict được cho action **0**.
- Mô phỏng như `Nav-v0` nhưng **va chạm robot–robot** cũng bị xử lý: robot bị chặn nếu tư thế mới chồng
  lên robot đứng yên hoặc lên tư thế mới của robot khác trong cùng sub-step.
- LiDAR của mỗi robot thấy cả thân các robot khác (trong tầm).
- Trả `(obs, rewards, terminations, truncations, infos)`, mỗi cái là dict theo agent; `rewards` luôn 0.0.
  Robot kết thúc bị loại khỏi `env.agents`; episode xong khi `env.agents` rỗng.
- `on_done="park"`: robot xong vẫn là chướng ngại tại chỗ; `"remove"`: biến mất khỏi bản đồ.

### Các hàm khác

| Lệnh | Cơ chế |
|---|---|
| `env.robot_state(agent)` | `RobotState` của một robot |
| `env.state()` | vector toàn cục cho critic tập trung: 10 số/robot `[x, y (chuẩn hoá theo bbox bản đồ), cosθ, sinθ, v, vy, w (chia giới hạn), goal x, goal y (chuẩn hoá như x,y), active]`, đã kẹp vào `state_space` |
| `env.state_space` | `Box` tương ứng |
| `env.observation_space(a)` / `env.action_space(a)` | không gian riêng của từng agent; `flatten_obs=True` dùng `spaces.flatten_space` (một `Box` phẳng, thứ tự: goal, lidar, velocity theo `Dict` đã sắp xếp của Gymnasium) |
| `env.tasks_of(map_id)`, `env.map_ids` | như [chọn episode](../concepts/episodes.md) |
| `nav_v0.ParallelReward(env, fn)` | [wrappers](../api/wrappers.md) |

```python
from navsim.multiagent import nav_v0
env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=4, split="train")
obs, infos = env.reset(seed=0)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, infos = env.step(actions)
```
