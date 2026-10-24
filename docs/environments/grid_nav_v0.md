# Grid Nav

Thuộc nhóm môi trường NavSim: robot điều hướng trên lưới ô.

| | |
|---|---|
| Action Space | `Discrete(4)` |
| Observation Space | `Box(0.0, lidar_range, (lidar_rays,), float32)`; với `goal_obs=True`: `Dict("lidar": Box(...), "goal": Box(-inf, inf, (2,)))` |
| import | `gymnasium.make("NavSim/GridNav-v0")` |

## Description

Một robot di chuyển **từng ô** trên bản đồ lưới lấy từ mặt bằng tòa nhà thật (dữ liệu loại `grid`). Một ô
vừa đúng một thân robot; mỗi step robot đi một ô theo một trong bốn hướng. Quan sát là LiDAR (khoảng cách
tính bằng ô) và, nếu bật `goal_obs`, vị trí tương đối của đích.

Mặc định **robot không biết đích ở đâu** (bài toán tìm kiếm). Muốn lái *tới* đích, bật `goal_obs=True`.
Giống `Nav-v0`, môi trường không có reward: xem [Rewards](#rewards).

## Action Space

| Giá trị | Hành động |
|---|---|
| 0 | lên (hàng − 1) |
| 1 | xuống (hàng + 1) |
| 2 | trái (cột − 1) |
| 3 | phải (cột + 1) |

**Ghi chú:** nước đi vào tường hoặc ra ngoài lưới thì robot **đứng yên**; step đó tính là va chạm
(`collided=True`, `n_collisions += 1`) và không tính vào `path_length`. Hàng tăng đi xuống, cột tăng sang phải.

## Observation Space

| `goal_obs` | Quan sát |
|---|---|
| `False` | `Box(0, lidar_range, (lidar_rays,))`: khoảng cách LiDAR theo ô |
| `True` | `Dict("lidar": Box(...), "goal": Box(2,))`, `goal = (số hàng, số cột)` từ robot tới đích |

`normalize_obs=True` chia cả `lidar` và `goal` cho `lidar_range`.

**Ghi chú về LiDAR:** `lidar_rays` tia ở các góc `linspace(0, 2π, n, endpoint=False)`, lấy mẫu mỗi 0,1 ô,
dừng ở biên lưới hoặc ô bị chặn; tia không chạm gì trả về `lidar_range`. Mặc định (`lidar_centered=False`) tia
xuất phát từ **góc trên-trái** của ô như công trình curriculum gốc, nên đọc xa hơn về phía dưới/phải (10,0 so
với 9,1 ô ở giữa phòng vuông). `lidar_centered=True` xuất phát từ tâm ô, đối xứng. Kết quả chỉ phụ thuộc vào
ô, nên mỗi ô được quét một lần rồi lưu lại.

## Rewards

Luôn `0.0`. Tự viết bằng `navsim.wrappers.Reward(env, fn)`:

```python
# +0.1 mỗi nước lại gần đích hơn, +1 khi tới đích
grid_reward = lambda state, info: 0.1 * info["progress"] + state.reached_goal
env = navsim.wrappers.Reward(gym.make("NavSim/GridNav-v0", goal_obs=True), grid_reward)
```

`info["progress"]` là mức giảm số nước còn lại tới đích trong step này (âm nếu đi xa ra).

## Starting State

Giống `Nav-v0`: bản đồ ngẫu nhiên của `split`, rồi task ngẫu nhiên (có seed). Start và goal là các ô
`(hàng, cột)`. Chọn cụ thể bằng `reset(options={"map_id": m, "task_idx": i})`, hoặc start/goal tự chọn
`options={"map_id": m, "start": (r, c), "goal": (r, c)}` (phải nằm trong lưới, ô trống, khác nhau và nối được
bằng BFS; `task_id = -1`). Tham số `difficulty` chỉ giữ các task của một mức.

## Episode End

1. **Terminated:** robot đứng trên ô đích.
2. **Truncated:** quá `budget` step.

Ở step cuối, `info` có thêm `success`, `spl`, `termination` (`"success"` hoặc `"time_limit"`), `path_length`
(mét = số nước × `cell_m`), `time` (số step), `n_collisions`, `geodesic_m`, `map_id`, `task_id`, `split`.

### Info

| Khoá | Khi nào | Ý nghĩa |
|---|---|---|
| `start`, `goal`, `d_bfs`, `level`, `p0`, `sa_uniform`, `sa_persist`, `starved`, `map_id`, `task_id`, `split` | `reset` | `d_bfs` = số nước ít nhất start → goal; các nhãn còn lại mô tả độ khó của task |
| `steps`, `collided`, `d_bfs` | mọi step | số step; nước vừa rồi bị chặn |
| `goal_geodesic`, `progress` | mọi step | số nước ít nhất tới đích bây giờ; mức giảm trong step này |

## Arguments

```python
env = gym.make("NavSim/GridNav-v0", dataset="demo-grid", goal_obs=True)
```

| Tham số | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `dataset` | `str` | `"demo-grid"` | tên hoặc thư mục dataset loại `grid` |
| `robot`, `split`, `map_id`, `task_idx`, `task_filter`, `render_mode`, `map_repeat` | | | như `Nav-v0` |
| `goal_obs` | `bool` | `False` | thêm vị trí đích vào quan sát |
| `normalize_obs` | `bool` | `False` | chia LiDAR và `goal` cho `lidar_range` |
| `budget` | `int` | `200` | số step tối đa trước khi `truncated` |
| `lidar_rays` | `int` | `22` | số tia LiDAR |
| `lidar_range` | `float` | `10.0` | tầm tối đa (ô) |
| `lidar_centered` | `bool` | `False` | `True`: tia xuất phát từ tâm ô |
| `difficulty` | `str` | `None` | `"easy"`, `"medium"`, `"hard"`, `"starved"` |
| `map_cache` | `int` | `256` | số bản đồ giữ trong RAM |
| `grid`, `tasks`, `cell_m` | | `None`, `None`, `1.0` | truyền lưới có sẵn trong bộ nhớ (ô khác 0 là vật cản) |

## Vectorized environment

```python
envs = gym.make_vec("NavSim/GridNav-v0", num_envs=8, vectorization_mode="async", goal_obs=True)
```

**Ghi chú khi đánh giá:** quan sát chỉ phụ thuộc ô, nên policy "greedy" có thể kẹt giữa hai ô chỉ vào nhau.
`navsim.evaluate` vì thế mặc định **lấy mẫu** action trên môi trường này.

## Version History

- v0 (navsim 0.1.0): phát hành đầu tiên.

> Cơ chế chạy bên trong (sub-step, va chạm, LiDAR, planner): [internals/simulator.md](../internals/simulator.md#navsimgridnav-v0).
