# Môi trường

NavSim có ba môi trường. Chọn theo số robot và loại bản đồ:

| | [`NavSim/Nav-v0`](nav_v0.md) | [`NavSim/GridNav-v0`](grid_nav_v0.md) | [`navsim.multiagent.nav_v0`](multiagent_nav_v0.md) |
|---|---|---|---|
| Thế giới | mặt bằng, đơn vị mét | lưới, một ô = một thân robot | mặt bằng, nhiều robot |
| API | Gymnasium | Gymnasium | PettingZoo `ParallelEnv` (và AEC) |
| Tạo | `gym.make("NavSim/Nav-v0", ...)` | `gym.make("NavSim/GridNav-v0", ...)` | `nav_v0.parallel_env(...)` |
| Dataset | loại `svg` | loại `grid` | loại `svg` |
| Reward | 0.0 | 0.0 | 0.0 mỗi robot |

Dùng chung cho cả ba: [chọn episode và curriculum](../concepts/episodes.md), [`RobotState`](../concepts/robot_state.md)
(để viết reward), [wrappers](../api/wrappers.md).

## Thuộc tính khác của môi trường

Truy cập qua `env.unwrapped`:

| Thuộc tính | Môi trường | Ý nghĩa |
|---|---|---|
| `map` | `Nav-v0` | `SvgMap` hiện tại: `.rings` (các đa giác), `.segments` (mảng `(N,4)` các cạnh), `.bbox`, `.meta`, `.raster(res=0.02, pad=0.2)` → `(mask, x0, y1, res)` |
| `robot` | `Nav-v0` | mô hình `Robot` |
| `pose` | `Nav-v0` | `(x, y, theta)` |
| `task` | `Nav-v0` | task hiện tại |
| `tasks` | cả hai | task của map hiện tại |
| `dt` | `Nav-v0` | số giây mỗi step |
| `beam_angles` | `Nav-v0` | hướng các beam LiDAR |
| `world`, `agent` | `Nav-v0` | bộ mô phỏng bên trong (`World`) và trạng thái nội bộ của robot |
| `grid` | `GridNav-v0` | grid hiện tại (0 = trống, khác 0 = chặn) |
| `agent_pos`, `start`, `goal` | `GridNav-v0` | `[hàng, cột]` |
| `lidar_ranges()` | `GridNav-v0` | LiDAR ở ô hiện tại (ô) |
| `rgb_frame(cell=8)` | `GridNav-v0` | ảnh grid (sàn trắng, tường đen, goal đỏ, robot xanh) |
| `map_id`, `split` | cả hai | map hiện tại; split |
| `render()` | cả hai | ảnh nếu `render_mode="rgb_array"` |
| `close()` | cả hai | đóng cửa sổ (nếu có) |

`world`, `agent` và module `navsim._sim` là phần bên trong, có thể đổi giữa các phiên bản;
nên dùng `robot_state()` thay vì dùng trực tiếp chúng.
