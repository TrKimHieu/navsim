# `RobotState`

*(`_sim/world.py` cho Nav/nav_v0; `envs/grid.py` cho `GridRobotState`)*

Lấy bằng `env.unwrapped.robot_state()` (Nav/Grid) hoặc `env.robot_state(agent)` (nav_v0). Đối tượng **chỉ giữ
tham chiếu** tới môi trường nên luôn phản ánh trạng thái mới nhất: lấy một lần rồi dùng suốt cũng được.
Gọi khi robot chưa được đặt (chưa `reset`) → `RuntimeError: the robot is not on the map: reset the environment first`.

## Thuộc tính (đọc ngay, O(1))

| Thuộc tính | Nav / nav_v0 | Grid |
|---|---|---|
| `position`, `pose`, `heading`, `velocity` | `(x,y)`; `(x,y,θ)`; θ; `(v,w)` hoặc `(v,vy,w)` | `position` = `(hàng,cột)`; không có phần còn lại |
| `start`, `goal` | `(x,y,θ)`; `(x,y)` | `(hàng,cột)` |
| `start_geodesic` | đường ngắn nhất start→goal (m) | số nước ít nhất |
| `trajectory` | `(N,4)`: `time,x,y,θ`; hàng 0 là điểm xuất phát | `(N,2)`: `hàng,cột` |
| `path_length`, `steps`, `n_collisions` | mét đã đi, số step, số step va chạm | số nước đã đi (nước bị chặn **không** tính), số step, số nước bị chặn |
| `collided`, `reached_goal`, `done` | step vừa rồi có va chạm; đang ở đích; episode đã xong | tương tự |
| `time`, `time_limit`, `success_radius`, `robot`, `beam_angles` | có | không có (Grid có `budget`, `cell_m`) |

## Hàm (tính khi gọi)

| Hàm | Cơ chế & chi phí |
|---|---|
| `goal_distance(point=None)` | `hypot` tới đích (grid: theo ô). Rẻ |
| `geodesic_distance(point=None)` | tra bảng khoảng cách-tới-đích (Dijkstra, giữ `field_cache` bảng gần nhất theo task) tại ô free gần `point` nhất, **cộng đoạn thẳng ngắn tới tâm ô**. `inf` nếu không tới được. Lần đầu mỗi task mới tốn một lần Dijkstra |
| `shortest_path(point=None)` | `(M,2)` điểm từ `point` tới đích, đã **khử bậc thang** bằng Douglas–Peucker; mảng rỗng nếu đích không tới được. Dùng cây tiền nhiệm Dijkstra (cache theo task) |
| `wall_distance(point=None)` | khoảng cách **chính xác** tới cạnh free gần nhất (m). Chỉ có ở Nav/nav_v0 |
| `lidar()` | quét **không nhiễu** từ tư thế hiện tại, kẹp `[range_min,range_max]`. Grid: khoảng cách theo ô |
| `fits(x,y,θ)` | thân robot có vừa không (Nav) |
| `is_free(cell)` | ô trong lưới và trống (Grid) |

**Vì sao `geodesic_distance` đôi khi không phải nghiêm ngặt "đường thật":** vùng free của planner hơi
bảo thủ hơn thân thật, nên robot có thể đứng ở nơi mà ô free gần nhất là một "đảo" nhỏ không nối được tới
đích. Khi đó hàm dùng ô *gần nhất có khoảng cách hữu hạn* để kết quả vẫn hữu hạn.
