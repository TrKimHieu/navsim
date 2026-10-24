# Tài liệu NavSim

NavSim là môi trường điều hướng robot cho học tăng cường (Gymnasium cho một robot, PettingZoo cho nhiều
robot). Tổng quan và cài đặt nhanh: [README.vi.md](../README.vi.md) ở thư mục gốc. Trang này chỉ đường tới
đúng tài liệu bạn cần.

Quy ước: đơn vị là **mét, giây, radian** (grid: **ô**); góc đo ngược chiều kim đồng hồ từ trục +x (REP-103).

## Bạn muốn làm gì?

| Mục tiêu | Đọc |
|---|---|
| Cài đặt, chạy ví dụ đầu tiên | [guide/getting-started.md](guide/getting-started.md) |
| Chọn và dùng một môi trường | [environments/](environments/README.md) |
| Viết reward, đọc trạng thái robot | [concepts/robot_state.md](concepts/robot_state.md), [api/wrappers.md](api/wrappers.md) |
| Chọn bản đồ/nhiệm vụ, học theo curriculum | [concepts/episodes.md](concepts/episodes.md) |
| Đánh giá policy, vẽ ảnh, ghi GIF | [api/evaluation.md](api/evaluation.md), [api/visualization.md](api/visualization.md) |
| Chạy nhiều môi trường song song | [api/parallel.md](api/parallel.md) |
| Công thức hay dùng (SB3, curriculum, …) | [guide/recipes.md](guide/recipes.md) |
| Gặp lỗi | [guide/troubleshooting.md](guide/troubleshooting.md) |
| Hiểu dữ liệu bên trong (dataset, `.npz`, `.svg`), LiDAR, một lần `step()` | [internals/](internals/overview.md): [dataset-format](internals/dataset-format.md), [lidar](internals/lidar.md), [step-and-planning](internals/step-and-planning.md), [robots-and-tasks](internals/robots-and-tasks.md) |
| Xem toàn bộ lệnh dạng cây | [navsim-api-tree.html](navsim-api-tree.html) (mở bằng trình duyệt) |

## Cấu trúc thư mục

```
docs/
  guide/          hướng dẫn: getting-started, recipes, troubleshooting
  environments/   mỗi môi trường một trang (khuôn như trang Gymnasium): nav_v0, grid_nav_v0, multiagent_nav_v0
  concepts/       khái niệm dùng chung: episodes (chọn episode, curriculum), robot_state
  api/            tra cứu theo nhóm lệnh: datasets, robots, wrappers, evaluation, visualization,
                  parallel, cli, environment-variables
  internals/      bên trong: overview, architecture, dataset-format, lidar, step-and-planning,
                  robots-and-tasks, simulator (cơ chế từng môi trường), code-layout, maintainers
  navsim-api-tree.html   cây lệnh tương tác
```
