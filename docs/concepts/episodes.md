# Chọn episode và curriculum

*(`envs/episodes.py`)*

## Luật chọn tại `reset` (luật đầu tiên áp dụng được sẽ thắng)

| # | Điều kiện | Kết quả |
|---|---|---|
| 1 | `options` có `start` **và** `goal` (thiếu một trong hai → `ValueError`) | episode tự chọn trên `map_id`; `task_id = -1`. Có nhiều bản đồ mà không cho `map_id` → `ValueError` |
| 2 | `options` có `task_idx` | task `i` (lấy `i % số_task`) của `map_id` |
| 3 | `map_id` / `task_idx` truyền lúc `gym.make` | cố định suốt đời môi trường |
| 4 | còn lại | bản đồ ngẫu nhiên của split, rồi task ngẫu nhiên (đều dùng `np_random` đã seed) |

**Start/goal tự chọn** (`Nav-v0`): `start=(x, y[, theta])`, `goal=(x, y)`; không cho `theta` thì robot quay mặt về
phía đích. Hợp lệ khi: thân robot vừa tại start; đích nằm trong vùng free của planner (cách tường ≥ bán kính
robot + `clearance`); và đích đến được từ start. Mỗi điều kiện có `ValueError` riêng, nói rõ lý do
(đè tường / sát tường / "another room, or a passage too narrow").

## Bản đồ trong bộ nhớ

Giữ `map_cache` bản đồ gần nhất (LRU). `map_repeat=K` giữ một bản đồ ngẫu nhiên trong K episode liên
tiếp (task vẫn đổi) → ít lần nạp bản đồ hơn K lần, đổi lại các episode liên tiếp giống nhau hơn.
`reset(seed=s)` luôn bắt đầu lại chuỗi (kể cả đoạn `map_repeat`), nên cùng seed → cùng episode.

## `env.unwrapped.set_task_filter(**criteria)`

Chỉ ảnh hưởng việc chọn **ngẫu nhiên** (luật 4). Mỗi tiêu chí được tìm trong task (`geodesic_m`,
`euclidean_m`) rồi trong `labels` (svg: `gdr`; grid: `d_bfs`, `level`, …); **task thiếu tiêu chí thì không khớp**.
Kiểu giá trị:

| Viết | Nghĩa |
|---|---|
| `(lo, hi)` | khoảng, gồm cả hai đầu; `None` = không chặn |
| `{"min": a, "max": b}` | như trên, dạng dict (khoá lạ → `ValueError`) |
| `[v1, v2]` / `set` | giá trị được phép |
| `v` | bằng đúng `v` |

Trả `{"maps": n, "tasks": n}`; không còn task nào → `ValueError: no task matches …`. Gọi không đối số = bỏ
lọc. Một bản đồ cố định bằng `map_id` mà không có task khớp sẽ lỗi lúc `reset`, trừ khi bạn cũng cố định
`task_idx`. Với vector env: `venv.call("set_task_filter", …)` (Gymnasium) hoặc
`venv.env_method("set_task_filter", …)` (SB3).
