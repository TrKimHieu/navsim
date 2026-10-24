# Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| `NamespaceNotFound: Namespace NavSim not found` | chưa `import navsim` (hoặc tiến trình con mới): thêm import, hoặc dùng `navsim.env_fn` |
| `dataset X is a grid dataset: use NavSim/GridNav-v0` | dùng sai môi trường so với `env_type` |
| `ImportError: navsim.multiagent needs PettingZoo` | `pip install "navsim[marl]"` |
| `PermissionError: no access to …` | repo gated: xin quyền trên trang dataset, đặt token *Read* vào `HF_TOKEN` |
| `ConnectionError: cannot reach Hugging Face` | kiểm tra mạng (Kaggle: bật Internet) |
| `no test maps for …` | split đó không có bản đồ cho robot này; xem `navsim info <dataset>` |
| `start … collides with the map` / `goal … too close to a wall` / `cannot be reached` | start/goal tự chọn không hợp lệ (xem [chọn episode](../concepts/episodes.md)) |
| `no task matches {…}` | `set_task_filter` loại hết task |
| `could not place n robots on …` | quá nhiều robot hoặc `separation` quá lớn so với bản đồ |
| `the robots use different maps (height classes …)` | đội robot khác `height_class` |
| `RuntimeError: the robot is not on the map` | dùng `RobotState` trước `reset` |
| `ResetNeeded` | gọi `step()` trước `reset()` |
| `SVG path command 'l' …` | SVG tự vẽ chỉ được dùng `M`, `L`, `Z` tuyệt đối |
| `GridNav-v0` điểm đánh giá thấp bất thường | policy greedy kẹt giữa hai ô; đừng ép `deterministic=True` |
| reward luôn 0 | đúng thiết kế: bọc `navsim.wrappers.Reward` |
| chậm | cài `navsim[fast]`; `nav_info=False` nếu không dùng; tăng `map_repeat`; tránh `style="plot"` khi ghi nhiều khung |
