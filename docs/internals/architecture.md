# Kiến trúc tổng quan

```
gym.make("NavSim/Nav-v0")          ──►  NavEnv        (envs/nav.py)        ┐
nav_v0.parallel_env(n_agents=4)    ──►  NavParallelEnv(multiagent/nav_v0.py)├─► World (_sim/world.py)
gym.make("NavSim/GridNav-v0")      ──►  GridNavEnv    (envs/grid.py)  (độc lập, chỉ dùng BFS + LiDAR lưới)
```

- **`World`** là bộ mô phỏng thật: giữ N robot trên một bản đồ, tiến mô phỏng, dò va chạm, quét
  LiDAR, tính đường ngắn nhất. `NavEnv` là `World` với 1 robot, `NavParallelEnv` là `World` với N robot.
- **`EpisodeSource`** (`envs/episodes.py`) là mixin dùng chung cho cả ba môi trường: chọn bản đồ/nhiệm
  vụ, giữ cache bản đồ, bộ lọc curriculum.
- **Môi trường không có reward.** `step()` luôn trả `0.0`. Bạn tự viết reward từ `RobotState` bằng
  `wrappers.Reward`.
- **Bản đồ liên tục** (svg) được đọc thành vùng "free" (đa giác, quy tắc even-odd). Robot hợp lệ khi
  đa giác thân robot không cắt cạnh nào của vùng free và tâm robot nằm trong free. **Chạm cạnh cũng
  tính là va chạm.**
- **Planner** (`_sim/planning.py`) rasterise bản đồ thành lưới ô 4 cm; ô là "free" khi đĩa bán kính
  (bán kính robot + `clearance`) không chạm tường. Mỗi ô nối 16 láng giềng (4 thẳng, 4 chéo, 8 nước
  "mã") không cắt góc, nên khoảng cách lưới dài hơn đường thẳng tối đa ~2,8%. Khoảng cách tới đích
  tính bằng Dijkstra.
