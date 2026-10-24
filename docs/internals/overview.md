# Tổng quan luồng hoạt động

```text
  DATASET (trên đĩa)                        NAVSIM (khi chạy)                      POLICY
 ┌──────────────────────┐
 │ maps/*.svg  hoặc     │  load  ┌─────────────────────────┐
 │ grids/*.npz          │───────▶│ Map (tường, vật cản)    │
 ├──────────────────────┤        └────────────┬────────────┘
 │ robots/*.json        │───────▶ Robot (footprint, giới hạn tốc độ, LiDAR)
 ├──────────────────────┤                     │
 │ tasks/*/*.json       │───────▶ Task: start + goal
 └──────────────────────┘                     ▼
                                   ┌─────────────────────┐   observation    ┌────────────┐
                                   │ World               │ ───────────────▶ │  policy    │
                                   │  • ray casting      │   lidar, goal,   │  của bạn   │
                                   │  • cập nhật pose    │   velocity       │            │
                                   │  • kiểm tra collision│ ◀─────────────── │            │
                                   │  • kiểm tra goal    │   action         └────────────┘
                                   └─────────────────────┘
```

Một episode gồm các bước:

1. `reset()` chọn một map và một task (start, goal), rồi đặt robot tại start.
2. Policy nhận observation (khoảng cách LiDAR, hướng và khoảng cách tới goal, velocity hiện tại) và
   trả về action (tốc độ tiến, tốc độ quay).
3. `step(action)` cho robot chạy trong `dt` giây (mặc định 0,1 s), kiểm tra collision, quét lại
   LiDAR và trả về observation mới.
4. Lặp lại tới khi robot tới goal (`terminated`) hoặc hết thời gian (`truncated`).

Ba environment:

| | `NavSim/Nav-v0` | `NavSim/GridNav-v0` | `navsim.multiagent.nav_v0` |
|---|---|---|---|
| Không gian | map liên tục, đơn vị mét | grid ô vuông, mỗi ô bằng kích thước một robot | như `Nav-v0`, nhiều robot |
| Dataset | loại `svg` | loại `grid` | loại `svg` |
| Action | 2 số thực: (tiến, quay) | 4 hướng: lên, xuống, trái, phải | như `Nav-v0`, mỗi robot một action |
| Khi nào dùng | cần mô phỏng sát robot thật (kích thước, tốc độ) | cần chạy nhanh, thuật toán rời rạc, curriculum theo độ khó | nghiên cứu multi-agent |
