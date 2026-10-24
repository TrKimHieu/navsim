# Hình ảnh và GIF

*(`visualize.py`)* Dùng được với cả ba môi trường, sau `reset` và ở mọi step. **Vẽ không đụng tới bộ sinh
số ngẫu nhiên** nên không làm đổi episode. Chưa `reset` → `ValueError: reset the environment before drawing it`.
Môi trường không phải của NavSim → `TypeError`.

| Hàm | Cơ chế |
|---|---|
| `plot_episode(env, ax=None, lidar=True, path=True, shortest_path=False, title=True, legend=True, figsize=(8,8), save=None, dpi=150)` | Vẽ bằng matplotlib: bản đồ (raster 0,04 m), tia LiDAR **không nhiễu** (chấm ở chỗ tia chạm vật), đường đã đi và điểm xuất phát, đích (hình tròn bán kính `success_radius`), thân robot và hướng. `shortest_path=True` thêm đường nét đứt (robot còn hoạt động). Nhiều robot: mỗi robot một màu. `save` tạo thư mục cha và ghi `.png/.pdf/.svg`. Trả `Axes`. Thiếu matplotlib → `ImportError` gợi ý `navsim[plot]` hoặc `style="fast"` |
| `render_frame(env, style="plot", size=480, lidar=True, shortest_path=False)` | Trả ảnh `(H,W,3)` uint8 cạnh lớn nhất ≤ `size`. `"plot"` vẽ bằng matplotlib trên canvas ngoài màn hình (~0,1 s/khung). `"fast"`: ảnh riêng của môi trường (`render_rgb` / `rgb_frame`), không cần matplotlib, nhanh hơn nhiều. `style` khác → `ValueError` |
| `save_gif(frames, path, fps=10)` | Pillow ghi GIF lặp vô hạn, `duration = 1000/fps` ms; tạo thư mục cha; list rỗng → `ValueError` |
| `record_video(env, policy=None, path="videos/episode.gif", episodes=1, seed=0, options=None, fps=None, max_steps=1000, every=1, size=480, style="plot", deterministic=None, show=True)` | Chạy policy (None = ngẫu nhiên), reset mỗi episode với `seed+e`, ghi khung mỗi `every` step (và luôn ghi khung cuối). `fps=None` → `render_fps` của môi trường; GIF dùng `fps/every`. Trong notebook (`show=True`) hiển thị ngay. Hỗ trợ nhiều robot (policy là dict hoặc một policy chung) |
| `env.render()` | với `render_mode="rgb_array"`: ảnh của chính môi trường |
