# Cấu trúc code

```text
src/navsim/
  __init__.py       đăng ký NavSim/Nav-v0 và NavSim/GridNav-v0, các hàm công khai
  envs/
    nav.py          environment liên tục Nav-v0
    grid.py         environment grid GridNav-v0, lidar_scan(), bfs()
    episodes.py     chọn map và task ở mỗi reset, task filter
  multiagent/
    nav_v0.py       nhiều robot (PettingZoo), ParallelReward
  data/
    dataset.py      tìm, đọc, kiểm tra dataset (class Dataset)
    download.py     tải dataset từ Hugging Face (navsim.setup)
    platform.py     nhận biết Colab/Kaggle, lấy token
    demo/           hai dataset demo đi kèm package
  robots/           class Robot và presets/*.json
  wrappers.py       Reward, DiscreteActions, ParallelReward
  evaluation.py     navsim.evaluate()
  visualize.py      plot_episode, render_frame, save_gif, record_video
  cli.py            lệnh `navsim`
  _sim/             simulator bên trong (không phải API công khai)
    svgmap.py       đọc SVG → polygon → ảnh nhị phân
    geometry.py     collision, ray casting (numpy), class Collider
    kernels.py      bản numba của các hàm hình học chậm nhất
    planning.py     grid 4 cm, Dijkstra, shortest path
    world.py        World: chuyển động, LiDAR, collision, RobotState
    cache.py        disk cache của map
tests/              pytest
scripts/            smoke_wheel.py (build và thử wheel), make_drive_folder.py
```
