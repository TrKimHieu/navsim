# Dòng lệnh `navsim`

*(`cli.py`; cũng chạy bằng `python -m navsim`)* Lỗi `FileNotFoundError, PermissionError, ConnectionError,
ValueError, KeyError` được in dạng `error: …` và thoát mã **1**; thành công mã 0.

| Lệnh | Cơ chế |
|---|---|
| `navsim setup [datasets…] [--token T] [--data-dir D] [--force]` | gọi `navsim.setup`; mặc định `demo-svg` |
| `navsim datasets` | `list_datasets()` rồi in bảng `name, env, robots, path` |
| `navsim robots` | nạp từng robot, in `id, drive, radius_m (r_circ), v_max, w_max, lidar_m, fov_deg, name` |
| `navsim info <dataset>` | in `Dataset.summary()` dạng JSON |
| `navsim check [--dataset D] [--robot R] [--agents N] [--seconds S]` | (1) chọn env theo `env_type`, chạy `gymnasium.utils.env_checker.check_env`; (2) in observation/action space; (3) chạy action ngẫu nhiên `S` giây (mặc định 2), in **step/s** (một tiến trình), reset khi hết episode; (4) nếu dataset svg và `N > 1` (mặc định 3), chạy `pettingzoo.test.parallel_api_test` 200 chu kỳ; thiếu PettingZoo thì in "skipped" |
| `navsim cache` | in thư mục cache và bảng `dataset, files, mb, path` |
| `navsim cache --clear [DATASET]` | xoá cache của một dataset, hoặc tất cả; in số thư mục đã xoá |
