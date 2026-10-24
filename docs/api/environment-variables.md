# Biến môi trường và cache

| Biến | Tác dụng |
|---|---|
| `HF_TOKEN` | token Hugging Face (cũng đọc secret `HF_TOKEN` của Colab/Kaggle) |
| `NAVSIM_HOME` | thư mục dữ liệu, mặc định `~/.navsim`; dataset tải về nằm ở `<nó>/datasets` |
| `NAVSIM_DATASETS` | thư mục tìm dataset thêm (ngăn cách `;` trên Windows, `:` trên Linux/macOS) |
| `NAVSIM_ROBOTS` | thư mục tìm file JSON robot thêm |
| `NAVSIM_CACHE` | thư mục cache bản đồ; `off`/`0`/`false`/`none`/rỗng để tắt. Mặc định: Windows `%LOCALAPPDATA%\navsim\cache`; Linux `$XDG_CACHE_HOME/navsim` hoặc `~/.cache/navsim` |
| `NAVSIM_ACCEL` | `0` tắt kernel numba (cài `navsim[fast]` để bật; nhanh ~3×, **kết quả y hệt**) |

**Cache đĩa** (`_sim/cache.py`): lưu đúng thứ code tính ra (đường viền bản đồ, raster, ảnh khoảng cách tới
cạnh, task file, planner) để nạp bản đồ chỉ mất vài ms thay vì ~100 ms. Khoá cache băm sha256 của file bản
đồ/task (lấy từ manifest, không đọc lại file) cùng mọi tham số ảnh hưởng kết quả, nên dữ liệu đổi thì tự ra
file mới; file cũ chỉ bị xoá bởi `navsim cache --clear`. **Kết quả giống hệt khi bật hoặc tắt cache.**
