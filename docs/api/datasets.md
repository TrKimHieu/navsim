# Dữ liệu

*(`data/dataset.py`, `data/download.py`, `data/platform.py`)*

## `navsim.setup(datasets="demo-svg", token=None, data_dir=None, force=False, repo_id=…, quiet=False)`

**Làm gì:** đảm bảo dataset có trên máy và trả về `Dataset`.

**Cơ chế, theo thứ tự:**
1. Nếu có `data_dir`: tạo thư mục, rồi thêm nó vào **đầu** `$NAVSIM_DATASETS` — để các tiến trình con
   (vector env) thừa hưởng biến môi trường và cũng tìm thấy dataset.
2. Với mỗi tên: nếu `force=False`, gọi `find_dataset` (xem dưới). Tìm thấy thì dùng luôn, **không tải**.
3. Không thấy: nếu `name` là đường dẫn tuyệt đối hoặc chứa `/`/`\` → `FileNotFoundError` (đường dẫn
   sai, không thử tải). Còn lại → lấy token (`hf_token`: tham số → `$HF_TOKEN` → secret Colab/Kaggle),
   gọi `download`.
4. `download` dùng `huggingface_hub.snapshot_download` chỉ với mẫu `<tên>/*`, rồi **kiểm tra sha256 từng
   file với `manifest.json`** (`validate_dataset`); lệch thì `ValueError`.

**Trả về:** `Dataset`, hoặc `list[Dataset]` nếu bạn truyền list.

**Lỗi:** `ImportError` (thiếu `navsim[hub]`), `PermissionError` (chưa được cấp quyền repo gated, hoặc
token sai; thư mục rỗng vừa tạo bị xoá), `ConnectionError` (mất mạng; Kaggle cần bật Internet),
`FileNotFoundError` (tên không có trong repo — thông báo liệt kê các tên có sẵn).

```python
navsim.setup()                                  # demo kèm gói, không tải
navsim.setup("isb-svg-v1")                      # tải một lần, lần sau dùng lại
navsim.setup("isb-svg-v1", data_dir="/content/drive/MyDrive/navsim")   # giữ trên Drive
```

## `navsim.load_dataset(name_or_path)` / `navsim.list_datasets()`

- `load_dataset`: nếu tham số là thư mục có `manifest.json` thì dùng luôn; nếu không, tìm theo tên
  trong **thứ tự ưu tiên**: `$NAVSIM_DATASETS` → `~/.navsim/datasets` (hoặc `$NAVSIM_HOME/datasets`)
  → `/kaggle/input/*/<tên>` → demo trong gói. Không thấy → `FileNotFoundError`.
- `list_datasets`: duyệt cùng các thư mục đó và trả mọi thư mục có `manifest.json`.

## Đối tượng `Dataset`

Đọc `manifest.json` (schema 2.0). Các hàm không nạp bản đồ vào RAM; chỉ đọc file khi bạn hỏi.

| Thành phần | Chi tiết kỹ thuật |
|---|---|
| `env_type` | `"svg"` → dùng `Nav-v0`/`nav_v0`; `"grid"` → dùng `GridNav-v0`. Dùng nhầm môi trường báo `ValueError: dataset X is a … dataset: use …` |
| `maps(robot, split)` | id bản đồ mà robot đó có task; `split=None` là tất cả |
| `tasks(robot, map_id)` | list dict `{"id","start":{x,y,theta},"goal":{x,y},"geodesic_m","euclidean_m","labels"}`; grid thêm `"cell":[row,col]` |
| `task_file(robot, map_id)` | cả file task: `split`, `success_radius`, `clearance`, `tasks` |
| `height_class(robot)` | `h36`/`h63`/`h145`: bộ bản đồ cắt chướng ngại ở độ cao đó; các robot cùng lớp dùng chung bản đồ |
| `default_robot(robot=None)` | robot truyền vào nếu hợp lệ; nếu dataset chỉ có 1 robot thì dùng nó; còn lại `"turtlebot4"` |
| `svg_path`, `grid_arrays`, `grid_meta` | đường dẫn SVG; `{"grid","state","main"}` (NumPy, từ `.npz`); nội dung `.json` (`cell_m`, `shape`…) |
| `summary()` | số bản đồ và task theo robot và split |
| `validate()` | list lỗi (rỗng = hợp lệ); đối chiếu sha256 |

## Tìm và tải dataset



**Thứ tự tìm dataset theo tên:**

1. các thư mục trong `$NAVSIM_DATASETS` (ngăn cách bằng `:` trên Linux, `;` trên Windows);
2. `~/.navsim/datasets` (hoặc `$NAVSIM_HOME/datasets`), nơi `navsim.setup` tải về;
3. input của Kaggle: `/kaggle/input/<gì cũng được>/<tên>`;
4. dataset demo trong package: `demo-svg`, `demo-grid`.

**Một task** (dict):

```python
{"id": 0,
 "start": {"x": 2.48, "y": 12.36, "theta": -2.1753},   # grid: thêm "cell": [hàng, cột]
 "goal":  {"x": 4.84, "y": 10.72},
 "geodesic_m": 3.1963, "euclidean_m": 2.8739,
 "labels": {"gdr": 1.1122}}                            # grid: d_bfs, level, p0, sa_uniform, sa_persist, starved
```

Dataset demo có sẵn trong package: `demo-svg` (robot `turtlebot4`, `clearpath_jackal`) và
`demo-grid` (robot `jetauto_pro`), mỗi dataset 3 tòa nhà train / 2 val / 1 test.

**Dataset đầy đủ** tải từ Hugging Face: xin quyền ở
[trang dataset](https://huggingface.co/datasets/PhysicalSI/2D_Navigation_Sim), lưu một token *Read*
thành secret `HF_TOKEN` (Colab: biểu tượng chìa khóa, bật *Notebook access*; Kaggle: *Add-ons →
Secrets*, bật *Internet*; máy cá nhân: `export HF_TOKEN=...`), rồi `navsim.setup("isb-svg-v1")`.

**Định dạng thư mục dataset (schema 2.0):**

```text
manifest.json                      name, env ("svg"|"grid"), robots, splits, sha256 của từng file
tasks/<robot>/<map_id>.json        {"split", "success_radius", "tasks": [...]}
maps/<height_class>/<map_id>.svg   map liên tục (dataset svg)
grids/<map_id>.npz + .json         grid (dataset grid)
robots/<robot>.json                robot như lúc build
```
