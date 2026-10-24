# NavSim

NavSim là thư viện môi trường (environment) cho reinforcement learning, dùng để huấn luyện robot
điều hướng (navigation) trong nhà. Robot xuất phát từ một điểm (start), phải đi tới một điểm khác
(goal) trên bản đồ tầng nhà thật mà không va chạm (collision). Robot có footprint, giới hạn tốc độ và
LiDAR theo thông số của robot thật.

- Một robot: API [Gymnasium](https://gymnasium.farama.org/). Nhiều robot: API
  [PettingZoo](https://pettingzoo.farama.org/).
- Chỉ cần CPU. Khi chạy không cần Isaac Sim hay ROS.
- NavSim chỉ cung cấp environment. Không có reward và không có code training. `step()` luôn trả về
  reward `0.0`; bạn tự viết reward (xem tài liệu sử dụng).

**Tài liệu của dự án**

| Tài liệu | Nội dung | Dành cho |
|---|---|---|
| README này | NavSim là gì, dữ liệu bên trong có gì, LiDAR, bản đồ, va chạm hoạt động thế nào | người muốn hiểu cơ chế |
| [docs/README.md](docs/README.md) | mục lục tài liệu: hướng dẫn, trang từng môi trường, tra cứu hàm và lệnh, cơ chế bên trong | người đang viết code, cần tra cứu |

English: [README.md](README.md)

## Mục lục

1. [Cài đặt và chạy thử](#1-cài-đặt-và-chạy-thử)
2. [Tổng quan](#2-tổng-quan)
3. [Cấu trúc một dataset](#3-cấu-trúc-một-dataset)
4. [File `.npz`: bản đồ dạng grid](#4-file-npz-bản-đồ-dạng-grid)
5. [File `.svg`: bản đồ liên tục](#5-file-svg-bản-đồ-liên-tục)
6. [LiDAR](#6-lidar)
7. [Một lần `step()`](#7-một-lần-step)
8. [Shortest path, progress và time limit](#8-shortest-path-progress-và-time-limit)
9. [Robot và task](#9-robot-và-task)
10. [Cấu trúc code](#10-cấu-trúc-code)

---

## 1. Cài đặt và chạy thử

NavSim được phát hành dưới dạng một file wheel: `navsim-0.1.0-py3-none-any.whl` (Python ≥ 3.10).

```bash
pip install "navsim-0.1.0-py3-none-any.whl[marl,fast]"
```

Các extra (cài thêm, tùy chọn): `hub` (tải dataset đầy đủ), `marl` (PettingZoo, nhiều robot),
`fast` (numba, nhanh hơn khoảng 3 lần, kết quả giống hệt), `plot` (matplotlib). Cách cài trên Colab
và Kaggle: xem [docs/guide/getting-started.md](docs/guide/getting-started.md).

Chỉ cài từ file wheel. Lệnh `pip install navsim` cài một package khác trên PyPI trùng tên.

Chạy thử:

```python
import gymnasium as gym
import navsim

env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
obs, info = env.reset(seed=0)
print(info)                      # map_id, task_id, geodesic_m, time_limit
print(obs["lidar"].shape)        # (72,): 72 khoảng cách LiDAR
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
navsim.plot_episode(env, save="episode.png")   # vẽ bản đồ, robot, goal và các tia LiDAR
```

Hai dataset demo (`demo-svg`, `demo-grid`) nằm sẵn trong package, không cần tải.

---

## 2. Tổng quan

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

---

## 3. Cấu trúc một dataset

Một dataset là một thư mục. Ví dụ `demo-svg` và `demo-grid`:

```text
demo-svg/                                  demo-grid/
├── manifest.json                          ├── manifest.json
├── maps/                                  ├── grids/
│   └── h36/                               │   ├── demo-S001_s0.npz    (dữ liệu nhị phân)
│       ├── demo-S001_s0.svg               │   ├── demo-S001_s0.json   (metadata của grid)
│       └── ...                            │   └── ...
├── robots/                                ├── robots/
│   ├── turtlebot4.json                    │   └── jetauto_pro.json
│   └── clearpath_jackal.json              └── tasks/
└── tasks/                                     └── jetauto_pro/
    ├── turtlebot4/                                ├── demo-S001_s0.json
    │   ├── demo-S001_s0.json                      └── ...
    │   └── ...
    └── clearpath_jackal/ ...
```

| Thành phần | Chứa gì |
|---|---|
| `manifest.json` | tên dataset, loại (`svg` hoặc `grid`), danh sách robot, cách chia train/val/test, sha256 của từng file (dùng để kiểm tra file không bị hỏng) |
| `maps/<height_class>/*.svg` | map liên tục (dataset `svg`), xem [mục 5](#5-file-svg-bản-đồ-liên-tục) |
| `grids/*.npz` và `*.json` | map dạng grid (dataset `grid`), xem [mục 4](#4-file-npz-bản-đồ-dạng-grid) |
| `robots/*.json` | bản sao mô hình robot tại thời điểm build dataset, xem [mục 9](#9-robot-và-task) |
| `tasks/<robot>/*.json` | các cặp start/goal cho từng map và từng robot, xem [mục 9](#9-robot-và-task) |

**Map id.** `demo-S001_s0` gồm: prefix của dataset (`demo`), tòa nhà `S001`, tầng `s0` (storey 0,
tầng trệt). Dataset đầy đủ dùng prefix khác, ví dụ `isb-S001_s0`.

**Chia train/val/test theo tòa nhà.** Mỗi tòa nhà thuộc đúng một split. Map trong split test là
những tòa nhà không có trong training, nên kết quả test đo khả năng generalization sang tòa nhà mới.

**Height class (`h36`, `h63`, `h145`).** Một vật chỉ là vật cản với robot nếu nó nằm trong khoảng
độ cao từ sàn tới đỉnh robot. Mặt bàn ở độ cao 0,75 m nằm cao hơn robot 0,35 m (TurtleBot 4) nên
robot đi qua được, chỉ các chân bàn là vật cản. Với robot cao 1,45 m thì mặt bàn cũng là vật cản. Vì
vậy mỗi height class có một bộ map riêng: `h36` cắt vật cản ở 0,36 m, `h63` ở 0,626 m, `h145` ở 1,45
m. Các robot cùng height class dùng chung map.

---

## 4. File `.npz`: bản đồ dạng grid

### 4.1. `.npz` là gì

`.npz` là định dạng của NumPy: một file zip chứa nhiều array có tên. Đây là file nhị phân, không
mở được bằng text editor. Đọc bằng Python:

```python
import numpy as np

d = np.load("demo-S040_s0.npz")
print(d.files)              # ['grid', 'state', 'main']
for k in d.files:
    print(k, d[k].dtype, d[k].shape)
# grid   uint8  (56, 56)
# state  uint8  (56, 56)
# main   bool   (56, 56)
```

Map `demo-S040_s0` là grid 56 × 56 ô. Cả ba array có cùng shape, và phần tử `[r, c]` của mỗi array
mô tả cùng một ô (hàng `r`, cột `c`) của tầng nhà. Với robot `jetauto_pro`, một ô rộng 0,325 m
(`cell_m` trong file `.json` đi kèm), xấp xỉ kích thước một robot.

### 4.2. Ba array: `grid`, `state`, `main`

| Array | Kiểu | Giá trị | Ý nghĩa |
|---|---|---|---|
| `grid` | `uint8` | `0` = free (đi được), `1` = blocked | Map mà environment dùng. LiDAR dừng ở ô `1`, robot không di chuyển vào ô `1`. |
| `state` | `uint8` | `0` = sàn trống, `1` = vật cản trong tòa nhà (tường, đồ đạc), `2` = bên ngoài tòa nhà | Phân biệt tường với vùng ngoài tòa nhà. `grid` bằng `state` khi gộp giá trị `1` và `2` thành blocked. |
| `main` | `bool` | `True` = ô thuộc main component | Main component là vùng free lớn nhất mà mọi ô trong đó đều đi được tới nhau (largest connected component). |

Kiểm tra trên dữ liệu thật: `(state != 0) == (grid == 1)` đúng ở mọi ô.

**Vai trò của `main`.** Một tầng nhà có thể có các vùng free bị cô lập, ví dụ một góc sàn bị đồ đạc
bao kín. Nếu goal nằm ở vùng đó, robot không thể tới được. Vì vậy start và goal của mọi task đều
được chọn trong `main`. File `.json` đi kèm có `free_cells` (số ô free), `main_cells` (số ô thuộc
`main`) và `main_frac` (tỉ lệ giữa hai số này). Với `demo-S040_s0`: 2129 ô free, 2129 ô thuộc
`main`, tức toàn bộ vùng free liên thông.

### 4.3. Ví dụ: array `state` của `demo-S040_s0`

`.` = sàn trống, `#` = vật cản, `o` = bên ngoài tòa nhà (20 hàng đầu):

```text
ooooooooo######################################ooooooooo
ooooooooo#..###..............................##ooooooooo
ooooooooo#.####.......#####..................##ooooooooo
ooooooooo#.####.......#####..................##ooooooooo
ooooooooo#.##.###.....#####..................##ooooooooo
ooooooooo#....###.....#####..................##ooooooooo
ooooooooo#....###.....#####.............##...##ooooooooo
ooooooooo#....###......................####..##ooooooooo
ooooooooo#....###......................####..##ooooooooo
##########.............................###...###########
#......................................................#
#.................................................###..#
#......................................##.........###..#
#.............##.....#####.............###.............#
#.............##.....#####........##...###.............#
####..........##.....#####.......###...###.............#
#####............................###...................#
#####............................###...................#
####...................................................#
####.##................................................#
```

Các khối `#` ở giữa phòng là đồ đạc. Grid là hình chữ nhật nên có cả phần `o` nằm ngoài tòa nhà.

### 4.4. File `.json` đi kèm

Mỗi `.npz` có một `.json` cùng tên chứa metadata:

| Khóa | Ý nghĩa |
|---|---|
| `map_id`, `scene`, `robot` | tên map, tòa nhà, robot mà grid này được tính cho |
| `shape` | `[56, 56]`: số hàng, số cột |
| `cell_m` | 0,325: kích thước một ô (m) |
| `res`, `H`, `W` | độ phân giải ảnh gốc: 0,025 m/pixel, 736 × 736 pixel |
| `k` | 13: gộp 13 × 13 pixel thành một ô (13 × 0,025 = 0,325 m) |
| `margin`, `floor_close`, `max_hole_m2`, … | tham số lúc build dataset |
| `free_cells`, `main_cells`, `main_frac`, `main_m2` | số ô free, số ô `main`, tỉ lệ, diện tích `main` (m²) |
| `flags` | cảnh báo chất lượng lúc build (rỗng = không có) |
| `source` | cảnh Isaac Sim mà map được tạo từ đó |

### 4.5. Đặc điểm của grid

`GridNav-v0` là bài toán rời rạc: một ô tương ứng một robot, mỗi step robot di chuyển sang một ô kề
(lên, xuống, trái, phải). Environment này nhanh và dễ phân tích (shortest path tính bằng BFS), nhưng
không biểu diễn chính xác hình dạng robot và tường. Cần mô phỏng chính xác hơn thì dùng `Nav-v0` với
map SVG.

---

## 5. File `.svg`: bản đồ liên tục

### 5.1. Vì sao dùng SVG

SVG là file text (XML) mô tả hình bằng các đường thẳng thay vì pixel. Nhờ vậy: (1) mở bằng trình
duyệt là xem được map; (2) tọa độ là mét, độ chính xác 0,1 mm; (3) file nhỏ (khoảng 16 kB cho một
tầng).

### 5.2. Cấu trúc một file map

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.1 -18.1 18.2 18.2" ...>
  <style>.free{fill:#ffffff;stroke:#000} ...</style>

  <metadata id="hm3denv"><![CDATA[ { "map_id": "demo-S001_s0", "height_class": "h36", ... } ]]></metadata>

  <g id="map" transform="scale(1,-1)">           <!-- lật trục y, xem bảng dưới -->
    <g id="viz"> ... các layer chỉ để hiển thị ... </g>
    <path id="free" fill-rule="evenodd"
          d="M 3.1000,15.0000 L 3.1000,17.9000 L 6.0000,17.9000 ... Z
             M 15.1884,11.7539 L 15.2707,11.3773 ... Z
             M 17.0468,12.4385 ... Z"/>
  </g>
</svg>
```

| Phần | Vai trò |
|---|---|
| `<metadata id="hm3denv">` | Khối JSON nằm trong SVG: `map_id`, `scene`, `height_class`, `z_band` (khoảng độ cao dùng để cắt vật cản), `frame` (hệ trục tọa độ, đơn vị mét), `vectorize` (thống kê: diện tích sàn, chất lượng vectorize), `provenance` (công cụ và nguồn tạo file) |
| `<path id="free">` | Phần duy nhất mà environment dùng: vùng robot được phép đứng |
| Các `<path>` khác (trong `<g id="viz">`) | Chỉ để hiển thị, bị bỏ qua khi mô phỏng |
| `transform="scale(1,-1)"` | Trục y của SVG hướng xuống, còn tọa độ map (chuẩn ROS) có y hướng lên. Dòng này lật trục y để trình duyệt hiển thị đúng chiều. Trong thuộc tính `d`, tọa độ vẫn là mét với y hướng lên. |

### 5.3. Cách đọc thuộc tính `d`

Thuộc tính `d` của `<path id="free">` là một chuỗi lệnh:

```text
M 3.1,15.0   L 3.1,17.9   L 6.0,17.9   L 6.0,15.0   Z
│            │                                        └─ Z: đóng path, tạo thành polygon kín
│            └─ L: đoạn thẳng tới điểm này
└─ M: bắt đầu một polygon mới tại điểm này
```

Mỗi đoạn từ `M` tới `Z` là một ring (polygon kín). Map `demo-S001_s0` có 33 ring. Có hai loại:

- ring ngoài: biên của các phòng (vùng robot đi được);
- ring trong (hole): biên của đồ đạc, cột nằm giữa sàn.

NavSim phân biệt hai loại bằng even-odd rule (`fill-rule="evenodd"`): một điểm nằm trong số lẻ ring
thì thuộc vùng free; nằm trong số chẵn ring thì không.

```text
   ┌────────────────────────┐   điểm A: nằm trong 1 ring (phòng)             → free
   │  A                     │   điểm B: nằm trong 2 ring (phòng + đồ đạc)    → blocked
   │        ┌──────┐        │   điểm C: nằm ngoài mọi ring                   → blocked
   │        │  B   │        │
   │        └──────┘        │   Không cần đánh dấu ring nào là hole:
   └────────────────────────┘   chỉ cần đếm số ring chứa điểm.
                  C
```

### 5.4. Các bước xử lý SVG (`svgmap.py`)

Khi load một map, NavSim làm lần lượt:

1. Đọc `<metadata>` (JSON) và `<path id="free">`.
2. Tách ring: `path_to_rings` chuyển chuỗi lệnh thành danh sách polygon. Chỉ hỗ trợ lệnh tuyệt đối
   `M`, `L`, `Z`. Gặp lệnh khác (curve, lệnh tương đối `l`, `h`, `v`, …) thì báo `ValueError` thay vì
   đọc sai. Nếu tự vẽ map trong Inkscape, đặt *Preferences → Input/Output → SVG output → Path data →
   Absolute* và chuyển các curve thành đoạn thẳng.
3. Lấy các cạnh (segment): mỗi cạnh của mỗi polygon là một đoạn `(x1, y1, x2, y2)`. LiDAR dùng các
   segment này để tính khoảng cách.
4. Rasterize: vẽ các ring thành ảnh nhị phân, 2 cm mỗi pixel, áp even-odd rule bằng cách XOR từng
   ring. Ảnh này dùng để kiểm tra nhanh một điểm `(x, y)` có free không.
5. Distance transform: tính khoảng cách từ mỗi pixel tới cạnh gần nhất. Nếu robot còn cách tường xa
   hơn bán kính của nó thì không cần kiểm tra collision chi tiết.
6. Tạo planner (grid 4 cm để tính shortest path, xem [mục 8](#8-shortest-path-progress-và-time-limit)).

Các bước 3 đến 6 tốn thời gian, nên kết quả được lưu vào disk cache (Windows:
`%LOCALAPPDATA%\navsim\cache`; Linux: `~/.cache/navsim`; đổi bằng biến `NAVSIM_CACHE`). Các lần load
sau nhanh hơn nhiều.

---

## 6. LiDAR

### 6.1. Nguyên lý

LiDAR 2D (ví dụ SLAMTEC RPLIDAR A1 trên TurtleBot 4) phát tia laser theo từng góc, đo thời gian tia
phản xạ để tính khoảng cách tới vật cản gần nhất theo hướng đó. Kết quả là một danh sách khoảng
cách, mỗi giá trị ứng với một hướng.

NavSim mô phỏng phần hình học bằng ray casting: từ tâm robot, với mỗi hướng, tìm giao điểm gần nhất
giữa tia (ray) và các cạnh tường. Không mô phỏng vật lý của ánh sáng.

```text
                 beam 0 (hướng mũi robot)
                  ↑   ╲  ╱
        ──────────┼────╲╱───────  tường
                  │     ·
        ╲         │         ╱     Mỗi beam trả về một khoảng cách: tới điểm chạm tường đầu tiên.
          ╲   ╲   │   ╱   ╱       Nếu không chạm gì trong tầm đo: trả về range_max.
            ╲  ╲  │  ╱  ╱
              ╲ ╲ │ ╱ ╱
          ◀─────── ● ───────▶     ● = robot; các beam chia đều 360°, beam 0 hướng về phía trước
```

### 6.2. LiDAR của `Nav-v0` (map liên tục)

Thông số lấy từ file robot, ví dụ TurtleBot 4:

| Thông số | Giá trị | Ý nghĩa |
|---|---|---|
| `range_min` | 0,15 m | khoảng cách nhỏ nhất đo được; giá trị nhỏ hơn bị clip lên 0,15 |
| `range_max` | 12 m | khoảng cách lớn nhất đo được; không có vật cản thì trả về 12 |
| `fov` | 2π (360°) | field of view |
| `beams` | 360 | số beam của cảm biến thật |

Environment chỉ đưa cho policy `n_beams` beam (mặc định 72) để observation nhỏ hơn:

- LiDAR quét đủ 360°: beam thứ `k` có góc `k × 2π / n_beams` so với hướng mũi robot, tăng dần ngược
  chiều kim đồng hồ. Với 72 beam, hai beam liền nhau cách 5°. Beam 0 hướng về phía trước, beam 18
  hướng sang trái (90°), beam 36 hướng ra sau, beam 54 hướng sang phải.
- LiDAR có `fov` nhỏ hơn 360°: các beam chia đều từ `-fov/2` tới `+fov/2`.

Mỗi lần `step()`:

1. Với mỗi beam, tìm khoảng cách tới cạnh tường gần nhất (giải giao điểm giữa tia và đoạn thẳng).
2. Robot khác (khi nhiều robot) cũng chặn beam: footprint của chúng cũng là các đoạn thẳng.
3. Nếu `lidar_noise > 0`, cộng Gaussian noise (độ lệch chuẩn tính bằng mét).
4. Clip kết quả vào `[range_min, range_max]`.
5. Nếu `normalize_obs=True` (mặc định), chia cho `range_max` để đưa về `[0, 1]`.

Ví dụ thật: TurtleBot 4 ngay sau `reset(seed=0)` (`normalize_obs=False`, 18 beam đầu, đơn vị mét):

```text
[1.35 1.41 1.48 1.33 1.21 1.12 1.04 0.63 0.60 0.57 0.54 0.53 0.51 0.50 0.50 0.50 0.89 0.92]
```

Beam 0 (phía trước) đo được tường ở 1,35 m. Các beam 13 đến 15 (khoảng 65° đến 75°, phía trái) chỉ
đo được 0,50 m, nghĩa là robot đang ở sát tường bên trái.

**Tốc độ tính.** Một map có hàng nghìn cạnh (`demo-S001_s0` có 1507 đỉnh). Tính 72 beam với từng
ấy cạnh sẽ chậm, nên có hai cách giảm chi phí:

- Với mỗi cạnh, chỉ kiểm tra các beam nằm trong góc nhìn của cạnh đó (tính từ vị trí robot).
- Khi cài numba (`navsim[fast]`), các cạnh được xếp vào một grid ô vuông; mỗi beam chỉ duyệt các ô
  mà nó đi qua và dừng khi đã tìm thấy cạnh gần hơn ô kế tiếp. Kết quả giống hệt bản numpy (cùng
  thứ tự phép tính, không dùng fastmath), chỉ nhanh hơn khoảng 3 lần.

### 6.3. LiDAR của `GridNav-v0` (grid)

Trên grid, khoảng cách tính bằng số ô:

- Mặc định 22 ray (`lidar_rays`), góc chia đều `2π / 22`, tầm tối đa 10 ô (`lidar_range`).
- Mỗi ray đi từng đoạn 0,1 ô từ vị trí robot và dừng khi gặp ô `grid == 1` hoặc ra ngoài grid.
  Khoảng cách tại điểm dừng là kết quả; nếu không gặp gì thì trả về `lidar_range`.

Mặc định, ray bắt đầu từ góc trên-trái của ô và tọa độ được làm tròn về phía 0 (`int()`), nên kết
quả không đối xứng: trong một phòng vuông đối xứng, ray hướng xuống/phải đọc 10,0 còn ray hướng
lên/trái đọc 9,1. Hành vi này được giữ nguyên từ code curriculum learning gốc để kết quả tái lập
được. Đặt `lidar_centered=True` để ray bắt đầu từ tâm ô (đối xứng).

Observation của `GridNav-v0` mặc định chỉ có LiDAR, không có goal (robot phải tự tìm goal). Đặt
`goal_obs=True` để thêm `goal = (số hàng, số cột)` từ robot tới goal vào observation.

---

## 7. Một lần `step()`

### 7.1. Từ action tới chuyển động

Action của `Nav-v0` nằm trong `[-1, 1]` và là tỉ lệ so với giới hạn tốc độ của robot:

```text
action = (a_tiến, a_quay)               robot differential drive (TurtleBot, Jackal, ...)
action = (a_tiến, a_ngang, a_quay)      robot omnidirectional (JetAuto Pro, bánh mecanum)

v = a_tiến × v_max          (nếu a_tiến < 0 thì v = a_tiến × |v_min|)
w = a_quay × w_max
```

TurtleBot 4 có `v_max = 0.31 m/s` và `w_max = 1.9 rad/s`. Action `(1, 0)` là chạy thẳng với tốc độ
tối đa: trong `dt = 0.1 s` robot đi `0.31 × 0.1 = 0.031 m`. Giá trị ngoài `[-1, 1]` bị clip.

### 7.2. Sub-step

Một step 0,1 s của robot nhanh có thể dài 20 cm. Nếu robot di chuyển hết quãng đường trong một lần,
nó có thể đi xuyên qua một bức tường mỏng mà không bị phát hiện. Vì vậy mỗi step được chia thành
các sub-step, mỗi sub-step đi tối đa 2 cm và quay tối đa 2°. Sau mỗi sub-step:

1. Tính pose mới theo công thức chính xác cho chuyển động với velocity không đổi (differential
   drive đi theo cung tròn).
2. Kiểm tra collision với map: đặt polygon footprint của robot tại pose mới và xem có cắt cạnh tường
   nào không (`pose_valid`). Nếu có, sub-step bị hủy, robot giữ pose cũ, step được tính là một
   collision (`collided = True`) và velocity về 0.
3. Nhiều robot: kiểm tra footprint của các robot có chồng lên nhau không (kiểm tra bằng hình tròn
   bao ngoài trước, sau đó mới kiểm tra polygon).
4. Kiểm tra goal: nếu khoảng cách từ tâm robot tới goal ≤ `success_radius` (mặc định 0,2 m) thì
   dừng và đặt `success = True`. Kiểm tra này chạy ở mọi sub-step, không chỉ cuối step, vì robot
   nhanh (Jackal đi 0,2 m mỗi step, bằng kích thước vùng goal) có thể đi qua goal trong một step
   mà không được ghi nhận.

### 7.3. Giá trị trả về của `step()`

`step()` trả về `(obs, reward, terminated, truncated, info)`:

| Giá trị | Nội dung |
|---|---|
| `obs["lidar"]` | `n_beams` khoảng cách ([mục 6](#6-lidar)) |
| `obs["goal"]` | 3 số: khoảng cách thẳng tới goal, sin và cos của góc tới goal tính từ hướng mũi robot |
| `obs["velocity"]` | velocity hiện tại, chia cho giới hạn tốc độ (khi `normalize_obs=True`) |
| `reward` | luôn `0.0`; dùng `navsim.wrappers.Reward` để tự định nghĩa |
| `terminated` | `True` khi tới goal (hoặc khi collision nếu `on_collision="terminate"`) |
| `truncated` | `True` khi hết `time_limit` |
| `info` | collision, thời gian, pose, quãng đường đã đi, `progress_m`, …; ở step cuối có thêm `success`, `spl`, `termination` |

**Lý do dùng (khoảng cách, sin, cos) thay vì một góc.** Góc không liên tục tại ±π (−179° và +179°
gần nhau nhưng khác nhau 358 khi biểu diễn bằng số). Cặp (sin, cos) liên tục, thuận lợi hơn cho
neural network. Trong ví dụ trên, `goal = [5.68, -0.25, -0.97]` có `cos = -0.97`: goal nằm gần như
ngay phía sau robot.

### 7.4. Reward

Environment không định nghĩa reward vì reward thường là phần cần thử nghiệm (sparse hay dense, có
phạt collision hay không). NavSim cung cấp các giá trị để tính reward: `RobotState` và `info`, xem
[docs/concepts/robot_state.md](docs/concepts/robot_state.md). Hai giá trị hay dùng nhất là `progress_m`
(shortest path tới goal ngắn đi bao nhiêu mét trong step này) và `state.reached_goal`.

---

## 8. Shortest path, progress và time limit

Một số giá trị cần khoảng cách còn lại tới goal theo đường đi thực, không phải đường thẳng xuyên
tường. Đó là khoảng cách geodesic, do planner tính.

**Planner của `Nav-v0`:**

1. Tạo grid 4 cm. Một ô là free nếu hình tròn bán kính (bán kính robot + clearance 5 cm) đặt ở đó
   không chạm tường, tức robot vừa ở đó với mọi hướng mũi. Do đó đường tìm được luôn đi được. Nhược
   điểm: có thể bỏ sót lối hẹp mà robot dài chỉ đi được khi đi dọc theo.
2. Mỗi ô nối với 16 ô lân cận (4 thẳng, 4 chéo, 8 kiểu knight move) mà không cắt góc tường. Đường
   tìm được dài hơn đường thẳng lý tưởng tối đa khoảng 2,8%.
3. Chạy Dijkstra một lần từ goal để có khoảng cách tới goal của mọi ô. Sau đó khoảng cách của một
   điểm bất kỳ chỉ cần tra bảng. Bảng được cache trong RAM.

Các đại lượng suy ra:

| Đại lượng | Cách tính | Dùng để |
|---|---|---|
| `goal_geodesic_m` | khoảng cách geodesic từ vị trí hiện tại tới goal | biết còn bao xa |
| `progress_m` | `geodesic(trước) − geodesic(sau)` | reward theo tiến độ (âm nếu robot đi xa goal hơn) |
| `spl` | `success × shortest / max(taken, shortest)` | đo độ hiệu quả của đường đi (SPL: Success weighted by Path Length, Anderson 2018); bằng 1 nếu đi đúng shortest path |
| `time_limit` | `time_factor × (shortest / v_max + tổng_góc_cua / w_max)` | quá thời gian này thì `truncated` |

`time_limit` mặc định là 3 lần (`time_factor = 3.0`) thời gian tối thiểu: chạy hết tốc độ trên
shortest path, cộng thời gian quay hết tốc độ qua các góc cua. Ví dụ khi chạy thử: shortest path
11,34 m, TurtleBot 4 → `time_limit` = 123,6 s.

**Trên grid** (`GridNav-v0`): BFS từ ô goal cho số nước đi ít nhất `d_bfs` tới mọi ô; `time_limit`
là `budget` step (mặc định 200).

---

## 9. Robot và task

### 9.1. Robot

Mỗi robot là một file JSON trong `robots/presets/`. Có 9 robot: `agilex_limo`, `clearpath_jackal`,
`jetauto_pro`, `kobuki`, `pal_tiago`, `turtlebot3_burger`, `turtlebot3_waffle_pi`, `turtlebot4`,
`turtlebot4_lite`. Ví dụ JetAuto Pro (rút gọn):

```json
{
  "id": "jetauto_pro",
  "footprint": {"type": "rectangle", "length": 0.324, "width": 0.26},
  "height": 0.626, "height_class": "h63",
  "drive": "omnidirectional",
  "limits": {"v_max": 0.6, "v_min": -0.6, "vy_max": 0.6, "w_max": 1.0},
  "lidar":  {"model": "SLAMTEC RPLIDAR A1M8", "range_min": 0.15, "range_max": 12.0, "fov": 6.2832, "beams": 360},
  "sources": { "limits.v_max": {"kind": "official", "url": "..."}, "limits.w_max": {"kind": "estimated", "method": "..."} }
}
```

| Khóa | Ý nghĩa |
|---|---|
| `footprint` | Hình chiếu thân robot nhìn từ trên xuống, là một polygon trong hệ tọa độ robot (x hướng tới trước, y sang trái, gốc ở tâm quay). Hình tròn được thay bằng polygon 16 cạnh bao ngoài. Dùng để kiểm tra collision. |
| `height`, `height_class` | chiều cao và height class tương ứng ([mục 3](#3-cấu-trúc-một-dataset)) |
| `drive` | `differential` (không đi ngang được) hoặc `omnidirectional` (bánh mecanum, đi ngang được; action có 3 chiều) |
| `limits` | tốc độ tối đa: tiến `v_max`, lùi `v_min` (≤ 0), ngang `vy_max` (m/s), quay `w_max` (rad/s) |
| `lidar` | thông số LiDAR ([mục 6](#6-lidar)) |
| `sources` | nguồn của từng con số: `official` (kèm URL của hãng) hoặc `estimated` (kèm phương pháp ước lượng) |

Để thêm robot riêng: viết một file JSON theo cùng cấu trúc rồi gọi
`navsim.load_robot("duong/dan/robot.json")`, hoặc đặt file trong thư mục nằm trong `$NAVSIM_ROBOTS`.

### 9.2. Task

Task là một cặp (start, goal) kèm label. Ví dụ `tasks/turtlebot4/demo-S001_s0.json`:

```json
{ "map_id": "demo-S001_s0", "robot": "turtlebot4", "split": "train",
  "height_class": "h36", "success_radius": 0.2, "clearance": 0.05,
  "tasks": [ { "id": 0,
               "start": {"x": 2.48, "y": 12.36, "theta": -2.1753},
               "goal":  {"x": 4.84, "y": 10.72},
               "geodesic_m": 3.1963, "euclidean_m": 2.8739,
               "labels": {"gdr": 1.1122} }, ... ] }
```

- `start` có `theta` (hướng mũi robot lúc xuất phát, rad); `goal` chỉ có vị trí.
- `geodesic_m` là độ dài shortest path, `euclidean_m` là khoảng cách đường thẳng. Tỉ số hai giá trị
  cho biết đường đi vòng nhiều hay ít (task 0: 3,20 m so với 2,87 m).
- `success_radius` (0,2 m) là bán kính coi là tới goal; `clearance` là khoảng an toàn của planner.
- Task được tính riêng cho từng robot (robot lớn có ít vị trí hợp lệ hơn) nhưng dùng chung map.
- Dataset `svg` có label `gdr`. Dataset `grid` có thêm label độ khó:

| Label | Ý nghĩa |
|---|---|
| `d_bfs` | số nước đi ít nhất từ start tới goal |
| `p0`, `sa_uniform`, `sa_persist` | xác suất do công cụ build dataset tính trước, đo mức độ khó để tới goal bằng cách đi ngẫu nhiên (`p0` và `sa_uniform` ứng với random đều, `sa_persist` ứng với random có quán tính). Giá trị càng nhỏ càng khó; giá trị cỡ 1e-6 nghĩa là gần như không bao giờ tới goal bằng random. |
| `level` | `easy`, `medium`, `hard` hoặc `starved` |
| `starved` | `true` khi task thuộc loại reward rất hiếm: random exploration gần như không bao giờ tới goal, nên khó học nếu reward sparse |

Dùng các label này để làm curriculum: train các task `easy` trước rồi tăng dần độ khó, ví dụ
`env.unwrapped.set_task_filter(level=["easy"])`.

---

## 10. Cấu trúc code

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

Cách dùng (tham số, hàm, lệnh): [docs/README.md](docs/README.md).

Giấy phép MIT. Các dataset build từ HM3D tuân theo điều khoản sử dụng HM3D của Matterport.
