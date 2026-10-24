# Định dạng dataset và bản đồ

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
| `robots/*.json` | bản sao mô hình robot tại thời điểm build dataset, xem [mục 9](./robots-and-tasks.md) |
| `tasks/<robot>/*.json` | các cặp start/goal cho từng map và từng robot, xem [mục 9](./robots-and-tasks.md) |

**Map id.** `demo-S001_s0` gồm: prefix của dataset (`demo`), tòa nhà `S001`, tầng `s0` (storey 0,
tầng trệt). Dataset đầy đủ dùng prefix khác, ví dụ `isb-S001_s0`.

**Chia train/val/test theo tòa nhà.** Mỗi tòa nhà thuộc đúng một split. Map trong split test là
những tòa nhà không có trong training, nên kết quả test đo khả năng generalization sang tòa nhà mới.

**Height class (`h36`, `h63`, `h145`).** Một vật chỉ là vật cản với robot nếu nó nằm trong khoảng
độ cao từ sàn tới đỉnh robot. Mặt bàn ở độ cao 0,75 m nằm cao hơn robot 0,35 m (TurtleBot 4) nên
robot đi qua được, chỉ các chân bàn là vật cản. Với robot cao 1,45 m thì mặt bàn cũng là vật cản. Vì
vậy mỗi height class có một bộ map riêng: `h36` cắt vật cản ở 0,36 m, `h63` ở 0,626 m, `h145` ở 1,45
m. Các robot cùng height class dùng chung map.


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
6. Tạo planner (grid 4 cm để tính shortest path, xem [mục 8](./step-and-planning.md#8-shortest-path-progress-và-time-limit)).

Các bước 3 đến 6 tốn thời gian, nên kết quả được lưu vào disk cache (Windows:
`%LOCALAPPDATA%\navsim\cache`; Linux: `~/.cache/navsim`; đổi bằng biến `NAVSIM_CACHE`). Các lần load
sau nhanh hơn nhiều.
