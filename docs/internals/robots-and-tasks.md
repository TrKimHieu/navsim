# Robot và task

## Robot

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
| `height`, `height_class` | chiều cao và height class tương ứng ([mục 3](./dataset-format.md#3-cấu-trúc-một-dataset)) |
| `drive` | `differential` (không đi ngang được) hoặc `omnidirectional` (bánh mecanum, đi ngang được; action có 3 chiều) |
| `limits` | tốc độ tối đa: tiến `v_max`, lùi `v_min` (≤ 0), ngang `vy_max` (m/s), quay `w_max` (rad/s) |
| `lidar` | thông số LiDAR ([mục 6](./lidar.md)) |
| `sources` | nguồn của từng con số: `official` (kèm URL của hãng) hoặc `estimated` (kèm phương pháp ước lượng) |

Để thêm robot riêng: viết một file JSON theo cùng cấu trúc rồi gọi
`navsim.load_robot("duong/dan/robot.json")`, hoặc đặt file trong thư mục nằm trong `$NAVSIM_ROBOTS`.

## Task

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
