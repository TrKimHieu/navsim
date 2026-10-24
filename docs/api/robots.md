# Robot

*(`robots/__init__.py`, `robots/presets/*.json`)*

## `navsim.list_robots()` / `navsim.load_robot(id_or_json_path)`

`load_robot` tìm theo thứ tự: thư mục `extra` (bản sao robot trong dataset) → các thư mục trong
`$NAVSIM_ROBOTS` → preset trong gói; hoặc nhận thẳng đường dẫn `.json`. File được kiểm tra
(`validate`): thiếu khoá bắt buộc (`schema_version, id, name, footprint, height, height_class, drive,
limits, lidar, sources`) thì lỗi.

## Đối tượng `Robot`

| Thành phần | Ý nghĩa kỹ thuật |
|---|---|
| `footprint` | đa giác `(M,2)` ngược chiều kim đồng hồ, gốc ở tâm quay; hình tròn được thay bằng đa giác 16 cạnh **không nhỏ hơn** robot thật |
| `r_circ` | bán kính đường tròn nhỏ nhất quanh tâm chứa trọn thân. Dùng cho kiểm tra va chạm nhanh (nếu hai tâm cách nhau > `r_a + r_b` thì chắc chắn không chạm) và cho planner |
| `drive`, `omni`, `action_dim` | vi sai → 2 chiều (tiến, quay); đa hướng → 3 chiều (tiến, ngang, quay) |
| `v_max`, `v_min`, `vy_max`, `w_max` | giới hạn tốc độ; `v_min ≤ 0` là tốc độ lùi |
| `lidar` | `{"model","range_min","range_max","fov","beams"}`; `fov ≥ 2π` coi là 360° |

Mỗi con số trong JSON phải có `sources` (`official` + URL, hoặc `estimated` + cách ước lượng).

---

## Các mẫu robot có sẵn và thêm robot riêng

Các robot: `agilex_limo`, `clearpath_jackal`, `jetauto_pro`, `kobuki`, `pal_tiago`,
`turtlebot3_burger`, `turtlebot3_waffle_pi`, `turtlebot4`, `turtlebot4_lite`
(`navsim robots` in thông số).

**Thêm robot riêng:** viết file JSON schema 1.0 (xem một file trong `src/navsim/robots/presets/`),
rồi `navsim.load_robot("duong/dan/robot.json")`, hoặc đặt vào thư mục nằm trong `$NAVSIM_ROBOTS` để
dùng được theo id. Mỗi con số nên có mục `sources` (`official` + URL hoặc `estimated` + cách ước
lượng). Robot phải thuộc cùng `height_class` với map của dataset bạn dùng.
