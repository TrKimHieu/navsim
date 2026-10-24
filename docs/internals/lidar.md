# LiDAR

## Nguyên lý

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

## LiDAR của `Nav-v0` (map liên tục)

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

## LiDAR của `GridNav-v0` (grid)

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
