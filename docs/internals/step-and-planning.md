# Một lần step, shortest path và time limit

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
| `obs["lidar"]` | `n_beams` khoảng cách ([mục 6](./lidar.md)) |
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
[docs/concepts/robot_state.md](../concepts/robot_state.md). Hai giá trị hay dùng nhất là `progress_m`
(shortest path tới goal ngắn đi bao nhiêu mét trong step này) và `state.reached_goal`.


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
