# Wrappers

*(`wrappers.py`, `multiagent/nav_v0.py`)*

## `navsim.wrappers.Reward(env, fn)`

Sau mỗi `step`, tính `reward = float(fn(env.unwrapped.robot_state(), info))` và thay reward `0.0`. `fn` có
dạng `(state, info) -> số`.

```python
def dense(state, info):
    return info["progress_m"] + 10 * state.reached_goal - 0.1 * state.collided - 0.01
env = navsim.wrappers.Reward(gym.make("NavSim/Nav-v0"), dense)
```

Lưu ý: `progress_m` (grid: `progress`) cần `nav_info=True` (mặc định). Muốn nhớ giá trị giữa các step, dùng
class có `__call__` và đặt lại khi `state.steps == 1`. Đặt `Reward` **ngoài** `DiscreteActions` để có một reward
cho mỗi action rời rạc.

## `navsim.wrappers.ParallelReward(env, fn)` (= `nav_v0.ParallelReward`)

Wrapper PettingZoo: `rewards[a] = float(fn(env.robot_state(a), infos[a]))` cho từng agent còn trong `obs`.
Được nạp lười (`__getattr__`) vì cần PettingZoo.

## `navsim.wrappers.DiscreteActions(env, step_m=0.25, turn_deg=15.0)`

Chỉ cho `Nav-v0`. Đổi action thành `Discrete(5)` (hoặc 7 với robot omni): `0` đứng yên một `dt`, `1` tiến,
`2` lùi, `3` quay trái, `4` quay phải, `5`/`6` đi ngang trái/phải.

Cơ chế: mỗi action rời rạc được **mở ra thành một dãy action liên tục** chạy ở tốc độ tối đa
(`số_step_đầy = amount // (tốc_độ·dt)` + một step lẻ cho phần dư), và chạy tuần tự qua `env.step`. **Dừng
sớm** nếu va chạm, `terminated` hoặc `truncated`. Trả về obs cuối, **tổng reward** và **tổng `progress_m`**
qua các step; `info["macro_steps"]` là số step thực chạy.

## `gymnasium.wrappers.FlattenObservation(env)`

Của Gymnasium: gộp observation `Dict` của `Nav-v0` thành một vector phẳng (cho thuật toán chỉ nhận `Box`).
