# Công thức hay dùng

**Chạy một episode với policy ngẫu nhiên:**

```python
env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
obs, info = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
print(info["success"], info["termination"], info["spl"])
```

**Cho robot đi từ A tới B tự chọn:**

```python
env.reset(options={"map_id": env.unwrapped.map_ids[0], "start": (2.0, 12.0), "goal": (4.8, 10.7)})
```

**Huấn luyện với Stable-Baselines3** (cài `stable-baselines3` riêng):

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

venv = make_vec_env(navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg", split="train",
                                  wrappers=[lambda e: navsim.wrappers.Reward(e, dense)]),
                    n_envs=8, vec_env_cls=SubprocVecEnv)
model = PPO("MultiInputPolicy", venv).learn(200_000)
print(navsim.evaluate(model, "NavSim/Nav-v0", dataset="demo-svg", split="test"))
```

(`MultiInputPolicy` vì observation là `Dict`.)

**Học từ dễ tới khó (curriculum):**

```python
env.unwrapped.set_task_filter(geodesic_m=(0.0, 4.0))      # chặng ngắn trước
# ... huấn luyện một thời gian ...
env.unwrapped.set_task_filter(geodesic_m=(0.0, 12.0))
env.unwrapped.set_task_filter()                            # rồi tất cả
```

**Tự đọc dữ liệu thô:**

```python
ds = navsim.load_dataset("demo-grid")
a = ds.grid_arrays("demo-S040_s0")          # {"grid", "state", "main"}
ds.grid_meta("demo-S040_s0")["cell_m"]      # 0.325
dsv = navsim.load_dataset("demo-svg")
dsv.svg_path("turtlebot4", "demo-S001_s0")  # đường dẫn file .svg, mở bằng trình duyệt
```

---
