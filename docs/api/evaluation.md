# `evaluate`

*(`evaluation.py`)*

```python
navsim.evaluate(policy, env="NavSim/Nav-v0", *, split="test", episodes=None, per_map=None, seed=0,
                deterministic=None, progress=False, return_episodes=False, **env_kwargs) -> dict
```

**Cơ chế:**
1. Nếu `env` là chuỗi: `gym.make(env, split=split, **env_kwargs)`. Nếu `env` là đối tượng mà bạn vẫn truyền
   `env_kwargs` → `TypeError`.
2. `deterministic=None` → `True`, **trừ `GridNav-v0`** (lấy mẫu, xem [GridNav-v0](../internals/simulator.md#navsimgridnav-v0)).
3. Chuyển `policy` thành hàm `obs → action`: chuỗi `"random"`, đối tượng có `predict(obs, deterministic=…)`
   (SB3), hoặc hàm gọi được; khác thì `TypeError`. Nếu policy có `reset()`, nó được gọi đầu mỗi episode.
4. Kế hoạch episode: `episodes=None` → **mọi task của mọi bản đồ một lần, theo thứ tự**
   (`per_map` giới hạn số task mỗi bản đồ), mỗi episode `reset(seed=seed, options={"map_id","task_idx"})`.
   `episodes=N` → N episode ngẫu nhiên với seed `seed, seed+1, …`.
5. Môi trường nhiều robot (`possible_agents`): mặc định 20 episode; `policy` có thể là dict `{agent: policy}`.
6. Tổng hợp `info` cuối mỗi episode.

**Trả về:** `split`, `episodes`, `success`, `spl`, `collision_episodes` (tỉ lệ episode có ≥1 va chạm),
`mean_progress` (phần đường ngắn nhất đã đi được lúc kết thúc: 1 nếu thành công, âm nếu cuối cùng xa hơn
lúc đầu), `mean_time`, `mean_path_m`. Thêm `deterministic` (với model), `episode_infos` (`return_episodes`),
và với nhiều robot: `robots`, `robot_episodes`, `team_success` (tỉ lệ episode mà **cả đội** tới đích).

```python
navsim.evaluate(model, "NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4")
navsim.evaluate("random", nav_v0.parallel_env(n_agents=4, split="test"), episodes=50)
```
