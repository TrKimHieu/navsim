# Chạy song song

## `navsim.env_fn(env_id="NavSim/Nav-v0", wrappers=(), **kwargs)`

Trả `functools.partial(_make_env, env_id, wrappers, kwargs)` — **pickle được**. Khi tiến trình con gọi nó:
`import gymnasium`, `gym.make(env_id, **kwargs)`, áp từng wrapper theo thứ tự. Quan trọng: tiến trình con
*mới khởi tạo* (Windows, macOS: kiểu `spawn`) chưa chạy `import navsim` nên chưa biết id `NavSim/...`; vì
`env_fn` nằm trong module `navsim`, việc unpickle nó đã import `navsim` — đó là lý do nó chạy mọi nơi.

Trên Linux, SB3 dùng `forkserver`; `navsim` tự gọi `_preload_in_forkserver()` lúc import để nạp thêm `navsim`
vào forkserver (không ảnh hưởng nếu forkserver đã chạy hoặc không có).

```python
venv = make_vec_env(navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg", split="train",
                                  wrappers=[lambda e: navsim.wrappers.Reward(e, dense)]),
                    n_envs=8, vec_env_cls=SubprocVecEnv)
```

Trong file `.py` đặt đoạn này dưới `if __name__ == "__main__":` (cần cho `spawn`). Wrapper dạng lambda dùng được với
`SubprocVecEnv` của SB3 vì nó pickle bằng cloudpickle; với thư viện chỉ dùng `pickle` chuẩn, hãy dùng hàm
đặt ở cấp module.

## `gym.make_vec("NavSim/Nav-v0", num_envs=8, vectorization_mode="async", …)`

Của Gymnasium; mỗi bản sao một tiến trình. Trên Windows/macOS cần `import navsim` trước, nếu không báo
`Namespace NavSim not found`.
