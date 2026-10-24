import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import navsim


def test_check_env(nav_env):
    check_env(nav_env.unwrapped, skip_render_check=True)


def test_reset_step_contract(nav_env):
    obs, info = nav_env.reset(seed=0)
    assert nav_env.observation_space.contains(obs)
    assert {"map_id", "task_id", "split", "geodesic_m", "time_limit"} <= set(info)
    obs, reward, terminated, truncated, info = nav_env.step(nav_env.action_space.sample())
    assert reward == 0.0
    assert {"collided", "n_collisions", "time", "pose", "path_length"} <= set(info)
    assert {"goal_geodesic_m", "progress_m", "goal_distance_m", "wall_distance_m"} <= set(info)


def test_seed_is_deterministic():
    def run():
        env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
        env.reset(seed=3)
        env.action_space.seed(3)
        out = [env.step(env.action_space.sample())[0]["lidar"].copy() for _ in range(20)]
        env.close()
        return np.stack(out)
    assert np.array_equal(run(), run())


def test_episode_ends_with_summary(nav_env):
    nav_env.reset(seed=1)
    info = {}
    for _ in range(100000):
        _, _, term, trunc, info = nav_env.step(np.array([0.0, 0.0], dtype=np.float32))
        if term or trunc:
            break
    else:
        raise AssertionError("episode never ended")
    assert info["termination"] in {"success", "collision", "time_limit"}
    assert 0.0 <= info["spl"] <= 1.0


def test_choose_task_and_custom_goal(nav_env):
    nav_env.reset(seed=0)
    u = nav_env.unwrapped
    m = u.map_ids[0]
    _, info = nav_env.reset(options={"map_id": m, "task_idx": 0})
    assert info["map_id"] == m
    t = u.tasks_of(m)[0]
    _, info = nav_env.reset(options={"map_id": m, "start": (t["start"]["x"], t["start"]["y"], 0.0),
                                     "goal": (t["goal"]["x"], t["goal"]["y"])})
    assert info["task_id"] == -1


def test_task_filter(nav_env):
    u = nav_env.unwrapped
    all_tasks = u.set_task_filter()
    some = u.set_task_filter(geodesic_m=(0.0, 5.0))
    assert some["tasks"] <= all_tasks["tasks"]
    u.set_task_filter()


def test_robot_state(nav_env):
    nav_env.reset(seed=0)
    nav_env.step(nav_env.action_space.sample())
    s = nav_env.unwrapped.robot_state()
    assert len(s.pose) == 3 and len(s.goal) == 2
    assert s.geodesic_distance() >= s.goal_distance() - 1e-6
    path = s.shortest_path()
    assert path.ndim == 2 and path.shape[1] == 2
    assert s.trajectory.shape[1] == 4 and len(s.trajectory) == s.steps + 1  # start pose + one row per step


def test_reward_wrapper(nav_env):
    env = navsim.wrappers.Reward(nav_env, lambda state, info: 2.5)
    env.reset(seed=0)
    assert env.step(env.action_space.sample())[1] == 2.5


def test_discrete_actions(nav_env):
    env = navsim.wrappers.DiscreteActions(nav_env)
    assert env.action_space.n == 5
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)
    assert info["macro_steps"] >= 1


def test_action_dim_differential():
    env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="clearpath_jackal")
    assert env.action_space.shape == (2,)
    env.close()


def test_collision_terminate_mode():
    env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", on_collision="terminate")
    env.reset(seed=0)
    term = trunc = False
    for _ in range(3000):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if term or trunc:
            break
    assert term or trunc
    env.close()


def test_render_and_plot(nav_env, tmp_path):
    nav_env.reset(seed=0)
    nav_env.step(nav_env.action_space.sample())
    frame = navsim.render_frame(nav_env, style="fast", size=200)
    assert frame.dtype == np.uint8 and frame.shape[2] == 3
    frames = [frame, navsim.render_frame(nav_env, style="fast", size=200)]
    gif = navsim.save_gif(frames, tmp_path / "x.gif")
    assert gif.exists()
    navsim.plot_episode(nav_env, shortest_path=True, save=str(tmp_path / "e.png"))
    assert (tmp_path / "e.png").exists()


def test_evaluate_random():
    res = navsim.evaluate("random", "NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4",
                          split="test", episodes=3)
    assert res["episodes"] == 3
    assert 0.0 <= res["success"] <= 1.0
    assert {"spl", "collision_episodes", "mean_progress", "mean_time", "mean_path_m"} <= set(res)


def test_env_fn_and_vec():
    venv = gym.make_vec("NavSim/Nav-v0", num_envs=2, vectorization_mode="sync",
                        dataset="demo-svg", robot="turtlebot4")
    obs, _ = venv.reset(seed=0)
    assert obs["lidar"].shape[0] == 2
    venv.close()
    env = navsim.env_fn("NavSim/Nav-v0", dataset="demo-svg")()
    env.reset(seed=0)
    env.close()
