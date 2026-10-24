import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env


def test_check_env(grid_env):
    check_env(grid_env.unwrapped, skip_render_check=True)


def test_spaces_and_info(grid_env):
    assert grid_env.action_space.n == 4
    obs, info = grid_env.reset(seed=0)
    assert set(obs) == {"lidar", "goal"}
    assert {"start", "goal", "d_bfs", "level", "map_id", "task_id", "split"} <= set(info)
    _, _, _, _, info = grid_env.step(0)
    assert {"steps", "collided", "goal_geodesic", "progress"} <= set(info)


def test_budget_truncates():
    env = gym.make("NavSim/GridNav-v0", dataset="demo-grid", budget=5)
    env.reset(seed=0)
    term = trunc = False
    for _ in range(5):
        _, _, term, trunc, info = env.step(0)
        if term or trunc:
            break
    assert term or trunc
    env.close()


def test_state(grid_env):
    grid_env.reset(seed=0)
    s = grid_env.unwrapped.robot_state()
    assert s.geodesic_distance() == s.start_geodesic
    assert s.shortest_path().shape[1] == 2


def test_shortest_path_reaches_goal(grid_env):
    """Following the shortest path cell by cell reaches the goal."""
    grid_env.reset(seed=0)
    s = grid_env.unwrapped.robot_state()
    path = s.shortest_path()
    moves = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    term = False
    info = {}
    for a, b in zip(path[:-1], path[1:]):
        d = (int(b[0] - a[0]), int(b[1] - a[1]))
        _, _, term, _, info = grid_env.step(moves[d])
    assert term and info["success"]
    assert np.isfinite(info["spl"])


def test_difficulty_filter():
    env = gym.make("NavSim/GridNav-v0", dataset="demo-grid", difficulty="easy")
    _, info = env.reset(seed=0)
    assert info["level"] == "easy"
    env.close()
