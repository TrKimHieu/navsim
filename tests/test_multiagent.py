import pytest

pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402

import navsim  # noqa: E402
from navsim.multiagent import nav_v0  # noqa: E402


def test_parallel_api():
    parallel_api_test(nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=3), num_cycles=100)


def test_episode_runs_until_all_done():
    env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=2, split="train")
    env.reset(seed=0)
    assert env.agents == ["robot_0", "robot_1"]
    for _ in range(100000):
        if not env.agents:
            break
        env.step({a: env.action_space(a).sample() for a in env.agents})
    assert not env.agents
    env.close()


def test_global_state_shape():
    env = nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=3)
    env.reset(seed=0)
    assert env.state().shape == (30,)
    env.close()


def test_parallel_reward_wrapper():
    env = navsim.wrappers.ParallelReward(
        nav_v0.parallel_env(dataset="demo-svg", robot="turtlebot4", n_agents=2), lambda s, i: 1.0)
    env.reset(seed=0)
    _, rewards, *_ = env.step({a: env.action_space(a).sample() for a in env.agents})
    assert set(rewards.values()) == {1.0}
