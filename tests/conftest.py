import gymnasium as gym
import pytest

import navsim  # noqa: F401  (registers the NavSim/* ids)


@pytest.fixture
def nav_env():
    env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
    yield env
    env.close()


@pytest.fixture
def grid_env():
    env = gym.make("NavSim/GridNav-v0", dataset="demo-grid", goal_obs=True)
    yield env
    env.close()
