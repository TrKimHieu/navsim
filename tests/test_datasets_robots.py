import pytest

import navsim


def test_demo_datasets_listed():
    names = {d.name for d in navsim.list_datasets()}
    assert {"demo-svg", "demo-grid"} <= names


@pytest.mark.parametrize("name", ["demo-svg", "demo-grid"])
def test_dataset_valid(name):
    ds = navsim.load_dataset(name)
    assert ds.validate() == []
    assert set(ds.splits) >= {"train", "val", "test"}
    assert [len(ds.splits[s]) for s in ("train", "val", "test")] == [3, 2, 1]


def test_dataset_tasks():
    ds = navsim.load_dataset("demo-svg")
    robot = "turtlebot4"
    map_id = ds.maps(robot, "train")[0]
    tasks = ds.tasks(robot, map_id)
    assert tasks
    t = tasks[0]
    assert {"id", "start", "goal", "geodesic_m", "euclidean_m", "labels"} <= set(t)
    assert t["geodesic_m"] >= t["euclidean_m"] - 1e-6
    assert ds.svg_path(robot, map_id).exists()


def test_robots_load():
    ids = navsim.list_robots()
    assert len(ids) == 9
    for rid in ids:
        r = navsim.load_robot(rid)
        assert r.v_max > 0 and r.v_min <= 0 and r.w_max > 0
        assert r.r_circ > 0
        assert r.action_dim == (3 if r.drive == "omnidirectional" else 2)
        assert len(r.footprint) >= 3


def test_default_download_repo():
    from navsim.data import download
    assert download.REPO_ID == "PhysicalSI/2D_Navigation_Sim"
