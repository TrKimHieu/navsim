import subprocess
import sys


def run(*args):
    return subprocess.run([sys.executable, "-m", "navsim", *args], capture_output=True, text=True)


def test_robots():
    r = run("robots")
    assert r.returncode == 0 and "turtlebot4" in r.stdout


def test_datasets_and_info():
    assert "demo-svg" in run("datasets").stdout
    assert run("info", "demo-grid").returncode == 0


def test_unknown_dataset_errors():
    assert run("info", "does-not-exist").returncode == 1
