"""Build dist/navsim-<version>-py3-none-any.whl and test it in a clean virtual environment.

    python scripts/smoke_wheel.py [--extras marl,plot] [--keep]

The test runs from a temporary folder (not the project), so it imports the installed wheel,
not src/. It checks that the demo datasets and robot presets are inside the wheel and runs
`navsim check`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SNIPPET = """
import gymnasium as gym, navsim
assert navsim.__file__ and "site-packages" in navsim.__file__.replace("\\\\", "/"), navsim.__file__
assert len(navsim.list_robots()) == 9
for name in ("demo-svg", "demo-grid"):
    assert navsim.load_dataset(name).validate() == [], name
env = gym.make("NavSim/Nav-v0", dataset="demo-svg", robot="turtlebot4", split="train")
env.reset(seed=0)
env.step(env.action_space.sample())
print("wheel OK", navsim.__version__)
"""


def run(cmd: list[str], **kw) -> None:
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def build() -> Path:
    dist = ROOT / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(dist)])
    wheels = sorted(dist.glob("navsim-*-py3-none-any.whl"))
    if len(wheels) != 1:
        sys.exit(f"expected one wheel in {dist}, found {wheels}")
    return wheels[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extras", default="marl", help="extras to install with the wheel (comma separated)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary virtual environment")
    args = ap.parse_args()

    wheel = build()
    print(f"built {wheel} ({wheel.stat().st_size / 1024:.0f} kB)")

    tmp = Path(tempfile.mkdtemp(prefix="navsim-smoke-"))
    try:
        venv.EnvBuilder(with_pip=True).create(tmp / "venv")
        py = tmp / "venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        spec = f"{wheel}[{args.extras}]" if args.extras else str(wheel)
        run([str(py), "-m", "pip", "install", "-q", spec])
        work = tmp / "work"
        work.mkdir()
        run([str(py), "-I", "-c", SNIPPET], cwd=work)
        run([str(py), "-m", "navsim", "check", "--seconds", "1"], cwd=work)
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"smoke test passed: {wheel.name}")


if __name__ == "__main__":
    main()
