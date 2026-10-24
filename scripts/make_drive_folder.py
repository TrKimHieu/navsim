"""Make GGDrive/navsim/ = the wheel + README.md + README.vi.md, ready to copy to Google Drive.

    python scripts/make_drive_folder.py

Uses dist/navsim-<version>-py3-none-any.whl (run scripts/smoke_wheel.py first, which builds it);
builds it if there is none. The folder is recreated each time.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "GGDrive" / "navsim"


def find_wheel() -> Path:
    wheels = sorted((ROOT / "dist").glob("navsim-*-py3-none-any.whl"))
    if not wheels:
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-w", str(ROOT / "dist")], check=True)
        wheels = sorted((ROOT / "dist").glob("navsim-*-py3-none-any.whl"))
    return wheels[-1]


def main() -> None:
    wheel = find_wheel()
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    for src in (wheel, ROOT / "README.md", ROOT / "README.vi.md"):
        shutil.copy2(src, OUT / src.name)
    print(f"{OUT}:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}  {f.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
