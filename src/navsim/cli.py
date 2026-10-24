"""The ``navsim`` command (also ``python -m navsim``).

    navsim setup isb-svg-v1     download a dataset (token: --token, $HF_TOKEN, Colab/Kaggle secret)
    navsim datasets             datasets on this machine
    navsim robots               robot models
    navsim info demo-svg        maps and tasks per robot and split
    navsim check                Gymnasium / PettingZoo API checks and speed
    navsim cache [--clear]      disk cache of loaded maps
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time


def print_table(rows: list[dict], columns: list[str]) -> None:
    width = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns} if rows else {}
    print("  ".join(c.ljust(width[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(width[c]) for c in columns))


def cmd_setup(args):
    from .data import setup
    setup(args.datasets, token=args.token, data_dir=args.data_dir, force=args.force)


def cmd_datasets(args):
    from .data import list_datasets
    print_table([{"name": d.name, "env": d.env_type, "robots": ",".join(d.robots), "path": str(d.root)}
                 for d in list_datasets()], ["name", "env", "robots", "path"])


def cmd_robots(args):
    from .robots import list_robots, load
    rows = []
    for robot_id in list_robots():
        r = load(robot_id)
        rows.append({"id": robot_id, "drive": r.drive, "radius_m": f"{r.r_circ:.3f}",
                     "v_max": r.v_max, "w_max": r.w_max, "lidar_m": r.lidar["range_max"],
                     "fov_deg": round(math.degrees(r.lidar["fov"])), "name": r.name})
    print_table(rows, ["id", "drive", "radius_m", "v_max", "w_max", "lidar_m", "fov_deg", "name"])


def cmd_info(args):
    from .data import load_dataset
    print(json.dumps(load_dataset(args.dataset).summary(), indent=1))


def cmd_check(args):
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env

    from .data import load_dataset
    ds = load_dataset(args.dataset)
    env_id = "NavSim/Nav-v0" if ds.env_type == "svg" else "NavSim/GridNav-v0"
    kwargs = {"dataset": args.dataset, **({"robot": args.robot} if args.robot else {})}
    env = gym.make(env_id, **kwargs)
    check_env(env.unwrapped, skip_render_check=True)
    print(f"{env_id} on {ds.name}: Gymnasium check_env passed")
    print(f"  observation {env.observation_space}\n  action      {env.action_space}")
    env.action_space.seed(0)
    env.reset(seed=0)
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            env.reset()
    print(f"  speed       {steps / (time.perf_counter() - t0):.0f} steps/s (one process, random actions)")
    if ds.env_type == "svg" and args.agents > 1:
        try:
            from pettingzoo.test import parallel_api_test

            from .multiagent import nav_v0
        except ImportError:
            print('multi-robot check skipped: pip install "navsim[marl]"')
            return
        parallel_api_test(nav_v0.parallel_env(n_agents=args.agents, **kwargs), num_cycles=200)
        print(f"navsim.multiagent.nav_v0 ({args.agents} robots): PettingZoo parallel_api_test passed")


def cmd_cache(args):
    from ._sim import cache
    if args.clear is not None:
        removed = cache.clear(args.clear or None)
        print(f"removed {len(removed)} folder(s)")
        return
    print(f"cache root: {cache.cache_root() or 'disabled (NAVSIM_CACHE=off)'}")
    print_table([{**e, "mb": f"{e['mb']:.1f}"} for e in cache.entries()], ["dataset", "files", "mb", "path"])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="navsim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="make datasets available (download from Hugging Face if needed)")
    s.add_argument("datasets", nargs="*", default=["demo-svg"])
    s.add_argument("--token", help="Hugging Face token (default: $HF_TOKEN or the Colab/Kaggle secret)")
    s.add_argument("--data-dir", help="download folder (default: ~/.navsim/datasets or $NAVSIM_HOME)")
    s.add_argument("--force", action="store_true", help="download again")
    s.set_defaults(fn=cmd_setup)
    sub.add_parser("datasets", help="datasets on this machine").set_defaults(fn=cmd_datasets)
    sub.add_parser("robots", help="robot models").set_defaults(fn=cmd_robots)
    s = sub.add_parser("info", help="maps and tasks of a dataset")
    s.add_argument("dataset")
    s.set_defaults(fn=cmd_info)
    s = sub.add_parser("check", help="API checks and speed of the environments on a dataset")
    s.add_argument("--dataset", default="demo-svg")
    s.add_argument("--robot")
    s.add_argument("--agents", type=int, default=3, help="robots for the multi-robot check (svg)")
    s.add_argument("--seconds", type=float, default=2.0, help="duration of the speed test")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("cache", help="disk cache of loaded maps")
    s.add_argument("--clear", nargs="?", const="", metavar="DATASET",
                   help="remove the cache (of one dataset, or everything)")
    s.set_defaults(fn=cmd_cache)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        args.fn(args)
    except (FileNotFoundError, PermissionError, ConnectionError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
