"""Pictures and GIFs of episodes, for checking a policy or presenting results.

    navsim.plot_episode(env, save="episode.png")          # map, path, start, goal, robot, LiDAR
    navsim.plot_episode(env, shortest_path=True, ax=ax)    # + the shortest path to the goal, in your Axes
    img = navsim.render_frame(env)                         # (H, W, 3) image of the current step
    navsim.save_gif(frames, "run.gif", fps=10)             # your own list of frames -> GIF
    navsim.record_video(env, model, "run.gif")             # run a policy and save it as a GIF

All of them work with ``NavSim/Nav-v0`` and ``NavSim/GridNav-v0`` (wrapped or not) and with
the multi-robot ``nav_v0.parallel_env`` (one colour per robot), after ``reset`` and at any
step. Maps are drawn in metres (x right, y up), grids in cells (column right, row down). The
LiDAR drawn is the noise-free scan from the current pose. Drawing never touches the
environment's random generators, so it does not change the episode.

Two frame styles: ``"plot"`` (matplotlib, like ``plot_episode``: axes, legend, LiDAR; for
slides) and ``"fast"`` (the environment's own small image; much faster, no matplotlib).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ._sim.geometry import place
from ._sim.world import COLORS, RobotState
from .evaluation import as_policy, default_deterministic, reset_policy

PATH_COLOR = (80 / 255, 140 / 255, 230 / 255)
GOAL_COLOR = (220 / 255, 0.0, 0.0)
SHORTEST_COLOR = (0.15, 0.65, 0.25)


# ------------------------------------------------------------------ pictures

def plot_episode(env, ax=None, lidar: bool = True, path: bool = True, shortest_path: bool = False,
                 title: bool = True, legend: bool = True, figsize=(8, 8), save=None, dpi: int = 150):
    """Draw the map, the path driven since ``reset`` (with its start), the goal, the robot at
    its current pose and its LiDAR rays; ``shortest_path=True`` adds the shortest path from
    the robot to the goal. ``save``: also write the figure to this file (.png, .pdf, .svg).
    Returns the matplotlib ``Axes``."""
    draw, u = _drawer(env)
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    draw(ax, u, lidar, path, shortest_path, title, legend)
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save, dpi=dpi, bbox_inches="tight")
    return ax


def render_frame(env, style: str = "plot", size: int = 480, lidar: bool = True,
                 shortest_path: bool = False) -> np.ndarray:
    """An RGB image (H, W, 3) of the current step, at most ``size`` pixels wide or high.
    ``style``: "plot" (like ``plot_episode``) or "fast" (the environment's own image)."""
    if style == "fast":
        return _shrink(_fast_frame(env), size)
    if style != "plot":
        raise ValueError(f"style must be 'plot' or 'fast', got {style!r}")
    draw, u = _drawer(env)
    _pyplot()                                          # clear error if matplotlib is missing
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6, 6), dpi=size / 6)         # not shown in the notebook
    canvas = FigureCanvasAgg(fig)
    draw(fig.add_subplot(), u, lidar, True, shortest_path, True, True)
    fig.tight_layout()
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames, path, fps: int = 10) -> Path:
    """Write a list of RGB images (e.g. from ``render_frame``) as an animated GIF."""
    from PIL import Image
    if not frames:
        raise ValueError("no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=False)
    return path


def record_video(env, policy=None, path="videos/episode.gif", episodes: int = 1,
                 seed: int | None = 0, options=None, fps: int | None = None,
                 max_steps: int = 1000, every: int = 1, size: int = 480, style: str = "plot",
                 deterministic: bool | None = None, show: bool = True) -> Path:
    """Run ``episodes`` episodes with ``policy`` and save every ``every``-th step as a GIF.

    ``policy``: a model with ``predict`` (Stable-Baselines3), a function obs -> action, or
    None for random actions; ``deterministic`` None = the most likely action, except on
    the grid (see ``navsim.evaluate``). ``style``/``size``: see ``render_frame``; ``fps``
    default: the environment's ``render_fps``. In a notebook the GIF is shown (``show``)."""
    if deterministic is None:
        deterministic = default_deterministic(env)
    multi = hasattr(env, "possible_agents")
    if multi:
        policies = {a: as_policy(policy[a] if isinstance(policy, dict) else (policy or "random"),
                                 env.action_space(a), deterministic) for a in env.possible_agents}
    else:
        act = as_policy(policy or "random", env.action_space, deterministic)
    frames = []
    for e in range(int(episodes)):
        obs, _ = env.reset(seed=None if seed is None else seed + e, options=options)
        if policy is not None and not isinstance(policy, dict):
            reset_policy(policy)
        frames.append(render_frame(env, style, size))
        for t in range(int(max_steps)):
            if multi:
                if not env.agents:
                    break
                obs, *_ = env.step({a: policies[a](obs[a]) for a in env.agents})
                done = not env.agents
            else:
                obs, _, terminated, truncated, _ = env.step(act(obs))
                done = terminated or truncated
            if done or t % every == 0:
                frames.append(render_frame(env, style, size))
            if done:
                break
    fps = fps or (getattr(env, "metadata", {}) or {}).get("render_fps", 10)
    path = save_gif(frames, path, fps / every)
    if show:
        _show_in_notebook(path)
    return path


# ------------------------------------------------------------------ drawing

def _drawer(env):
    u = getattr(env, "unwrapped", env)
    if hasattr(u, "world"):
        return _draw_world, u
    if hasattr(u, "agent_pos") and hasattr(u, "lidar_rays"):
        return _draw_grid, u
    raise TypeError(f"navsim can draw NavSim environments (NavSim/Nav-v0, NavSim/GridNav-v0, "
                    f"navsim.multiagent.nav_v0), not {type(u).__name__}")


def show_frame(env, img: np.ndarray) -> None:
    """``render_mode="human"``: show ``img`` in the environment's own matplotlib window
    (opened on the first call, then updated) and wait one frame (``render_fps``). For a
    desktop script; in a notebook use ``render_mode="rgb_array"`` and ``record_video``."""
    plt = _pyplot()
    window = getattr(env, "_human_window", None)
    if window is None or not plt.fignum_exists(window[0].number):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        fig.tight_layout()
        window = env._human_window = (fig, ax.imshow(img))
    else:
        window[1].set_data(img)
    plt.pause(1 / env.metadata["render_fps"])


def close_window(env) -> None:
    """Close the window of ``show_frame``, if any."""
    window = getattr(env, "_human_window", None)
    if window is not None:
        _pyplot().close(window[0])
        env._human_window = None


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('drawing needs matplotlib: pip install "navsim[plot]" (or pip install '
                          'matplotlib), or use style="fast"') from None
    return plt


def _rgb(k: int) -> tuple:
    return tuple(v / 255 for v in COLORS[k % len(COLORS)])


def _rays(ax, x0, y0, ends, hit, color, label):
    """One line per ray, and a dot where a ray hit something (not at its maximum range)."""
    from matplotlib.collections import LineCollection
    segments = np.stack([np.broadcast_to([x0, y0], ends.shape), ends], 1)
    ax.add_collection(LineCollection(segments, colors=[(*color, 0.35)], linewidths=0.6, label=label))
    ax.plot(ends[hit, 0], ends[hit, 1], ".", ms=2.5, color=color, label=f"{label} hits")


def _draw_world(ax, u, lidar, path, shortest_path, title, legend):
    from matplotlib.patches import Circle, Polygon
    world = u.world
    if world.map_data is None or all(a.task is None for a in world.agents):
        raise ValueError("reset the environment before drawing it")
    mask, x0, y1, res = world.map_data.svgmap.raster(0.04)
    H, W = mask.shape
    ax.imshow(mask, cmap="gray", vmin=-0.45, vmax=1.0, interpolation="nearest",
              extent=[x0, x0 + W * res, y1 - H * res, y1])
    one = world.n_robots == 1
    names = getattr(u, "possible_agents", None) or ["robot"]
    for k, agent in enumerate(world.agents):
        if agent.task is None:
            continue
        prefix = "" if one else f"{names[k]} "
        body = _rgb(k)
        line = PATH_COLOR if one else body
        x, y, theta = agent.pose
        if lidar and agent.present:
            spec = agent.robot.lidar
            d = np.clip(world.scan(k), spec["range_min"], spec["range_max"])
            angles = theta + agent.beam_angles
            _rays(ax, x, y, np.stack([x + d * np.cos(angles), y + d * np.sin(angles)], 1),
                  d < spec["range_max"] - 1e-6, GOAL_COLOR if one else body, f"{prefix}lidar")
        if shortest_path and agent.present and not agent.done:
            best = RobotState(world, k).shortest_path()
            if len(best):
                ax.plot(best[:, 0], best[:, 1], "--", color=SHORTEST_COLOR if one else body, lw=1.5,
                        label=f"{prefix}shortest path")
        if path:
            t = np.asarray(agent.traj)
            ax.plot(t[:, 1], t[:, 2], "-", color=line, lw=1.5, label=f"{prefix}path")
            ax.plot(t[0, 1], t[0, 2], "o", ms=7, mfc="white", mec=line, mew=2, label=f"{prefix}start")
        goal = agent.task["goal"]
        ax.add_patch(Circle((goal["x"], goal["y"]), agent.success_radius, fill=one, lw=2,
                            color=GOAL_COLOR if one else body, alpha=0.85, label=f"{prefix}goal"))
        if not one:
            ax.plot(goal["x"], goal["y"], "*", ms=12, color=body, mec="black", mew=0.5,
                    label=f"{prefix}goal marker")
        if agent.present:
            ax.add_patch(Polygon(place(agent.robot.footprint, x, y, theta), closed=True, color=body,
                                 alpha=0.9, label=f"{prefix}robot"))
            r = agent.robot.r_circ
            ax.plot([x, x + r * np.cos(theta)], [y, y + r * np.sin(theta)], "-", color="black",
                    lw=1.5, label=f"{prefix}heading")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        placed = [a for a in world.agents if a.task is not None]
        what = f"task {placed[0].task_id}" if one else f"{len(placed)} robots"
        ax.set_title(f"{world.map_data.map_id} · {what} · t = {max(a.time for a in placed):.1f} s")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if one:
            keep = ("path", "start", "goal", "robot", "lidar", "shortest path")
            pairs = [(h, lab) for h, lab in zip(handles, labels) if lab in keep]
        else:                                        # one entry per robot, named by its agent
            pairs = [(h, lab[:-len(" robot")]) for h, lab in zip(handles, labels) if lab.endswith(" robot")]
        if pairs:
            ax.legend(*zip(*pairs), loc="upper right", fontsize="small")


def _draw_grid(ax, u, lidar, path, shortest_path, title, legend):
    if u.agent_pos is None:
        raise ValueError("reset the environment before drawing it")
    H, W = u.grid.shape
    ax.imshow(u.grid == 0, cmap="gray", vmin=-0.45, vmax=1.0, interpolation="nearest",
              extent=[-0.5, W - 0.5, H - 0.5, -0.5])
    row, col = u.agent_pos
    if lidar:
        d = u.lidar_ranges()
        angles = np.linspace(0, 2 * np.pi, u.lidar_rays, endpoint=False)
        # the grid LiDAR starts at (row, col) = the top-left corner of the robot's cell, or
        # at its centre with lidar_centered
        off = 0.0 if u.lidar_centered else -0.5
        _rays(ax, col + off, row + off,
              np.stack([col + off + d * np.sin(angles), row + off + d * np.cos(angles)], 1),
              d < u.lidar_range - 1e-6, GOAL_COLOR, "lidar")
    if shortest_path:
        best = u.robot_state().shortest_path()
        if len(best):
            ax.plot(best[:, 1], best[:, 0], "--", color=SHORTEST_COLOR, lw=1.5, label="shortest path")
    if path:
        t = np.asarray(u.traj)
        ax.plot(t[:, 1], t[:, 0], "-", color=PATH_COLOR, lw=2, label="path")
        ax.plot(t[0, 1], t[0, 0], "o", ms=8, mfc="white", mec=PATH_COLOR, mew=2, label="start")
    ax.plot(u.goal[1], u.goal[0], "s", ms=9, color=GOAL_COLOR, label="goal")
    ax.plot(col, row, "o", ms=9, color=_rgb(0), label="robot")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if title:
        ax.set_title(f"{u.map_id} · task {u.task_idx} · step {u.steps}")
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(*zip(*[(h, lab) for h, lab in zip(handles, labels) if lab != "lidar hits"]),
                  loc="upper right", fontsize="small")


def _fast_frame(env) -> np.ndarray:
    """The environment's own RGB image, whatever render_mode it was made with."""
    if getattr(env, "render_mode", None) == "rgb_array":
        return env.render()
    u = getattr(env, "unwrapped", env)
    if hasattr(u, "world"):
        return u.world.render_rgb()
    if hasattr(u, "rgb_frame"):
        return u.rgb_frame()
    raise ValueError("this environment cannot render frames: make it with render_mode='rgb_array'")


def _shrink(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _show_in_notebook(path: Path) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import Image as IPImage, display
    except ImportError:
        return
    if get_ipython() is not None:
        display(IPImage(filename=str(path)))
