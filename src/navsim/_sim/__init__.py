"""The simulator (internal; you never need to import it).

    world.py      robots moving on a map: motion, collisions, LiDAR, episode end; RobotState
    geometry.py   collision checks and LiDAR rays on a map
    kernels.py    the same geometry compiled with numba (optional, faster)
    planning.py   shortest paths (time limit, SPL, RobotState.shortest_path)
    svgmap.py     reading SVG floor plans
    cache.py      disk cache of loaded maps
"""
