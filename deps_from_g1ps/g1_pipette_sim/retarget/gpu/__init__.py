"""GPU-batched stage-2 hand retargeting (feasibility prototype, 2026-10-26).

export.py (repo .venv, CPU): runs tools/retarget_demo.py `hand` with the production flags but captures
topo.run_scene's arguments instead of solving, and flattens them (targets, attraction entries, scene,
interaction-mesh Laplacians, collision model for MuJoCo Warp) into one npz + MJCF per hand.
solver.py (conda env with torch + mujoco_warp): whole-trajectory Levenberg-Marquardt over all frames
at once, collisions from MuJoCo Warp (worlds = frames and between-frame midpoints).
"""
