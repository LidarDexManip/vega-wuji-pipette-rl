"""Physically valid replay of the retargeted pipetting demo (Do As I Do physics stage).

Modules: `proxies` (primitive furniture fitted to the LIDAR scan), `daid_scene` (the MJCF the
physics runs on: two floating Wuji hands, the pipette with its plunger on a welded carrier, the
tube, the proxies), `daid_stage` (stage cache -> Do As I Do inputs for a time window),
`daid_checks` (CPU checks, MuJoCo/numpy only, also importable in the Do As I Do conda env) and
`daid_inputs` (the input gate, milestone M0). Driver: tools/daid_physics.py.
"""
