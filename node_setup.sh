#!/usr/bin/env bash
# One-time setup on a GPU node (e.g. B200 / Blackwell: needs the CUDA 13 torch wheels pinned in requirements_rl_head.txt, driver >= 580).
# Needs internet access and uv (https://docs.astral.sh/uv/). Then runs the CPU unit tests and prints the GPU.
#   ./node_setup.sh
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found: install it first (https://docs.astral.sh/uv/getting-started/installation/)"; exit 1; }
uv venv --python 3.12.15 .venv-rl-head
uv pip install --python .venv-rl-head -r requirements_rl_head.txt
uv pip install --python .venv-rl-head --no-deps "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@4911167e3f2d"
.venv-rl-head/bin/python -I rl/test_contact_terms.py | tail -1
.venv-rl-head/bin/python -I rl/test_arm_clamp.py | tail -1
.venv-rl-head/bin/python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))"
