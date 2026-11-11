# Head-camera estimate of the holders and the tube

`perception/ego_holders.py` takes ONE head-camera depth frame at the start of an episode. It returns:

- the stand pose (x, y, yaw)
- the rack pose (x, y, yaw)
- the tube position, and which rack hole the tube is in

The pipette pose is the stand estimate applied to its seat.

The student uses this estimate through an error model, not through images. Training therefore costs no extra rendering.

## Camera

| | Value | Status |
|---|---|---|
| Camera | `zed_left_camera` of the robot model (on `head_l3`, ZED X Mini, 50 mm baseline) | from the URDF |
| Neck | `head_l1` 0.878 rad about y, `head_l2` 0.017 rad about z, `head_l3` 0.386 rad about −y | the real robot's reading |
| Lens / mode | 2.2 mm lens (VFOV 80°), SVGA depth 960 × 600 | **assumed, confirm on the real unit** (`CAM`) |
| Depth noise | Stereo model: disparity noise 0.08 px white + 0.12 px correlated, edge bleeding, 3 % holes, 0.4 % scale | assumed (`NOISE`) |
| Head calibration error | 1.2 mm and 0.2° per axis | assumed (`CAL_ERR`) |

The holders are about 0.8 m from the camera, at about 2.4 mm per pixel. At the episode start the hands hover beside the holders. They do not cover them.

## Method (`estimate(depth, K, T_wc, Model(task))`, numpy / scipy only)

1. Back-project the depth into the world. The camera pose comes from the robot's forward kinematics.
2. Fit the table plane by RANSAC near the known table height. Heights are measured from the fitted plane, which absorbs camera pitch, roll and height error.
3. Find the coarse holder poses by height-map matching. Each holder's model is a top-down height map, ray-cast from the scene geoms. The fit searches (dx, dy, dyaw) in the env's parametrisation, coarse to fine:
   - model cells seen at the model height count +
   - cells seen lower (the table through the holder) count −
   - a ring around the footprint seen tall counts −
   - cells seen higher (pipette, tube, hand) are neutral
4. Refine the stand with a trimmed 3D ICP of stand plus seated pipette, from 7 yaw starts. The stand alone is small and nearly round, and its yaw error was 5.4°; the pipette's shape carries the yaw.
5. Find the tube. Choose the rack hole whose 10–19 mm ring is filled with points over the most 5 mm height bins between the rack top and the rim; a finger fills only a few. Fit a circle of the tube radius to the camera-facing half of the wall. Then take the hole nearest to that centre.

## Accuracy in simulation (`tasks/task_d02_straight_v2/ego/summary.json`)

400 scenes with the training randomisation:

- holders ±5 cm / ±10°
- a random pick hole
- object slack 2 mm / 3°
- stereo noise and calibration error as above

The teacher was trained with Gaussian holder noise of 5 mm / 2°.

| | Error std | 95th percentile |
|---|---|---|
| Stand x / y | 2.9 / 3.1 mm | 6.4 / 5.9 mm |
| Stand yaw | 1.9° | 3.8° |
| Rack x / y | 2.6 / 3.1 mm | 5.8 / 6.5 mm |
| Rack yaw | 0.45° | 0.9° |
| Tube x / y / z | 3.1 / 3.2 / 2.4 mm | 6.3 / 6.5 / 6.6 mm |
| Pick hole | 400 / 400 correct | |

Contributions:

- Without calibration error, positions are about 1–2.5 mm.
- The 0.2° camera yaw error alone moves points about 3 mm at 0.8 m.

A better head calibration is the main lever.

## Use in training (`rl/vega_extras.py`, `extras.holder.meas_model`)

```json
"meas_model": {"errors": "ego/errors.npz", "scale": 1.25, "extra_xy_mm": 1.0, "extra_yaw_deg": 0.5, "extra_tube_mm": 1.0,
               "pipette_xy_mm": 2.0, "pipette_yaw_deg": 1.5}
```

The model draws one row per episode: the stand, rack and tube errors from one characterised scene, so their correlations are kept.

- **Variance for randomisation:** the row is scaled by `scale`. Independent Gaussian noise (`extra_*`) is added on top.
- **Holders:** the estimate re-anchors the arm command, as the old 5 mm / 2° Gaussian did.
- **Pipette:** the stand estimate applied to the pipette, plus its slack in the stand, which the camera does not observe.
- **Tube:** the measured tube spin is 0, because the camera does not observe it.

Without `meas_model`, nothing changes.

Regenerate the error model after any change to the scene, camera or noise model:

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv-rl-head/bin/python -I perception/ego_holders.py tasks/task_d02_straight_v2 --n 400 --procs 28
```

It takes about 13 minutes on 28 cores. It also writes `overlays.png`: depth with the true (red) and estimated (green) outlines.

## On the real robot

Call `estimate(depth_m, K, T_wc, Model(task))` with:

- the ZED depth in metres (0 = invalid)
- its intrinsics `K` (OpenCV)
- the camera pose `T_wc` (OpenCV axes) in the scene's world frame: the head forward kinematics composed with the calibrated camera mount

The output offsets are in the env's convention. They go straight into the policy's measured-holder inputs.

The estimate takes 10–50 s on the CPU, once per episode.

### Not covered yet

- The real holders must match the scene meshes.
- The ZED's neural depth may bleed edges differently from the model.
- Clutter on the table is not modelled. The estimator only looks within ±10 cm of the nominal holder places.
