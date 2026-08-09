# Warehouse Traversability — Isaac Sim Baseline

This repository records the first simulation baseline for a BTP on learning traversability in warehouse environments.

## Current baseline

- Isaac Sim 6.0.1 runs on a remote NVIDIA L4 GPU VM via WebRTC.
- A `Simple_Warehouse` scene contains a lightweight, physics-enabled Nova Carter AMR.
- `scripts/warehouse_fixed_run.py` commands a five-second straight drive and writes a CSV motion log.
- The controller deliberately uses no ROS, cameras, or LiDAR yet; it is a stable motion-and-logging baseline.

## Run the experiment

Open the warehouse stage in Isaac Sim, then open `Window → Script Editor` and run:

```python
exec(open("/home/user/isaac_assets/warehouse_fixed_run.py").read())
start_experiment()
```

The robot settles for one second, drives for five seconds, stops, and writes a timestamped file at:

```text
/home/user/isaac_assets/nova_carter_fixed_run_YYYYMMDD_HHMMSS.csv
```

The script expects the robot reference root at `/World/nova_carter_reference`.

## CSV schema

| Column | Meaning |
| --- | --- |
| `elapsed_s` | Seconds since the drive command began. |
| `x_m`, `y_m` | Chassis position in the warehouse frame, in metres. |
| `yaw_rad` | Chassis heading in radians. |
| `speed_mps` | Finite-difference speed estimated from successive pose samples. |
| `command_wheel_deg_s` | Requested wheel angular speed in degrees per second. |

## Repository layout

- `scripts/warehouse_fixed_run.py` — Script Editor-compatible fixed-run controller/logger.
- `tests/test_warehouse_fixed_run.py` — Local tests for log schema and robot-path selection helpers.
- `01_problem_understanding.md`, `02_traversability_survey.md` — project framing and literature notes.
- `docs/superpowers/specs/` — baseline experiment design.
- `docs/superpowers/plans/` — implementation plan.

## Next research milestones

1. Verify the corrected script returns roughly 50 pose samples in a five-second run.
2. Add one perception source (camera or LiDAR) and synchronize it with pose/velocity.
3. Introduce controlled warehouse hazards or surface changes.
4. Define a traversability label from robot outcomes: speed loss, collision/contact, slip, or controller intervention.

## Important boundaries

Do not commit cloud credentials, SSH keys, local Isaac Sim caches, or generated run logs by default. Keep selected, reviewed datasets in a separate versioned dataset release later.
