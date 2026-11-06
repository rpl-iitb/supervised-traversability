"""Run a short, logged Nova Carter drive from Isaac Sim's Script Editor.

Run this file in an already-open warehouse stage, then call start_experiment().
The script deliberately has no ROS or sensor dependency.
"""

import asyncio
import csv
import math
import time
from datetime import datetime
from pathlib import Path


ROBOT_PATH = "/World/nova_carter_reference"
LOG_DIRECTORY = Path.home() / "isaac_assets"
SETTLE_SECONDS = 1.0
DRIVE_SECONDS = 5.0
SAMPLE_PERIOD_SECONDS = 0.1
WHEEL_TARGET_DEG_S = 120.0
LOG_COLUMNS = ["elapsed_s", "x_m", "y_m", "yaw_rad", "speed_mps", "command_wheel_deg_s"]


def wheel_joint_paths(names: list[str]) -> list[str]:
    """Return only the two Nova Carter drive-wheel joint names."""
    wheels = [name for name in names if "joint_wheel" in name.lower()]
    if len(wheels) != 2:
        raise RuntimeError(f"Expected exactly two drive-wheel joints, found: {wheels}")
    return wheels


def csv_row(
    elapsed_s: float,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    speed_mps: float,
    command: float,
) -> dict[str, float]:
    """Build one stable, spreadsheet-friendly motion-log row."""
    return {
        "elapsed_s": elapsed_s,
        "x_m": x_m,
        "y_m": y_m,
        "yaw_rad": yaw_rad,
        "speed_mps": speed_mps,
        "command_wheel_deg_s": command,
    }


def motion_prim_path(paths: list[str]) -> str:
    """Select Nova Carter's moving chassis from composed stage paths."""
    chassis_paths = [path for path in paths if path.rsplit("/", 1)[-1] == "chassis_link"]
    if len(chassis_paths) != 1:
        raise RuntimeError(f"Expected one chassis_link prim, found: {chassis_paths}")
    return chassis_paths[0]


def _require_runtime():
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    return omni, Usd, UsdGeom, UsdPhysics


def _wheel_drive_attributes(stage, usd_physics):
    joint_prims = [prim for prim in stage.Traverse() if "joint_wheel" in prim.GetName().lower()]
    joint_names = wheel_joint_paths([prim.GetName() for prim in joint_prims])
    joint_by_name = {prim.GetName(): prim for prim in joint_prims}
    attributes = []
    for name in joint_names:
        drive = usd_physics.DriveAPI.Get(joint_by_name[name], "angular")
        attribute = drive.GetTargetVelocityAttr()
        if not attribute or not attribute.IsValid():
            raise RuntimeError(f"Missing angular target-velocity drive on {name}")
        attributes.append(attribute)
    return attributes


def _set_wheel_targets(attributes, target_deg_s: float) -> None:
    for attribute in attributes:
        attribute.Set(target_deg_s)


def _chassis_prim(stage, robot_path: str):
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError(f"Robot root was not found at {robot_path}")

    stage_paths = [str(prim.GetPath()) for prim in stage.Traverse()]
    chassis_path = motion_prim_path(
        [path for path in stage_paths if path.startswith(f"{robot_path}/")]
    )
    chassis_prim = stage.GetPrimAtPath(chassis_path)
    if not chassis_prim or not chassis_prim.IsValid():
        raise RuntimeError(f"Chassis prim was not found at {chassis_path}")
    return chassis_prim


def _pose(chassis_prim, usd, usd_geom):
    matrix = usd_geom.Xformable(chassis_prim).ComputeLocalToWorldTransform(usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    real = quaternion.GetReal()
    imaginary = quaternion.GetImaginary()
    yaw = math.atan2(
        2.0 * (real * imaginary[2] + imaginary[0] * imaginary[1]),
        1.0 - 2.0 * (imaginary[1] ** 2 + imaginary[2] ** 2),
    )
    return float(translation[0]), float(translation[1]), float(yaw)


def _write_log(rows: list[dict[str, float]]) -> Path:
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = LOG_DIRECTORY / f"nova_carter_fixed_run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


async def run_experiment() -> Path:
    """Drive, sample pose, stop safely, and write a CSV log."""
    omni, usd, usd_geom, usd_physics = _require_runtime()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open")

    drives = _wheel_drive_attributes(stage, usd_physics)
    chassis_prim = _chassis_prim(stage, ROBOT_PATH)
    timeline = omni.timeline.get_timeline_interface()
    app = omni.kit.app.get_app()
    rows: list[dict[str, float]] = []
    previous_sample = None

    timeline.play()
    try:
        settle_end = time.monotonic() + SETTLE_SECONDS
        _set_wheel_targets(drives, 0.0)
        while time.monotonic() < settle_end:
            await app.next_update_async()

        drive_started = time.monotonic()
        next_sample = drive_started
        _set_wheel_targets(drives, WHEEL_TARGET_DEG_S)
        while time.monotonic() - drive_started < DRIVE_SECONDS:
            await app.next_update_async()
            now = time.monotonic()
            if now < next_sample:
                continue

            elapsed = now - drive_started
            x_m, y_m, yaw_rad = _pose(chassis_prim, usd, usd_geom)
            speed_mps = 0.0
            if previous_sample is not None:
                previous_elapsed, previous_x, previous_y = previous_sample
                dt = elapsed - previous_elapsed
                if dt > 0.0:
                    speed_mps = math.hypot(x_m - previous_x, y_m - previous_y) / dt
            rows.append(csv_row(elapsed, x_m, y_m, yaw_rad, speed_mps, WHEEL_TARGET_DEG_S))
            previous_sample = (elapsed, x_m, y_m)
            next_sample += SAMPLE_PERIOD_SECONDS
    finally:
        _set_wheel_targets(drives, 0.0)

    log_path = _write_log(rows)
    print(f"Fixed run complete: {log_path}")
    return log_path


def start_experiment():
    """Schedule the experiment without blocking the Isaac Sim interface."""
    task = asyncio.ensure_future(run_experiment())
    print("Nova Carter fixed run started; the robot will settle for 1 second, then drive for 5 seconds.")
    return task
