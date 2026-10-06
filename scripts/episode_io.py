"""Simulator-agnostic episode format for traversability data collection.

Every collector (Isaac Sim, the MuJoCo proxy, later a real Jackal
bag converter) writes the same directory layout, so labelling and training
never depend on which simulator produced the data::

    <episode_dir>/
      scenario.json   immutable scenario manifest that was spawned
      meta.json       episode id, condition, camera model, rates, termination
      state.csv       high-rate robot state, command, IMU and hazard contact
      frames.csv      one row per camera frame with the pose at capture time
      rgb/000123.png  camera images referenced by frames.csv

Coordinates: world frame is the scenario frame (x, y on the floor, z up).
The robot base frame is x forward, y left, z up. Camera pixels use the
OpenCV convention (u right, v down).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


FORMAT_VERSION = "1.0"

STATE_COLUMNS = (
    "t_s",
    "x_m",
    "y_m",
    "z_m",
    "yaw_rad",
    "pitch_rad",
    "roll_rad",
    "v_fwd_mps",
    "yaw_rate_rps",
    "cmd_v_mps",
    "cmd_w_rps",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "imu_gx",
    "imu_gy",
    "imu_gz",
    "hazard_contact_n",
    "contact_object",
)
FRAME_COLUMNS = ("frame_idx", "t_s", "x_m", "y_m", "yaw_rad", "rgb_path")
_STRING_COLUMNS = {"contact_object", "rgb_path"}


def camera_model(width: int, height: int, horizontal_fov_deg: float, mount_xyz_m, pitch_down_deg: float, rate_hz: float) -> dict:
    """Pinhole camera description stored in ``meta.json``."""
    fx = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
    return {
        "width": width,
        "height": height,
        "fx": round(fx, 4),
        "fy": round(fx, 4),
        "cx": width / 2.0,
        "cy": height / 2.0,
        "horizontal_fov_deg": horizontal_fov_deg,
        "mount_xyz_m": [float(v) for v in mount_xyz_m],
        "pitch_down_deg": pitch_down_deg,
        "rate_hz": rate_hz,
    }


def world_to_base(robot_pose: dict, x_m: float, y_m: float) -> tuple[float, float]:
    """Express a world floor point in the robot base frame (forward, left)."""
    dx, dy = x_m - robot_pose["x_m"], y_m - robot_pose["y_m"]
    cos_yaw, sin_yaw = math.cos(robot_pose["yaw_rad"]), math.sin(robot_pose["yaw_rad"])
    return cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy


def project_base_point(camera: dict, forward_m: float, left_m: float, up_m: float = 0.0):
    """Project a base-frame point into pixels; ``None`` if behind the camera."""
    mx, my, mz = camera["mount_xyz_m"]
    rx, ry, rz = forward_m - mx, left_m - my, up_m - mz
    pitch = math.radians(camera["pitch_down_deg"])
    z_cam = rx * math.cos(pitch) - rz * math.sin(pitch)
    if z_cam <= 1e-6:
        return None
    x_cam = -ry
    y_cam = -rx * math.sin(pitch) - rz * math.cos(pitch)
    return (
        camera["fx"] * x_cam / z_cam + camera["cx"],
        camera["fy"] * y_cam / z_cam + camera["cy"],
    )


def footprint_bbox(camera: dict, robot_pose: dict, object_item: dict, samples: int = 16):
    """Pixel bounding box of an object's floor footprint circle, clipped to the image.

    Returns ``None`` when no part of the footprint is in front of the camera
    and inside the image.
    """
    pose, radius = object_item["pose"], object_item["footprint_radius_m"]
    height = max(0.0, float(object_item.get("height_m", 0.0)))
    pixels = []
    for index in range(samples):
        angle = 2.0 * math.pi * index / samples
        forward, left = world_to_base(
            robot_pose, pose["x_m"] + radius * math.cos(angle), pose["y_m"] + radius * math.sin(angle)
        )
        for up in (0.0, height):
            pixel = project_base_point(camera, forward, left, up)
            if pixel is not None:
                pixels.append(pixel)
    if not pixels:
        return None
    us, vs = [p[0] for p in pixels], [p[1] for p in pixels]
    u0, u1 = max(0.0, min(us)), min(float(camera["width"]), max(us))
    v0, v1 = max(0.0, min(vs)), min(float(camera["height"]), max(vs))
    if u0 >= u1 or v0 >= v1:
        return None
    return [round(u0, 1), round(v0, 1), round(u1, 1), round(v1, 1)]


def _write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return [
            {key: (value if key in _STRING_COLUMNS else float(value)) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def write_episode(episode_dir, *, scenario: dict, meta: dict, state_rows, frame_rows) -> Path:
    """Write the non-image parts of an episode. Images are written by the collector."""
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
    (episode_dir / "meta.json").write_text(json.dumps({"format_version": FORMAT_VERSION, **meta}, indent=2) + "\n")
    _write_csv(episode_dir / "state.csv", STATE_COLUMNS, state_rows)
    _write_csv(episode_dir / "frames.csv", FRAME_COLUMNS, frame_rows)
    return episode_dir


def load_episode(episode_dir) -> dict:
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / "meta.json").read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported episode format {meta.get('format_version')!r} in {episode_dir}")
    frames = _read_csv(episode_dir / "frames.csv")
    for frame in frames:
        frame["frame_idx"] = int(frame["frame_idx"])
    return {
        "dir": episode_dir,
        "meta": meta,
        "scenario": json.loads((episode_dir / "scenario.json").read_text()),
        "state": _read_csv(episode_dir / "state.csv"),
        "frames": frames,
    }
