"""Convert a real Jackal + RealSense D435i rosbag into the episode format.

The labeller (``frame_labels_v2``) and the trainers only read episodes, so a
real drive goes through exactly the same pipeline as a simulated one::

    python3 -m scripts.rosbag_to_episode BAG --out data/real/ep-clean-001 \\
        --condition clean --camera-mount 0.20 0 0.30 --camera-pitch-deg 15
    python3 -m scripts.rosbag_to_episode BAG --out data/real/ep-cable-001 \\
        --condition hazard --hazard cable:2.5:0.05:0.03 ...

Reads ROS 1 (.bag) and ROS 2 (.db3 / .mcap) bags with the pure-Python
``rosbags`` library, so no ROS install is needed on the analysis machine.

What it does:

* Robot state is resampled onto a uniform 100 Hz grid, the sim's state rate.
  Pose and velocities are interpolated; the speed command is held (and
  treated as zero once it is older than ``--cmd-timeout-s``, like the robot's
  own command watchdog). IMU samples are averaged in 10 ms bins, which is how
  the simulated IMU was produced, so vibration levels mean the same thing.
* The IMU may be the Jackal's (already in the base frame) or the D435i's
  (``--imu-frame optical``); optical readings are rotated into the base frame
  using the camera pitch.
* Camera frames are kept at ``--frame-hz`` (10 Hz, as in the sim) and saved as
  JPEGs. The camera model comes from ``camera_info`` plus the measured mount.
* There is no planned route on a teleoperated drive, so the route is the
  straight line from where the robot started moving along its initial heading.
  Drive along a taped straight line.
* Hazard positions measured with a tape (distance along the line) can be given
  with ``--hazard``; they are used only for the truth check, never for labels.
* Motor feedback (Clearpath ``platform/motors/feedback``,
  clearpath_platform_msgs/Feedback, 50 Hz), when recorded, fills the optional
  ``motor_current_{l,r}_a`` and ``motor_vel_{l,r}_rps`` columns (driver 0 =
  left, 1 = right). The type is decoded from the definition embedded in the
  bag (MCAP and rosbag2 v8 store it); for older bags the bundled copy of
  Clearpath's .msg files is used.

Known limitation: if ``--odom`` is the wheel-based odometry, ground speed and
wheel speed are the same signal and slip is invisible. Use a ground-speed
source that does not come from the wheels for slip (e.g. visual odometry).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

from scripts.episode_io import write_episode

STATE_HZ = 100.0

# Candidate topic names, first match wins; a candidate also matches any topic
# that ends with it. ROS 1 Jackal (Noetic), ROS 2 Jackal (clearpath_robot,
# namespaced e.g. /j100_0434/..., camera under sensors/camera_0/color/...),
# realsense2_camera (/camera/camera/color/...). Colour only: depth never matches.
TOPIC_CANDIDATES = {
    "odom": ("/odometry/filtered", "/platform/odom/filtered"),
    "wheel_odom": ("/jackal_velocity_controller/odom", "/platform/odom"),
    "cmd": ("/jackal_velocity_controller/cmd_vel", "/platform/cmd_vel", "/cmd_vel"),
    "imu": ("/imu/data", "/sensors/imu_0/data", "/camera/imu", "/camera/camera/imu"),
    "image": ("/color/image_raw/compressed", "/color/image/compressed", "/color/compressed",
              "/color/image_raw", "/color/image"),
    "camera_info": ("/color/camera_info",),
    "motors": ("/platform/motors/feedback",),
}

# clearpath_platform_msgs (github.com/clearpathrobotics/clearpath_msgs, branch jazzy), comments removed.
# Only used when a bag does not carry its own message definitions.
CLEARPATH_MSGS = {
    "clearpath_platform_msgs/msg/DriveFeedback": """
float32 current
float32 duty_cycle
float32 bridge_temperature
float32 motor_temperature
float32 measured_velocity
float32 measured_travel
bool driver_fault
""",
    "clearpath_platform_msgs/msg/Feedback": """
std_msgs/Header header
DriveFeedback[2] drivers
int8 commanded_mode
int8 actual_mode
""",
}


def clearpath_typestore():
    """ROS 2 Jazzy typestore that also knows Clearpath's motor feedback types."""
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    store = get_typestore(Stores.ROS2_JAZZY)
    types = {}
    for name, text in CLEARPATH_MSGS.items():
        types.update(get_types_from_msg(text, name))
    store.register(types)
    return store


def pick_topic(available: list[str], role: str, override: str | None) -> str | None:
    if override:
        if override not in available:
            raise SystemExit(f"--{role.replace('_', '-')} {override!r} is not in the bag. Topics: {sorted(available)}")
        return override
    for candidate in TOPIC_CANDIDATES[role]:
        exact = [t for t in available if t == candidate]
        namespaced = [t for t in available if t.endswith(candidate)]
        if exact or namespaced:
            chosen = (exact or namespaced)[0]
            if not exact and len(namespaced) > 1:
                print(f"note: several {role} topics end with {candidate!r}; using {chosen} "
                      f"(pass --{role.replace('_', '-')} to choose)", file=sys.stderr)
            return chosen
    return None


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def optical_to_base(pitch_down_deg: float) -> np.ndarray:
    """Rotation taking camera-optical vectors (x right, y down, z forward) to base (x fwd, y left, z up)."""
    p = math.radians(pitch_down_deg)
    return np.array([[0.0, -math.sin(p), math.cos(p)],
                     [-1.0, 0.0, 0.0],
                     [0.0, -math.cos(p), -math.sin(p)]])


def stamp_s(msg, bag_time_ns: int) -> float:
    header = getattr(msg, "header", None)
    if header is not None and (header.stamp.sec or header.stamp.nanosec):
        return header.stamp.sec + header.stamp.nanosec * 1e-9
    return bag_time_ns * 1e-9


def decode_image(msg, msgtype: str):
    """Return a PIL RGB image from sensor_msgs/Image or CompressedImage."""
    from PIL import Image

    if msgtype.endswith("CompressedImage"):
        return Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    enc = msg.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(enc)
    if channels is None:
        raise SystemExit(f"Unsupported image encoding {msg.encoding!r}")
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * channels]
    data = data.reshape(msg.height, msg.width, channels)
    if enc in ("bgr8", "bgra8"):
        data = data[..., [2, 1, 0]]
    elif channels == 1:
        data = np.repeat(data, 3, axis=2)
    return Image.fromarray(np.ascontiguousarray(data[..., :3]))


def read_bag(bag: Path, topics: dict) -> dict:
    from rosbags.highlevel import AnyReader

    out = {role: [] for role in topics}
    with AnyReader([bag], default_typestore=clearpath_typestore()) as reader:
        wanted = {t: role for role, t in topics.items() if t}
        connections = [c for c in reader.connections if c.topic in wanted]
        for connection, bag_time, raw in reader.messages(connections=connections):
            role = wanted[connection.topic]
            msg = reader.deserialize(raw, connection.msgtype)
            out[role].append((stamp_s(msg, bag_time), msg, connection.msgtype))
    for role in out:
        out[role].sort(key=lambda item: item[0])
    return out


def list_topics(bag: Path) -> list[str]:
    from rosbags.highlevel import AnyReader

    with AnyReader([bag]) as reader:
        return sorted({c.topic for c in reader.connections})


def _interp(grid, t, values):
    return np.interp(grid, t, values) if len(t) else np.full(len(grid), np.nan)


def resample_state(msgs: dict, *, imu_frame: str, pitch_down_deg: float, cmd_timeout_s: float):
    odom = msgs["odom"]
    if len(odom) < 2:
        raise SystemExit("Need odometry messages to place the robot on the floor.")
    t_odom = np.array([m[0] for m in odom])
    t0, t1 = t_odom[0], t_odom[-1]
    imu = msgs.get("imu", [])
    if imu:
        t0, t1 = max(t0, imu[0][0]), min(t1, imu[-1][0])
    grid = np.arange(t0, t1, 1.0 / STATE_HZ)

    pose = lambda m: m[1].pose.pose
    x = _interp(grid, t_odom, [pose(m).position.x for m in odom])
    y = _interp(grid, t_odom, [pose(m).position.y for m in odom])
    z = _interp(grid, t_odom, [pose(m).position.z for m in odom])
    yaw = _interp(grid, t_odom, np.unwrap([yaw_from_quaternion(pose(m).orientation) for m in odom]))
    v = _interp(grid, t_odom, [m[1].twist.twist.linear.x for m in odom])
    w = _interp(grid, t_odom, [m[1].twist.twist.angular.z for m in odom])

    wheel = msgs.get("wheel_odom", [])
    wheel_v = _interp(grid, np.array([m[0] for m in wheel]), [m[1].twist.twist.linear.x for m in wheel]) if wheel else np.full(len(grid), np.nan)

    motor = {name: np.full(len(grid), np.nan) for name in ("motor_current_l_a", "motor_current_r_a", "motor_vel_l_rps", "motor_vel_r_rps")}
    motors = msgs.get("motors", [])
    if motors:
        t_motor = np.array([m[0] for m in motors])
        for side, index in (("l", 0), ("r", 1)):
            motor[f"motor_current_{side}_a"] = _interp(grid, t_motor, [m[1].drivers[index].current for m in motors])
            motor[f"motor_vel_{side}_rps"] = _interp(grid, t_motor, [m[1].drivers[index].measured_velocity for m in motors])

    cmd_v, cmd_w = np.zeros(len(grid)), np.zeros(len(grid))
    cmd = msgs.get("cmd", [])
    if cmd:
        t_cmd = np.array([m[0] for m in cmd])
        twist = [m[1].twist if m[2].endswith("TwistStamped") else m[1] for m in cmd]
        idx = np.searchsorted(t_cmd, grid, side="right") - 1
        fresh = (idx >= 0) & (grid - t_cmd[np.clip(idx, 0, None)] <= cmd_timeout_s)
        cmd_v = np.where(fresh, np.array([tw.linear.x for tw in twist])[np.clip(idx, 0, None)], 0.0)
        cmd_w = np.where(fresh, np.array([tw.angular.z for tw in twist])[np.clip(idx, 0, None)], 0.0)

    acc = np.full((len(grid), 3), np.nan)
    gyr = np.full((len(grid), 3), np.nan)
    if imu:
        t_imu = np.array([m[0] for m in imu])
        a = np.array([[m[1].linear_acceleration.x, m[1].linear_acceleration.y, m[1].linear_acceleration.z] for m in imu])
        g = np.array([[m[1].angular_velocity.x, m[1].angular_velocity.y, m[1].angular_velocity.z] for m in imu])
        if imu_frame == "optical":
            rot = optical_to_base(pitch_down_deg)
            a, g = a @ rot.T, g @ rot.T
        # Average all samples in each 10 ms bin (the sim IMU did the same);
        # bins with no sample (IMU slower than 100 Hz) are interpolated.
        bins = np.floor((t_imu - grid[0]) * STATE_HZ).astype(int)
        inside = (bins >= 0) & (bins < len(grid))
        counts = np.bincount(bins[inside], minlength=len(grid)).astype(float)
        filled = counts > 0
        for k in range(3):
            acc[:, k] = np.bincount(bins[inside], a[inside, k], minlength=len(grid)) / np.maximum(counts, 1)
            gyr[:, k] = np.bincount(bins[inside], g[inside, k], minlength=len(grid)) / np.maximum(counts, 1)
            if not filled.all():
                acc[~filled, k] = np.interp(grid[~filled], grid[filled], acc[filled, k])
                gyr[~filled, k] = np.interp(grid[~filled], grid[filled], gyr[filled, k])

    rows = []
    for i, t in enumerate(grid):
        rows.append({
            "t_s": round(float(t - grid[0]), 4), "x_m": float(x[i]), "y_m": float(y[i]), "z_m": float(z[i]),
            "yaw_rad": float(yaw[i]), "pitch_rad": "", "roll_rad": "", "v_fwd_mps": float(v[i]),
            "yaw_rate_rps": float(w[i]), "cmd_v_mps": float(cmd_v[i]), "cmd_w_rps": float(cmd_w[i]),
            "imu_ax": float(acc[i, 0]), "imu_ay": float(acc[i, 1]), "imu_az": float(acc[i, 2]),
            "imu_gx": float(gyr[i, 0]), "imu_gy": float(gyr[i, 1]), "imu_gz": float(gyr[i, 2]),
            "hazard_contact_n": "", "contact_object": "",
            "wheel_v_mps": "" if np.isnan(wheel_v[i]) else float(wheel_v[i]),
            **{name: "" if np.isnan(values[i]) else float(values[i]) for name, values in motor.items()},
        })
    return rows, float(grid[0])


def straight_route(rows: list[dict], start_speed_mps: float = 0.05, heading_m: float = 0.5) -> dict:
    """Route = straight line from where the robot started moving, along its first ``heading_m`` of travel."""
    moving = next((i for i, r in enumerate(rows) if abs(r["v_fwd_mps"]) > start_speed_mps), 0)
    sx, sy = rows[moving]["x_m"], rows[moving]["y_m"]
    ahead = next((r for r in rows[moving:] if math.hypot(r["x_m"] - sx, r["y_m"] - sy) >= heading_m), rows[-1])
    heading = math.atan2(ahead["y_m"] - sy, ahead["x_m"] - sx)
    end = rows[-1]
    length = (end["x_m"] - sx) * math.cos(heading) + (end["y_m"] - sy) * math.sin(heading)
    return {
        "start_pose": {"x_m": sx, "y_m": sy, "yaw_rad": heading},
        "goal_pose": {"x_m": sx + length * math.cos(heading), "y_m": sy + length * math.sin(heading)},
        "length_m": round(length, 3),
    }


def parse_hazard(text: str, route: dict, index: int) -> dict:
    """``kind:distance_m[:radius_m[:height_m]]`` measured along the taped line from the start."""
    parts = text.split(":")
    kind, distance = parts[0], float(parts[1])
    radius = float(parts[2]) if len(parts) > 2 else 0.05
    height = float(parts[3]) if len(parts) > 3 else 0.0
    start, heading = route["start_pose"], route["start_pose"]["yaw_rad"]
    return {"instance_id": f"{kind}-{index}", "asset_key": kind, "role": "target_hazard",
            "pose": {"x_m": start["x_m"] + distance * math.cos(heading), "y_m": start["y_m"] + distance * math.sin(heading)},
            "footprint_radius_m": radius, "height_m": height, "measured_distance_m": distance}


def camera_from_info(info, mount_xyz_m, pitch_down_deg: float, rate_hz: float, scale: float) -> dict:
    k = list(info.k if hasattr(info, "k") else info.K)
    width, height = int(round(info.width * scale)), int(round(info.height * scale))
    fx, fy, cx, cy = k[0] * scale, k[4] * scale, k[2] * scale, k[5] * scale
    return {"width": width, "height": height, "fx": round(fx, 4), "fy": round(fy, 4), "cx": round(cx, 3), "cy": round(cy, 3),
            "horizontal_fov_deg": round(math.degrees(2 * math.atan(width / (2 * fx))), 2),
            "mount_xyz_m": [float(v) for v in mount_xyz_m], "pitch_down_deg": pitch_down_deg, "rate_hz": rate_hz,
            "source": "camera_info"}


def nearest_visible_floor_m(camera: dict) -> float:
    """Distance ahead of the robot centre of the floor seen at the bottom image row."""
    below = math.radians(camera["pitch_down_deg"]) + math.atan((camera["height"] - camera["cy"]) / camera["fy"])
    return camera["mount_xyz_m"][0] + camera["mount_xyz_m"][2] / math.tan(below) if below > 0 else math.inf


def convert(bag: Path, out: Path, *, episode_id: str, condition: str, mount_xyz_m, pitch_down_deg: float,
            imu_frame: str, frame_hz: float, image_scale: float, hazards: list[str], termination: str,
            cmd_timeout_s: float, overrides: dict, rotate_180: bool = False) -> dict:
    available = list_topics(bag)
    topics = {role: pick_topic(available, role, overrides.get(role)) for role in TOPIC_CANDIDATES}
    for role in ("odom", "imu", "image", "camera_info"):
        if topics[role] is None:
            raise SystemExit(f"No {role} topic found; pass --{role.replace('_', '-')}. Topics: {available}")
    if topics["cmd"] is None:
        print("warning: no velocity command topic; every sample will look idle to the labeller")
    if topics["motors"] is None:
        print("note: no motor feedback topic; the motor current columns stay empty")
    msgs = read_bag(bag, topics)
    rows, t0 = resample_state(msgs, imu_frame=imu_frame, pitch_down_deg=pitch_down_deg, cmd_timeout_s=cmd_timeout_s)
    route = straight_route(rows)

    camera = camera_from_info(msgs["camera_info"][0][1], mount_xyz_m, pitch_down_deg, frame_hz, image_scale)
    if rotate_180:
        # Camera mounted upside down: photos are saved turned upright, so the principal point mirrors.
        camera["cx"] = round(camera["width"] - camera["cx"], 3)
        camera["cy"] = round(camera["height"] - camera["cy"], 3)
        camera["rotated_180"] = True
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    t_grid = np.array([r["t_s"] for r in rows])
    frames, next_t = [], 0.0
    for stamp, msg, msgtype in msgs["image"]:
        t = stamp - t0
        if t < 0 or t > t_grid[-1] or t + 1e-6 < next_t:
            continue
        i = int(np.clip(np.searchsorted(t_grid, t), 0, len(rows) - 1))
        image = decode_image(msg, msgtype)
        if image_scale != 1.0:
            image = image.resize((camera["width"], camera["height"]))
        if rotate_180:
            image = image.rotate(180)
        name = f"rgb/{len(frames):06d}.jpg"
        image.save(out / name, quality=92)
        frames.append({"frame_idx": len(frames), "t_s": round(t, 4), "x_m": rows[i]["x_m"], "y_m": rows[i]["y_m"],
                       "yaw_rad": rows[i]["yaw_rad"], "rgb_path": name})
        next_t = t + 1.0 / frame_hz

    scenario = {"scenario_id": episode_id, "seed": -1, "source": "real", "route": route,
                "objects": [parse_hazard(h, route, i) for i, h in enumerate(hazards)]}
    meta = {"episode_id": episode_id, "condition": condition, "robot": "jackal", "camera": camera,
            "rates": {"state_hz": STATE_HZ, "camera_hz": frame_hz}, "termination": {"reason": termination},
            "imu_frame": imu_frame, "nearest_visible_floor_m": round(nearest_visible_floor_m(camera), 3),
            "source": {"bag": str(bag), "topics": topics, "start_stamp_s": t0}}
    write_episode(out, scenario=scenario, meta=meta, state_rows=rows, frame_rows=frames)
    return {"episode": str(out), "state_rows": len(rows), "frames": len(frames), "route_length_m": route["length_m"],
            "nearest_visible_floor_m": meta["nearest_visible_floor_m"], "topics": topics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bag", type=Path, help="ROS 1 .bag file or ROS 2 bag directory")
    parser.add_argument("--out", type=Path, help="episode directory to create (required unless --list-topics)")
    parser.add_argument("--episode-id", help="default: the output directory name")
    parser.add_argument("--condition", choices=("clean", "hazard"))
    parser.add_argument("--camera-mount", type=float, nargs=3, metavar=("FWD_M", "LEFT_M", "UP_M"),
                        help="camera position relative to the robot centre on the floor (measure it)")
    parser.add_argument("--camera-pitch-deg", type=float, help="downward tilt of the camera")
    parser.add_argument("--imu-frame", choices=("base", "optical"), default="base",
                        help="'base' for the Jackal IMU, 'optical' for the D435i IMU")
    parser.add_argument("--frame-hz", type=float, default=10.0)
    parser.add_argument("--rotate-180", action="store_true", help="camera is mounted upside down: save photos upright")
    parser.add_argument("--image-scale", type=float, default=1.0, help="e.g. 0.5 to halve the saved resolution")
    parser.add_argument("--hazard", action="append", default=[], metavar="KIND:DIST_M[:RADIUS_M[:HEIGHT_M]]",
                        help="measured hazard position along the taped line (truth check only)")
    parser.add_argument("--termination", default="goal_reached", choices=("goal_reached", "stalled", "timeout"))
    parser.add_argument("--cmd-timeout-s", type=float, default=0.5)
    for role in TOPIC_CANDIDATES:
        parser.add_argument(f"--{role.replace('_', '-')}", dest=f"topic_{role}", help=f"override the {role} topic")
    parser.add_argument("--list-topics", action="store_true", help="print the bag's topics and exit")
    args = parser.parse_args()
    if args.list_topics:
        print("\n".join(list_topics(args.bag)))
        return
    missing = [flag for flag, value in (("--out", args.out), ("--condition", args.condition),
                                        ("--camera-mount", args.camera_mount), ("--camera-pitch-deg", args.camera_pitch_deg))
               if value is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    result = convert(args.bag, args.out, episode_id=args.episode_id or args.out.name, condition=args.condition,
                     mount_xyz_m=args.camera_mount, pitch_down_deg=args.camera_pitch_deg, imu_frame=args.imu_frame,
                     frame_hz=args.frame_hz, image_scale=args.image_scale, hazards=args.hazard, rotate_180=args.rotate_180,
                     termination=args.termination, cmd_timeout_s=args.cmd_timeout_s,
                     overrides={role: getattr(args, f"topic_{role}") for role in TOPIC_CANDIDATES})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
