"""MuJoCo proxy collector: runs a scenario manifest and writes an episode.

This is a *pipeline-validation* simulator, not a replacement for Isaac Sim.
It consumes exactly the same scenario manifests as the Isaac spawner and
writes exactly the same episode format as the Isaac collector, so the
labelling and training code can be developed and tested end to end while
An Isaac Sim collector is in isaac/.

What is modelled:
  * a Nova-Carter-sized differential-drive robot (two driven wheels, two
    trailing ball casters, torque-limited velocity control);
  * a front camera 0.35 m above the floor pitched 15 degrees down;
  * a chassis IMU (accelerometer + gyro, averaged over each 10 ms sample
    like an IMU anti-aliasing filter), pose/velocity, commands;
  * procedural hazards from the manifest: curved cable (capsule chain),
    thin plastic wrapper, low-friction oil patch, raised lip, cardboard boxes;
  * per-seed appearance randomisation (floor tone, lighting), identical for
    the clean and hazard runs of a seed so they form a paired control.

What is *not* claimed: realistic rendering, validated contact parameters, or
the real Nova Carter / Jackal dynamics. Outcomes here validate the pipeline;
they are not evidence about real robots.

Usage::

    python3 -m scripts.proxy_sim --seeds 0-99 --output data/episodes
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path

import numpy as np

from scripts.episode_io import camera_model, write_episode
from scripts.isaac_scenario_spawner import cable_geometry
from scripts.route_follower import CRUISE_SPEED_MPS, RAMP_S, Route, TerminationMonitor, route_command
from scripts.starter_pipeline import create_warehouse_scenario


# Robot geometry (approximate, Nova-Carter-sized). Base frame origin is the
# chassis centre projected to the floor.
WHEEL_RADIUS_M = 0.14
WHEEL_HALF_TRACK_M = 0.25
WHEEL_X_M = 0.10
CASTER_RADIUS_M = 0.045
CASTER_X_M = -0.24
CASTER_HALF_TRACK_M = 0.17
CHASSIS_HALF_SIZE_M = (0.30, 0.22, 0.09)
CHASSIS_Z_M = 0.23
CHASSIS_MASS_KG = 40.0
WHEEL_TORQUE_LIMIT_NM = 18.0

CAMERA = camera_model(320, 240, 90.0, (0.31, 0.0, 0.35), 15.0, 10.0)
PHYSICS_DT_S = 0.002
CONTROL_HZ = 50
STATE_HZ = 100
FLOOR_FRICTION = 1.0
OIL_FRICTION = 0.08
# White IMU noise at the 100 Hz output, of the order of a D455's BMI055.
ACC_NOISE_MPS2 = 0.02
GYRO_NOISE_RPS = 0.002


def _fmt(values) -> str:
    return " ".join(f"{v:.5g}" for v in values)


APPEARANCE_REGIMES = ("default", "shifted")


def _appearance(seed: int, regime: str = "default") -> dict:
    """Per-seed look. ``shifted`` is a held-out appearance regime for stress
    tests: darker warm lighting, brownish floor, non-black cables and
    floor-coloured lips. Physics is identical between regimes."""
    if regime == "shifted":
        rng = random.Random(20_000 + seed)
        tone = rng.uniform(0.28, 0.4)
        return {
            "floor_rgb1": (tone * 1.15, tone, tone * 0.8),
            "floor_rgb2": (tone * 0.85, tone * 0.75, tone * 0.6),
            "light_dir": (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1.0),
            "light_diffuse": rng.uniform(0.35, 0.55),
            "ambient": rng.uniform(0.15, 0.25),
            "box_tint": rng.uniform(0.7, 1.2),
            "cable_rgb": rng.choice(((0.55, 0.55, 0.55), (0.9, 0.75, 0.1), (0.15, 0.3, 0.7))),
            "lip_rgb": (tone * 1.2, tone * 1.1, tone),
        }
    rng = random.Random(10_000 + seed)
    grey = rng.uniform(0.42, 0.62)
    return {
        "cable_rgb": (0.07, 0.07, 0.07),
        "lip_rgb": (0.46, 0.46, 0.5),
        "floor_rgb1": (grey, grey * rng.uniform(0.97, 1.02), grey * rng.uniform(0.94, 1.0)),
        "floor_rgb2": (grey * 0.8, grey * 0.8, grey * 0.78),
        "light_dir": (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -1.0),
        "light_diffuse": rng.uniform(0.55, 0.85),
        "ambient": rng.uniform(0.25, 0.4),
        "box_tint": rng.uniform(0.85, 1.1),
    }


def _hazard_xml(item: dict, rng: random.Random, appearance: dict) -> str:
    """MJCF body for one manifest object. All geoms are named ``obj::<id>::n``."""
    pose, key, iid = item["pose"], item["asset_key"], item["instance_id"]
    head = f'<body name="obj::{iid}" pos="{pose["x_m"]:.4f} {pose["y_m"]:.4f} 0" euler="0 0 {pose["yaw_rad"]:.4f}">'
    geoms = []

    def geom(body: str) -> None:
        geoms.append(f'<geom name="obj::{iid}::{len(geoms)}" {body}/>')

    if key == "cable_hose":
        radius, centres = cable_geometry(item)
        for (x0, y0), (x1, y1) in zip(centres, centres[1:]):
            geom(f'type="capsule" size="{radius:.4f}" fromto="{x0:.4f} {y0:.4f} {radius:.4f} {x1:.4f} {y1:.4f} {radius:.4f}" '
                 f'rgba="{_fmt(appearance["cable_rgb"])} 1" friction="{FLOOR_FRICTION} 0.005 0.0001"')
    elif key == "plastic_wrapper":
        radius = item["footprint_radius_m"]
        for _ in range(4):
            sx, sy = radius * rng.uniform(0.35, 0.7), radius * rng.uniform(0.25, 0.55)
            geom(f'type="box" size="{sx:.4f} {sy:.4f} 0.0015" pos="{rng.uniform(-0.3, 0.3) * radius:.4f} {rng.uniform(-0.3, 0.3) * radius:.4f} {rng.uniform(0.003, 0.012):.4f}" '
                 f'euler="{rng.uniform(-0.35, 0.35):.3f} {rng.uniform(-0.35, 0.35):.3f} {rng.uniform(-math.pi, math.pi):.3f}" '
                 f'rgba="0.92 0.9 0.8 0.85" contype="0" conaffinity="0"')
        # Thick box sunk into the floor, proud by the manifest's effective
        # collision height (film flattens under the robot), plastic friction.
        proud = item.get("collision_height_m", 0.0005)
        geom(f'type="box" size="{radius:.4f} {radius * 0.7:.4f} 0.01" pos="0 0 {proud - 0.01:.4f}" rgba="0 0 0 0" group="3" '
             'priority="1" friction="0.5 0.005 0.0001"')
    elif key == "oil_spill":
        radius = item["footprint_radius_m"]
        for index in range(4):
            r = radius * (0.55 if index else 0.75) * rng.uniform(0.8, 1.1)
            off = (0.0, 0.0) if index == 0 else (rng.uniform(-0.4, 0.4) * radius, rng.uniform(-0.4, 0.4) * radius)
            geom(f'type="cylinder" size="{r:.4f} 0.0006" pos="{off[0]:.4f} {off[1]:.4f} 0.0006" material="oil" '
                 'contype="0" conaffinity="0"')
        # Friction collider: a thick box sunk into the floor with its top
        # 0.1 mm proud. Thin discs make wheel contacts unstable in MuJoCo.
        geom(f'type="box" size="{0.8 * radius:.4f} {0.8 * radius:.4f} 0.01" pos="0 0 -0.0099" rgba="0 0 0 0" group="3" '
             f'priority="1" friction="{OIL_FRICTION} 0.005 0.0001"')
    elif key == "small_lip":
        diameter = 2.0 * item["footprint_radius_m"]
        height = item["height_m"]
        geom(f'type="box" size="{diameter / 2:.4f} {diameter * 0.125:.4f} {height / 2:.4f}" pos="0 0 {height / 2:.4f}" rgba="{_fmt(appearance["lip_rgb"])} 1"')
    elif key == "cardboard_box":
        geom('type="box" size="0.22 0.16 0.175" pos="0 0 0.175" material="cardboard"')
    else:
        raise ValueError(f"No proxy geometry for {key!r}")
    return head + "".join(geoms) + "</body>"


def _racks_xml(start: dict) -> str:
    """Pallet racking on both sides of the aisle, for visual context only."""
    parts = []
    yaw = start["yaw_rad"]
    for side in (-1, 1):
        for bay in range(-1, 5):
            fwd, lat = bay * 2.0 + 1.0, side * 2.7
            x = start["x_m"] + fwd * math.cos(yaw) - lat * math.sin(yaw)
            y = start["y_m"] + fwd * math.sin(yaw) + lat * math.cos(yaw)
            parts.append(
                f'<body pos="{x:.3f} {y:.3f} 0" euler="0 0 {yaw:.4f}">'
                f'<geom type="box" size="0.05 0.05 1.5" pos="-0.95 -0.45 1.5" rgba="0.9 0.45 0.1 1"/>'
                f'<geom type="box" size="0.05 0.05 1.5" pos="-0.95 0.45 1.5" rgba="0.9 0.45 0.1 1"/>'
                f'<geom type="box" size="1.0 0.5 0.03" pos="0 0 0.15" rgba="0.2 0.35 0.7 1"/>'
                f'<geom type="box" size="1.0 0.5 0.03" pos="0 0 1.1" rgba="0.2 0.35 0.7 1"/>'
                f'<geom type="box" size="0.45 0.4 0.3" pos="-0.4 0 0.48" material="cardboard"/>'
                f'<geom type="box" size="0.4 0.38 0.25" pos="0.5 0 1.38" material="cardboard"/>'
                "</body>"
            )
    # Painted aisle lines.
    for lat in (-1.35, 1.35):
        x = start["x_m"] + 3.0 * math.cos(yaw) - lat * math.sin(yaw)
        y = start["y_m"] + 3.0 * math.sin(yaw) + lat * math.cos(yaw)
        parts.append(f'<geom type="box" size="6 0.04 0.0005" pos="{x:.3f} {y:.3f} 0.0005" euler="0 0 {yaw:.4f}" '
                     'rgba="0.95 0.8 0.1 1" contype="0" conaffinity="0"/>')
    return "".join(parts)


def build_mjcf(scenario: dict, appearance: dict) -> str:
    start = scenario["robot"]["start_pose"]
    rng = random.Random(scenario["seed"])
    objects = "".join(_hazard_xml(item, rng, appearance) for item in scenario["objects"])
    hx, hy, hz = CHASSIS_HALF_SIZE_M
    cam_x, _, cam_z = CAMERA["mount_xyz_m"]
    pitch = math.radians(CAMERA["pitch_down_deg"])
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(CAMERA["horizontal_fov_deg"]) / 2) * CAMERA["height"] / CAMERA["width"]))
    a = appearance
    return f"""
<mujoco model="warehouse_proxy">
  <compiler angle="radian"/>
  <option timestep="{PHYSICS_DT_S}" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <global offwidth="{CAMERA['width']}" offheight="{CAMERA['height']}"/>
    <headlight ambient="{a['ambient']:.3f} {a['ambient']:.3f} {a['ambient']:.3f}" diffuse="0.2 0.2 0.2" specular="0 0 0"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="flat" width="512" height="512" rgb1="{_fmt(a['floor_rgb1'])}" rgb2="{_fmt(a['floor_rgb2'])}" mark="random" random="0.08" markrgb="{_fmt(a['floor_rgb2'])}"/>
    <material name="floor" texture="floor" texrepeat="6 6" reflectance="0.05"/>
    <material name="oil" rgba="0.03 0.025 0.02 0.95" specular="0.9" shininess="0.9" reflectance="0.25"/>
    <material name="cardboard" rgba="{0.62 * a['box_tint']:.3f} {0.45 * a['box_tint']:.3f} {0.27 * a['box_tint']:.3f} 1"/>
  </asset>
  <default>
    <geom friction="{FLOOR_FRICTION} 0.005 0.0001"/>
  </default>
  <worldbody>
    <light directional="true" pos="0 0 6" dir="{_fmt(a['light_dir'])}" diffuse="{a['light_diffuse']:.3f} {a['light_diffuse']:.3f} {a['light_diffuse']:.3f}" castshadow="true"/>
    <light pos="{start['x_m'] + 3:.3f} {start['y_m']:.3f} 5" dir="0 0 -1" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="floor"/>
    {_racks_xml(start)}
    {objects}
    <body name="robot" pos="{start['x_m']:.4f} {start['y_m']:.4f} {CHASSIS_Z_M}" euler="0 0 {start['yaw_rad']:.4f}">
      <freejoint name="robot_root"/>
      <geom name="robot::chassis" type="box" size="{hx} {hy} {hz}" mass="{CHASSIS_MASS_KG}" rgba="0.15 0.5 0.3 1"/>
      <site name="imu" pos="0 0 0"/>
      <camera name="front" pos="{cam_x} 0 {cam_z - CHASSIS_Z_M:.4f}" xyaxes="0 -1 0 {math.sin(pitch):.5f} 0 {math.cos(pitch):.5f}" fovy="{fovy:.3f}"/>
      <body name="wheel_left" pos="{WHEEL_X_M} {WHEEL_HALF_TRACK_M} {WHEEL_RADIUS_M - CHASSIS_Z_M:.4f}">
        <joint name="wheel_left" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom name="robot::wheel_left" type="cylinder" size="{WHEEL_RADIUS_M} 0.025" euler="1.5708 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="wheel_right" pos="{WHEEL_X_M} {-WHEEL_HALF_TRACK_M} {WHEEL_RADIUS_M - CHASSIS_Z_M:.4f}">
        <joint name="wheel_right" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom name="robot::wheel_right" type="cylinder" size="{WHEEL_RADIUS_M} 0.025" euler="1.5708 0 0" mass="2" rgba="0.1 0.1 0.1 1"/>
      </body>
      <body name="caster_left" pos="{CASTER_X_M} {CASTER_HALF_TRACK_M} {CASTER_RADIUS_M - CHASSIS_Z_M:.4f}">
        <joint name="caster_left" type="ball" damping="0.002"/>
        <geom name="robot::caster_left" type="sphere" size="{CASTER_RADIUS_M}" mass="0.5" rgba="0.3 0.3 0.3 1"/>
      </body>
      <body name="caster_right" pos="{CASTER_X_M} {-CASTER_HALF_TRACK_M} {CASTER_RADIUS_M - CHASSIS_Z_M:.4f}">
        <joint name="caster_right" type="ball" damping="0.002"/>
        <geom name="robot::caster_right" type="sphere" size="{CASTER_RADIUS_M}" mass="0.5" rgba="0.3 0.3 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="left" joint="wheel_left" kv="60" forcerange="-{WHEEL_TORQUE_LIMIT_NM} {WHEEL_TORQUE_LIMIT_NM}" forcelimited="true"/>
    <velocity name="right" joint="wheel_right" kv="60" forcerange="-{WHEEL_TORQUE_LIMIT_NM} {WHEEL_TORQUE_LIMIT_NM}" forcelimited="true"/>
  </actuator>
  <sensor>
    <accelerometer name="acc" site="imu"/>
    <gyro name="gyro" site="imu"/>
  </sensor>
</mujoco>
"""


def _euler_from_quat(q) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw, pitch, roll


def clean_variant(scenario: dict) -> dict:
    """Same scene with every target hazard removed: the paired clean control."""
    clean = copy.deepcopy(scenario)
    clean["objects"] = [item for item in scenario["objects"] if item.get("role") != "target_hazard"]
    clean["removed_target_hazards"] = [item["instance_id"] for item in scenario["objects"] if item.get("role") == "target_hazard"]
    return clean


def run_episode(scenario: dict, condition: str, output_root: Path, seed: int, render: bool = True,
                appearance_regime: str = "default") -> Path:
    import mujoco
    from PIL import Image

    scene = clean_variant(scenario) if condition == "clean" else scenario
    episode_id = f"{scenario['scenario_id']}-{condition}"
    episode_dir = output_root / episode_id
    (episode_dir / "rgb").mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_string(build_mjcf(scene, _appearance(seed, appearance_regime)))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, CAMERA["height"], CAMERA["width"]) if render else None
    robot_body = model.body("robot").id
    dof = model.jnt_dofadr[model.joint("robot_root").id]
    geom_names = [model.geom(i).name for i in range(model.ngeom)]
    robot_geoms = {i for i, name in enumerate(geom_names) if name.startswith("robot::")}
    object_of_geom = {i: name.split("::")[1] for i, name in enumerate(geom_names) if name.startswith("obj::")}
    acc_adr, gyro_adr = model.sensor("acc").adr[0], model.sensor("gyro").adr[0]

    route = Route.from_manifest(scene)
    monitor = TerminationMonitor(route)

    steps_per_control = round(1.0 / (CONTROL_HZ * PHYSICS_DT_S))
    steps_per_state = round(1.0 / (STATE_HZ * PHYSICS_DT_S))
    steps_per_frame = round(1.0 / (CAMERA["rate_hz"] * PHYSICS_DT_S))
    state_rows, frame_rows = [], []
    cmd_v = cmd_w = 0.0
    force = np.zeros(6)
    imu_sum = np.zeros(6)
    imu_count = 0
    noise = np.random.default_rng(seed * 2 + (condition == "hazard"))
    step = 0
    wall_start = time.time()

    # Let the robot settle on its wheels before the episode clock starts.
    for _ in range(int(0.5 / PHYSICS_DT_S)):
        mujoco.mj_step(model, data)

    while True:
        t = step * PHYSICS_DT_S
        x, y = data.xpos[robot_body][:2]
        yaw, pitch, roll = _euler_from_quat(data.xquat[robot_body])
        v_fwd = float(data.qvel[dof] * math.cos(yaw) + data.qvel[dof + 1] * math.sin(yaw))
        yaw_rate = float(data.qvel[dof + 5])

        if step % steps_per_control == 0:
            cmd_v, cmd_w = route_command(route, t, float(x), float(y), yaw)
            left = (cmd_v - cmd_w * WHEEL_HALF_TRACK_M) / WHEEL_RADIUS_M
            right = (cmd_v + cmd_w * WHEEL_HALF_TRACK_M) / WHEEL_RADIUS_M
            data.ctrl[model.actuator("left").id] = left
            data.ctrl[model.actuator("right").id] = right

        if step % steps_per_state == 0:
            contact_n, contact_obj = 0.0, ""
            per_object: dict[str, float] = {}
            for index in range(data.ncon):
                contact = data.contact[index]
                pair = {contact.geom1, contact.geom2}
                other = next((g for g in pair if g in object_of_geom), None)
                if other is None or not pair & robot_geoms:
                    continue
                mujoco.mj_contactForce(model, data, index, force)
                per_object[object_of_geom[other]] = per_object.get(object_of_geom[other], 0.0) + abs(force[0])
            if per_object:
                contact_obj = max(per_object, key=per_object.get)
                contact_n = sum(per_object.values())
            imu_mean = imu_sum / max(imu_count, 1) if imu_count else np.concatenate(
                [data.sensordata[acc_adr:acc_adr + 3], data.sensordata[gyro_adr:gyro_adr + 3]])
            acc = imu_mean[:3] + noise.normal(0.0, ACC_NOISE_MPS2, 3)
            gyro = imu_mean[3:] + noise.normal(0.0, GYRO_NOISE_RPS, 3)
            imu_sum[:] = 0.0
            imu_count = 0
            state_rows.append({
                "t_s": round(t, 4), "x_m": round(float(x), 5), "y_m": round(float(y), 5),
                "z_m": round(float(data.xpos[robot_body][2]), 5), "yaw_rad": round(yaw, 5),
                "pitch_rad": round(pitch, 5), "roll_rad": round(roll, 5), "v_fwd_mps": round(v_fwd, 5),
                "yaw_rate_rps": round(yaw_rate, 5), "cmd_v_mps": round(cmd_v, 4), "cmd_w_rps": round(cmd_w, 4),
                "imu_ax": round(float(acc[0]), 4), "imu_ay": round(float(acc[1]), 4), "imu_az": round(float(acc[2]), 4),
                "imu_gx": round(float(gyro[0]), 5), "imu_gy": round(float(gyro[1]), 5), "imu_gz": round(float(gyro[2]), 5),
                "hazard_contact_n": round(contact_n, 3), "contact_object": contact_obj,
            })

        if step % steps_per_frame == 0:
            rgb_path = f"rgb/{len(frame_rows):06d}.jpg"
            if renderer is not None:
                renderer.update_scene(data, "front")
                Image.fromarray(renderer.render()).save(episode_dir / rgb_path, quality=92)
            frame_rows.append({"frame_idx": len(frame_rows), "t_s": round(t, 4), "x_m": round(float(x), 5),
                               "y_m": round(float(y), 5), "yaw_rad": round(yaw, 5), "rgb_path": rgb_path})

        termination = monitor.update(t, float(x), float(y), v_fwd, cmd_v)
        if termination is not None:
            break
        mujoco.mj_step(model, data)
        imu_sum[:3] += data.sensordata[acc_adr:acc_adr + 3]
        imu_sum[3:] += data.sensordata[gyro_adr:gyro_adr + 3]
        imu_count += 1
        step += 1

    meta = {
        "episode_id": episode_id,
        "scenario_id": scenario["scenario_id"],
        "seed": scenario["seed"],
        "condition": condition,
        "appearance_regime": appearance_regime,
        "collector": "scripted_route_following",
        "simulator": {"name": "mujoco_proxy", "version": mujoco.__version__, "physics_dt_s": PHYSICS_DT_S},
        "robot": {
            "model": "nova_carter_sized_proxy",
            "wheel_radius_m": WHEEL_RADIUS_M,
            "half_track_m": WHEEL_HALF_TRACK_M,
            "caster_radius_m": CASTER_RADIUS_M,
            "wheel_torque_limit_nm": WHEEL_TORQUE_LIMIT_NM,
        },
        "camera": CAMERA,
        "rates": {"state_hz": STATE_HZ, "control_hz": CONTROL_HZ, "camera_hz": CAMERA["rate_hz"]},
        "command": {"cruise_speed_mps": CRUISE_SPEED_MPS, "ramp_s": RAMP_S},
        "termination": termination,
        "wall_time_s": round(time.time() - wall_start, 2),
    }
    write_episode(episode_dir, scenario=scene, meta=meta, state_rows=state_rows, frame_rows=frame_rows)
    return episode_dir


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        if "-" in part:
            first, last = part.split("-")
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", required=True, help="e.g. 0-99 or 3,23")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", default="clean,hazard")
    parser.add_argument("--appearance", choices=APPEARANCE_REGIMES, default="default")
    args = parser.parse_args()
    for seed in parse_seeds(args.seeds):
        scenario = create_warehouse_scenario(seed)
        for condition in args.conditions.split(","):
            path = run_episode(scenario, condition, args.output, seed, appearance_regime=args.appearance)
            meta = (path / "meta.json").read_text()
            reason = meta.split('"reason": "')[1].split('"')[0]
            print(f"{path.name:28s} {reason}", flush=True)


if __name__ == "__main__":
    main()
