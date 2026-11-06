"""Headless Isaac Sim collector: one scenario manifest -> one episode directory.

Writes exactly the format of ``scripts/episode_io.py`` (the same one the
MuJoCo proxy writes), so ``scripts/frame_labels.py`` and ``learning/`` consume
Isaac episodes unchanged.

Run with Isaac Sim's own Python (inside the Isaac Sim container, or on any
RTX workstation)::

    ./python.sh isaac/collect_episode.py \
        --manifest scenarios/warehouse-0003.json --condition hazard \
        --output /scratch/$USER/episodes

Status: written against the Isaac Sim 4.5/5.x ``isaacsim.*`` Python API and
**not run end to end; the results in this repository use the MuJoCo proxy**. Run ``--check-only`` first; it loads the stage,
spawns the scenario, validates the route and exits without driving.

Known limits: the oil spill is visual-only in
Isaac (no friction material yet), and the procedural hazards are first-pass
assets. The robot is driven by the shared scripted route follower rather
than a human teleoperator; teleoperation will reuse the same recorder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--condition", choices=("clean", "hazard"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--environment", default="/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                    help="Warehouse USD relative to the Isaac assets root")
parser.add_argument("--robot-usd", default="/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd")
parser.add_argument("--robot-path", default="/World/nova_carter_reference")
parser.add_argument("--wheel-joints", default="joint_wheel_left,joint_wheel_right")
parser.add_argument("--wheel-radius-m", type=float, default=0.14)
parser.add_argument("--wheel-base-m", type=float, default=0.4132)
parser.add_argument("--assets-root", default=None, help="Override the Isaac assets root (e.g. a local mirror)")
parser.add_argument("--check-only", action="store_true", help="Load, spawn and validate the route, then exit")
parser.add_argument("--skip-route-check", action="store_true")
args, _unknown = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402  (must precede every other Isaac import)

CAMERA_W, CAMERA_H, CAMERA_HZ = 320, 240, 10.0
simulation_app = SimulationApp({"headless": True, "width": CAMERA_W, "height": CAMERA_H, "anti_aliasing": 0})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, open_stage  # noqa: E402
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController  # noqa: E402
from isaacsim.robot.wheeled_robots.robots import WheeledRobot  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.sensors.physics import IMUSensor  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

from scripts.episode_io import write_episode  # noqa: E402
from scripts.isaac_scenario_spawner import SCENARIO_ROOT_PATH, spawn_manifest  # noqa: E402
from scripts.proxy_sim import clean_variant  # noqa: E402
from scripts.route_follower import LOOKAHEAD_TAIL_M, Route, TerminationMonitor, route_command  # noqa: E402

PHYSICS_HZ, STATE_HZ, CONTROL_HZ = 200, 100, 50
CAMERA_MOUNT_BASE = (0.31, 0.0, 0.35)
CAMERA_PITCH_DOWN_DEG = 15.0
CAMERA_HFOV_DEG = 90.0


def log(message: str) -> None:
    print(f"[collect_episode] {message}", flush=True)


def yaw_pitch_roll(quat_wxyz) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return yaw, pitch, roll


def base_to_world(pose: dict, forward: float, left: float) -> tuple[float, float]:
    c, s = math.cos(pose["yaw_rad"]), math.sin(pose["yaw_rad"])
    return pose["x_m"] + c * forward - s * left, pose["y_m"] + s * forward + c * left


def check_route_clear(stage, scene: dict, robot_path: str) -> list[str]:
    """Overlap-test the robot corridor along the route; return blocking prim paths.

    Ignores the floor, the robot and the managed scenario objects, so any hit
    is a warehouse obstacle (rack, wall, pallet) sitting in the chosen aisle.
    """
    from omni.physx import get_physx_scene_query_interface

    start = scene["route"]["start_pose"]
    hits: set[str] = set()

    def report(hit) -> bool:
        path = str(hit.rigid_body or hit.collision)
        if not (path.startswith(robot_path) or path.startswith(SCENARIO_ROOT_PATH)):
            hits.add(path)
        return True

    quat = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(start["yaw_rad"])).GetQuat())
    for forward in np.arange(0.0, scene["route"]["length_m"] + LOOKAHEAD_TAIL_M + 0.5, 0.5):
        x, y = base_to_world(start, float(forward), 0.0)
        # Box from 5 cm to 60 cm above the floor so the floor itself is not hit.
        get_physx_scene_query_interface().overlap_box(
            Gf.Vec3f(0.35, 0.35, 0.275), Gf.Vec3f(x, y, 0.325),
            Gf.Vec4f(quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2], quat.GetReal()),
            report, False,
        )
    return sorted(hits)


def enable_contact_reports(stage, robot_path: str) -> None:
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith(robot_path) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(0.0)


def main() -> None:
    scenario = json.loads(args.manifest.read_text())
    scene = clean_variant(scenario) if args.condition == "clean" else scenario
    episode_id = f"{scenario['scenario_id']}-{args.condition}"
    episode_dir = args.output / episode_id
    (episode_dir / "rgb").mkdir(parents=True, exist_ok=True)

    assets_root = args.assets_root or get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac assets root unavailable; pass --assets-root pointing at a local asset mirror")
    log(f"assets root: {assets_root}")
    open_stage(assets_root.rstrip("/") + args.environment)
    for _ in range(20):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(args.robot_path).IsValid():
        add_reference_to_stage(assets_root.rstrip("/") + args.robot_usd, args.robot_path)

    scene_path = episode_dir / "scenario.json"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n")
    spawn_manifest(str(scene_path), robot_path=args.robot_path)
    enable_contact_reports(stage, args.robot_path)

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / CAMERA_HZ)
    wheels = args.wheel_joints.split(",")
    robot = world.scene.add(WheeledRobot(prim_path=args.robot_path, name="robot", wheel_dof_names=wheels, create_robot=False))
    controller = DifferentialController(name="route", wheel_radius=args.wheel_radius_m, wheel_base=args.wheel_base_m)

    start = scene["robot"]["start_pose"]
    cam_x, cam_y = base_to_world(start, CAMERA_MOUNT_BASE[0], CAMERA_MOUNT_BASE[1])
    chassis_path = next(
        (str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith(args.robot_path) and p.GetName() == "chassis_link"),
        args.robot_path,
    )
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    camera = Camera(
        prim_path=f"{chassis_path}/traversability_camera",
        position=np.array([cam_x, cam_y, CAMERA_MOUNT_BASE[2]]),
        orientation=euler_angles_to_quat(np.array([0.0, CAMERA_PITCH_DOWN_DEG, math.degrees(start["yaw_rad"])]), degrees=True),
        frequency=CAMERA_HZ,
        resolution=(CAMERA_W, CAMERA_H),
    )
    usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(f"{chassis_path}/traversability_camera"))
    focal = 10.0
    usd_camera.GetFocalLengthAttr().Set(focal)
    usd_camera.GetHorizontalApertureAttr().Set(2 * focal * math.tan(math.radians(CAMERA_HFOV_DEG / 2)))
    usd_camera.GetVerticalApertureAttr().Set(2 * focal * math.tan(math.radians(CAMERA_HFOV_DEG / 2)) * CAMERA_H / CAMERA_W)
    usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
    imu = IMUSensor(prim_path=f"{chassis_path}/traversability_imu", name="imu", frequency=STATE_HZ)

    world.reset()
    camera.initialize()
    for _ in range(10):
        world.step(render=True)

    if not args.skip_route_check:
        blocking = check_route_clear(stage, scene, args.robot_path)
        if blocking:
            log("ROUTE BLOCKED by: " + ", ".join(blocking[:10]))
            log("Choose another --origin-x-m/--origin-y-m when generating the manifest.")
            simulation_app.close()
            sys.exit(3)
        log("route corridor is clear")
    if args.check_only:
        rgb = camera.get_rgba()
        log(f"check-only OK: camera frame {None if rgb is None else rgb.shape}, intrinsics {camera.get_intrinsics_matrix().tolist()}")
        simulation_app.close()
        return

    intrinsics = camera.get_intrinsics_matrix()
    camera_meta = {
        "width": CAMERA_W, "height": CAMERA_H,
        "fx": float(intrinsics[0][0]), "fy": float(intrinsics[1][1]),
        "cx": float(intrinsics[0][2]), "cy": float(intrinsics[1][2]),
        "horizontal_fov_deg": CAMERA_HFOV_DEG, "mount_xyz_m": list(CAMERA_MOUNT_BASE),
        "pitch_down_deg": CAMERA_PITCH_DOWN_DEG, "rate_hz": CAMERA_HZ,
    }

    from omni.physx import get_physx_simulation_interface

    object_ids = {item["instance_id"] for item in scene["objects"]}
    step_contacts: dict[str, float] = {}

    def on_contacts(headers, data) -> None:
        for header in headers:
            paths = (str(PhysicsSchemaTools.intToSdfPath(header.actor0)), str(PhysicsSchemaTools.intToSdfPath(header.actor1)))
            other = next((p for p in paths if p.startswith(SCENARIO_ROOT_PATH + "/")), None)
            if other is None or not any(p.startswith(args.robot_path) for p in paths):
                continue
            instance = other[len(SCENARIO_ROOT_PATH) + 1:].split("/")[0]
            if instance not in object_ids:
                continue
            impulse = sum(
                np.linalg.norm(np.array(data[i].impulse))
                for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data)
            )
            step_contacts[instance] = step_contacts.get(instance, 0.0) + float(impulse) * PHYSICS_HZ

    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)

    route = Route.from_manifest(scene)
    monitor = TerminationMonitor(route)
    state_rows, frame_rows = [], []
    status = {"step": 0, "cmd": (0.0, 0.0), "termination": None, "t0": world.current_time}

    def on_physics_step(_dt: float) -> None:
        t = world.current_time - status["t0"]
        position, orientation = robot.get_world_pose()
        yaw, pitch, roll = yaw_pitch_roll(orientation)
        linear, angular = robot.get_linear_velocity(), robot.get_angular_velocity()
        v_fwd = float(linear[0] * math.cos(yaw) + linear[1] * math.sin(yaw))
        if status["step"] % (PHYSICS_HZ // CONTROL_HZ) == 0:
            status["cmd"] = route_command(route, t, float(position[0]), float(position[1]), yaw)
            robot.apply_wheel_actions(controller.forward(np.array(status["cmd"])))
        if status["step"] % (PHYSICS_HZ // STATE_HZ) == 0:
            frame = imu.get_current_frame()
            acc, gyro = frame["lin_acc"], frame["ang_vel"]
            contact_obj = max(step_contacts, key=step_contacts.get) if step_contacts else ""
            state_rows.append({
                "t_s": round(t, 4), "x_m": round(float(position[0]), 5), "y_m": round(float(position[1]), 5),
                "z_m": round(float(position[2]), 5), "yaw_rad": round(yaw, 5), "pitch_rad": round(pitch, 5),
                "roll_rad": round(roll, 5), "v_fwd_mps": round(v_fwd, 5), "yaw_rate_rps": round(float(angular[2]), 5),
                "cmd_v_mps": round(status["cmd"][0], 4), "cmd_w_rps": round(status["cmd"][1], 4),
                "imu_ax": round(float(acc[0]), 4), "imu_ay": round(float(acc[1]), 4), "imu_az": round(float(acc[2]), 4),
                "imu_gx": round(float(gyro[0]), 5), "imu_gy": round(float(gyro[1]), 5), "imu_gz": round(float(gyro[2]), 5),
                "hazard_contact_n": round(sum(step_contacts.values()), 3), "contact_object": contact_obj,
            })
            step_contacts.clear()
            if status["termination"] is None:
                status["termination"] = monitor.update(t, float(position[0]), float(position[1]), v_fwd, status["cmd"][0])
        status["step"] += 1

    world.add_physics_callback("traversability_recorder", on_physics_step)
    wall_start = time.time()
    from PIL import Image

    while simulation_app.is_running() and status["termination"] is None:
        world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            continue
        position, orientation = robot.get_world_pose()
        yaw, _, _ = yaw_pitch_roll(orientation)
        rgb_path = f"rgb/{len(frame_rows):06d}.jpg"
        Image.fromarray(np.asarray(rgba)[:, :, :3].astype(np.uint8)).save(episode_dir / rgb_path, quality=92)
        frame_rows.append({"frame_idx": len(frame_rows), "t_s": round(world.current_time - status["t0"], 4),
                           "x_m": round(float(position[0]), 5), "y_m": round(float(position[1]), 5),
                           "yaw_rad": round(yaw, 5), "rgb_path": rgb_path})

    world.remove_physics_callback("traversability_recorder")
    contact_subscription = None  # noqa: F841  (unsubscribes)
    meta = {
        "episode_id": episode_id,
        "scenario_id": scenario["scenario_id"],
        "seed": scenario["seed"],
        "condition": args.condition,
        "collector": "scripted_route_following",
        "simulator": {"name": "isaac_sim", "physics_hz": PHYSICS_HZ, "environment": args.environment},
        "robot": {"model": "nova_carter", "usd": args.robot_usd, "wheel_radius_m": args.wheel_radius_m,
                  "wheel_base_m": args.wheel_base_m},
        "camera": camera_meta,
        "rates": {"state_hz": STATE_HZ, "control_hz": CONTROL_HZ, "camera_hz": CAMERA_HZ},
        "termination": status["termination"] or {"reason": "app_closed"},
        "wall_time_s": round(time.time() - wall_start, 2),
        "known_limitations": ["oil_spill is visual-only in Isaac (no friction material)"],
    }
    write_episode(episode_dir, scenario=scene, meta=meta, state_rows=state_rows, frame_rows=frame_rows)
    log(f"wrote {episode_dir} ({len(frame_rows)} frames, {len(state_rows)} state rows, {meta['termination']})")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
