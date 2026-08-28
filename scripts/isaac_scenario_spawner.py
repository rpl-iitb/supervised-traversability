"""Spawn a starter scenario manifest into an already-open Isaac Sim stage.

Run from Isaac Sim's Script Editor after copying this file and a manifest to
the VM. The adapter creates only primitive collision proxies under
``/World/ScenarioStarter`` and never saves the stage automatically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


DEFAULT_ROBOT_PATH = "/World/nova_carter_reference"
SCENARIO_ROOT_PATH = "/World/ScenarioStarter"


def proxy_spec(object_item: dict) -> dict:
    """Map a manifest item to a primitive proxy description.

    This pure mapping is intentionally usable in normal Python tests.  Runtime
    code below turns it into USD geometry and physics collision prims.
    """
    pose = object_item["pose"]
    asset_key = object_item["asset_key"]
    yaw_deg = round(math.degrees(pose["yaw_rad"]), 2)

    if asset_key == "cable_hose":
        return {
            "primitive": "capsule",
            "translation_m": (pose["x_m"], pose["y_m"], object_item["height_m"]),
            "rotation_deg": (0.0, 90.0, yaw_deg),
            "scale": (1.0, 1.0, 1.0),
            "color": (0.95, 0.45, 0.05),
        }
    if asset_key == "low_debris":
        diameter = 2.0 * object_item["footprint_radius_m"]
        return {
            "primitive": "cube",
            "translation_m": (pose["x_m"], pose["y_m"], object_item["height_m"] / 2.0),
            "rotation_deg": (0.0, 0.0, yaw_deg),
            "scale": (diameter, diameter, object_item["height_m"]),
            "color": (0.85, 0.15, 0.15),
        }
    if asset_key == "small_lip":
        diameter = 2.0 * object_item["footprint_radius_m"]
        return {
            "primitive": "cube",
            "translation_m": (pose["x_m"], pose["y_m"], object_item["height_m"] / 2.0),
            "rotation_deg": (0.0, 0.0, yaw_deg),
            "scale": (diameter, diameter * 0.25, object_item["height_m"]),
            "color": (0.25, 0.25, 0.95),
        }
    raise ValueError(f"Unsupported starter asset: {asset_key}")


def _require_runtime():
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    return omni.usd, Gf, UsdGeom, UsdPhysics


def _set_transform(geom, spec: dict, gf, usd_geom) -> None:
    xform = usd_geom.XformCommonAPI(geom)
    xform.SetTranslate(gf.Vec3d(*spec["translation_m"]))
    xform.SetRotate(gf.Vec3f(*spec["rotation_deg"]), usd_geom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale(gf.Vec3f(*spec["scale"]))
    geom.CreateDisplayColorAttr([gf.Vec3f(*spec["color"])])


def _spawn_proxy(stage, object_item: dict, gf, usd_geom, usd_physics):
    spec = proxy_spec(object_item)
    prim_path = f"{SCENARIO_ROOT_PATH}/{object_item['instance_id']}"

    if spec["primitive"] == "capsule":
        geom = usd_geom.Capsule.Define(stage, prim_path)
        geom.CreateRadiusAttr().Set(object_item["height_m"] / 2.0)
        geom.CreateHeightAttr().Set(2.0 * object_item["footprint_radius_m"])
        geom.CreateAxisAttr().Set("Z")
    else:
        geom = usd_geom.Cube.Define(stage, prim_path)
        geom.CreateSizeAttr().Set(1.0)

    _set_transform(geom, spec, gf, usd_geom)
    usd_physics.CollisionAPI.Apply(geom.GetPrim())
    return geom.GetPrim()


def _position_robot(stage, robot_path: str, robot_pose: dict, gf, usd_geom) -> None:
    robot = stage.GetPrimAtPath(robot_path)
    if not robot or not robot.IsValid():
        raise RuntimeError(
            f"Robot root {robot_path!r} was not found. Drag the Nova Carter reference into the stage first."
        )
    xform = usd_geom.XformCommonAPI(robot)
    xform.SetTranslate(gf.Vec3d(robot_pose["x_m"], robot_pose["y_m"], 0.0))
    xform.SetRotate(
        gf.Vec3f(0.0, 0.0, math.degrees(robot_pose["yaw_rad"])),
        usd_geom.XformCommonAPI.RotationOrderXYZ,
    )


def spawn_manifest(manifest_path: str | Path, robot_path: str = DEFAULT_ROBOT_PATH) -> None:
    """Create primitive collision hazards and position the robot from a manifest.

    The current stage must already contain the warehouse and Nova Carter.  Only
    the managed scenario root is replaced, and the user must save manually.
    """
    omni_usd, gf, usd_geom, usd_physics = _require_runtime()
    stage = omni_usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open")

    manifest = json.loads(Path(manifest_path).read_text())
    if stage.GetPrimAtPath(SCENARIO_ROOT_PATH).IsValid():
        stage.RemovePrim(SCENARIO_ROOT_PATH)
    usd_geom.Xform.Define(stage, SCENARIO_ROOT_PATH)

    _position_robot(stage, robot_path, manifest["robot"]["start_pose"], gf, usd_geom)
    spawned = [
        _spawn_proxy(stage, object_item, gf, usd_geom, usd_physics)
        for object_item in manifest["objects"]
    ]
    print(
        f"Spawned {len(spawned)} starter proxies for {manifest['scenario_id']}. "
        "Use File → Save As to preserve this stage."
    )
