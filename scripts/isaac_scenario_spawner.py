"""Spawn a seeded warehouse scenario into an already-open Isaac Sim stage.

Run from Isaac Sim's Script Editor after copying this file and a manifest to
the VM. Task-specific hazards are generated procedurally; normal clutter can
be referenced from Isaac Sim's asset pack. The adapter never saves the stage.
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


def spawn_spec(object_item: dict) -> dict:
    """Describe the visual/physics implementation for one manifest item.

    Kept free of Isaac imports so that the scenario catalogue can be tested
    outside the simulator.
    """
    asset_key = object_item["asset_key"]
    if asset_key == "cable_hose":
        return {"kind": "curved_cable", "has_collision": True}
    if asset_key == "plastic_wrapper":
        return {"kind": "crumpled_wrapper", "has_collision": True}
    if asset_key == "oil_spill":
        return {"kind": "irregular_spill", "has_collision": False}
    if asset_key == "small_lip":
        return {"kind": "small_lip", "has_collision": True}
    if asset_key == "cardboard_box":
        return {
            "kind": "isaac_usd_reference",
            "has_collision": False,
            "usd_relative_path": object_item["usd_relative_path"],
        }
    return {"kind": "legacy_proxy", "has_collision": True, "proxy": proxy_spec(object_item)}


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


def _set_xform(prim, translation_m, rotation_deg, gf, usd_geom, scale=(1.0, 1.0, 1.0)) -> None:
    xform = usd_geom.XformCommonAPI(prim)
    xform.SetTranslate(gf.Vec3d(*translation_m))
    xform.SetRotate(gf.Vec3f(*rotation_deg), usd_geom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale(gf.Vec3f(*scale))


def _object_root(stage, object_item: dict, gf, usd_geom):
    pose = object_item["pose"]
    root = usd_geom.Xform.Define(stage, f"{SCENARIO_ROOT_PATH}/{object_item['instance_id']}")
    _set_xform(
        root,
        (pose["x_m"], pose["y_m"], 0.0),
        (0.0, 0.0, math.degrees(pose["yaw_rad"])),
        gf,
        usd_geom,
    )
    return root


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


def _spawn_curved_cable(stage, object_item: dict, gf, usd_geom, usd_physics):
    """Create a visible, curved cable from small collision-backed capsules."""
    root = _object_root(stage, object_item, gf, usd_geom)
    radius_m = max(0.012, object_item["height_m"] * 0.45)
    half_length_m = object_item["footprint_radius_m"]
    segment_count = 7
    centres = []
    for index in range(segment_count + 1):
        progress = index / segment_count
        x_m = -half_length_m + 2.0 * half_length_m * progress
        y_m = 0.055 * math.sin(progress * math.pi * 2.0)
        centres.append((x_m, y_m))

    for index, (first, second) in enumerate(zip(centres, centres[1:])):
        dx_m, dy_m = second[0] - first[0], second[1] - first[1]
        segment_length_m = math.hypot(dx_m, dy_m)
        capsule = usd_geom.Capsule.Define(stage, f"{root.GetPath()}/segment_{index:02d}")
        capsule.CreateRadiusAttr().Set(radius_m)
        capsule.CreateHeightAttr().Set(max(segment_length_m, radius_m * 2.0))
        capsule.CreateAxisAttr().Set("Z")
        _set_xform(
            capsule,
            ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0, radius_m),
            (0.0, 90.0, math.degrees(math.atan2(dy_m, dx_m))),
            gf,
            usd_geom,
        )
        capsule.CreateDisplayColorAttr([gf.Vec3f(0.08, 0.08, 0.08)])
        usd_physics.CollisionAPI.Apply(capsule.GetPrim())
    return root.GetPrim()


def _spawn_crumpled_wrapper(stage, object_item: dict, gf, usd_geom, usd_physics):
    """Create a thin, folded wrapper mesh plus an invisible collision proxy."""
    root = _object_root(stage, object_item, gf, usd_geom)
    radius_m = object_item["footprint_radius_m"]
    mesh = usd_geom.Mesh.Define(stage, f"{root.GetPath()}/visual_mesh")
    mesh.CreatePointsAttr(
        [
            gf.Vec3f(0.0, 0.0, object_item["height_m"]),
            gf.Vec3f(-radius_m, -radius_m * 0.60, 0.003),
            gf.Vec3f(radius_m * 0.75, -radius_m * 0.70, 0.010),
            gf.Vec3f(radius_m, radius_m * 0.50, 0.002),
            gf.Vec3f(-radius_m * 0.65, radius_m * 0.80, 0.014),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1])
    mesh.CreateDisplayColorAttr([gf.Vec3f(0.90, 0.88, 0.72)])

    collider = usd_geom.Cube.Define(stage, f"{root.GetPath()}/collision_proxy")
    collider.CreateSizeAttr().Set(1.0)
    _set_xform(
        collider,
        (0.0, 0.0, object_item["height_m"] / 2.0),
        (0.0, 0.0, 0.0),
        gf,
        usd_geom,
        scale=(radius_m * 2.0, radius_m * 1.4, object_item["height_m"]),
    )
    usd_geom.Imageable(collider.GetPrim()).MakeInvisible()
    usd_physics.CollisionAPI.Apply(collider.GetPrim())
    return root.GetPrim()


def _spawn_irregular_spill(stage, object_item: dict, gf, usd_geom):
    """Create a visually irregular spill. Friction is intentionally not faked."""
    root = _object_root(stage, object_item, gf, usd_geom)
    radius_m = object_item["footprint_radius_m"]
    mesh = usd_geom.Mesh.Define(stage, f"{root.GetPath()}/visual_mesh")
    points = [gf.Vec3f(0.0, 0.0, object_item["height_m"])]
    radial_scale = (0.78, 1.00, 0.83, 1.10, 0.92, 1.04, 0.75)
    for index, scale in enumerate(radial_scale):
        angle = index * 2.0 * math.pi / len(radial_scale)
        points.append(gf.Vec3f(radius_m * scale * math.cos(angle), radius_m * scale * math.sin(angle), 0.0))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(radial_scale))
    indices = []
    for index in range(len(radial_scale)):
        indices.extend((0, index + 1, (index + 1) % len(radial_scale) + 1))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDisplayColorAttr([gf.Vec3f(0.06, 0.04, 0.02)])
    return root.GetPrim()


def _spawn_isaac_reference(stage, object_item: dict, gf, usd_geom):
    root = _object_root(stage, object_item, gf, usd_geom)
    try:
        from isaacsim.storage.native import get_assets_root_path

        asset_root = get_assets_root_path()
    except Exception as error:
        raise RuntimeError("Could not resolve the configured Isaac Sim asset root") from error
    if not asset_root:
        raise RuntimeError("Isaac Sim asset root is unavailable; check Utilities → Check Default Assets Root Path")
    root.GetPrim().GetReferences().AddReference(asset_root.rstrip("/") + object_item["usd_relative_path"])
    return root.GetPrim()


def _spawn_object(stage, object_item: dict, gf, usd_geom, usd_physics):
    kind = spawn_spec(object_item)["kind"]
    if kind == "legacy_proxy":
        return _spawn_proxy(stage, object_item, gf, usd_geom, usd_physics)
    if kind == "curved_cable":
        return _spawn_curved_cable(stage, object_item, gf, usd_geom, usd_physics)
    if kind == "crumpled_wrapper":
        return _spawn_crumpled_wrapper(stage, object_item, gf, usd_geom, usd_physics)
    if kind == "irregular_spill":
        return _spawn_irregular_spill(stage, object_item, gf, usd_geom)
    if kind == "small_lip":
        return _spawn_proxy(stage, object_item, gf, usd_geom, usd_physics)
    if kind == "isaac_usd_reference":
        return _spawn_isaac_reference(stage, object_item, gf, usd_geom)
    raise RuntimeError(f"No Isaac runtime spawner for {kind!r}")


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
    spawned = [_spawn_object(stage, object_item, gf, usd_geom, usd_physics) for object_item in manifest["objects"]]
    print(
        f"Spawned {len(spawned)} scenario objects for {manifest['scenario_id']}. "
        "Use File → Save As to preserve this stage."
    )
