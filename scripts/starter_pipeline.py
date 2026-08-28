"""Create deterministic starter scenarios and provisional traversability labels.

This module deliberately has no Isaac Sim runtime dependency.  It produces a
manifest that a later Isaac adapter will turn into USD references, and it keeps
the initial label rule explicit and reviewable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


SCHEMA_VERSION = "1.0"
ROBOT_FOOTPRINT_RADIUS_M = 0.45
ROBOT_OBJECT_CLEARANCE_M = 0.75
OBJECT_OBJECT_CLEARANCE_M = 0.30
MAX_PLACEMENT_ATTEMPTS = 200
CONTACT_IMPULSE_THRESHOLD_NS = 5.0

STARTER_OBJECTS = (
    {
        "asset_key": "cable_hose",
        "semantic_class": "low_profile_cable",
        "footprint_radius_m": 0.18,
        "height_m": 0.03,
    },
    {
        "asset_key": "low_debris",
        "semantic_class": "low_profile_debris",
        "footprint_radius_m": 0.22,
        "height_m": 0.08,
    },
    {
        "asset_key": "small_lip",
        "semantic_class": "height_discontinuity",
        "footprint_radius_m": 0.40,
        "height_m": 0.04,
    },
)


def _distance(first: dict[str, float], second: dict[str, float]) -> float:
    return math.hypot(first["x_m"] - second["x_m"], first["y_m"] - second["y_m"])


def _random_robot_start(rng: random.Random) -> dict[str, float]:
    """Sample a pose facing generally into the warehouse aisle."""
    return {
        "x_m": round(rng.uniform(-5.0, -4.0), 3),
        "y_m": round(rng.uniform(-2.0, 2.0), 3),
        "yaw_rad": round(rng.uniform(-0.15, 0.15), 3),
    }


def _is_clear(candidate: dict, robot_pose: dict[str, float], placed: list[dict]) -> bool:
    if _distance(candidate["pose"], robot_pose) < (
        ROBOT_FOOTPRINT_RADIUS_M
        + candidate["footprint_radius_m"]
        + ROBOT_OBJECT_CLEARANCE_M
    ):
        return False

    for existing in placed:
        if _distance(candidate["pose"], existing["pose"]) < (
            candidate["footprint_radius_m"]
            + existing["footprint_radius_m"]
            + OBJECT_OBJECT_CLEARANCE_M
        ):
            return False
    return True


def create_starter_scenario(seed: int) -> dict:
    """Create a reproducible Nova Carter scenario plan with three starter hazards."""
    rng = random.Random(seed)
    robot_pose = _random_robot_start(rng)
    placed: list[dict] = []

    for index, object_spec in enumerate(STARTER_OBJECTS, start=1):
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            candidate = {
                **object_spec,
                "instance_id": f"{object_spec['asset_key']}_{index:02d}",
                "pose": {
                    "x_m": round(rng.uniform(0.5, 4.5), 3),
                    "y_m": round(rng.uniform(-2.2, 2.2), 3),
                    "yaw_rad": round(rng.uniform(-math.pi, math.pi), 3),
                },
            }
            if _is_clear(candidate, robot_pose, placed):
                placed.append(candidate)
                break
        else:
            raise RuntimeError(f"Could not place {object_spec['asset_key']} safely")

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": f"starter-{seed:04d}",
        "seed": seed,
        "robot": {
            "model": "nova_carter",
            "start_pose": robot_pose,
        },
        "objects": placed,
        "collection": {
            "mode": "teleoperation",
            "planned_sensors": ["d455_rgbd_imu", "lidar_2d", "odometry", "tf", "joint_state"],
        },
    }


def minimum_clearance_ok(scenario: dict) -> bool:
    """Validate the spacing invariants used by the starter scenario generator."""
    robot_pose = scenario["robot"]["start_pose"]
    objects = scenario["objects"]
    for index, object_item in enumerate(objects):
        if _distance(object_item["pose"], robot_pose) < (
            ROBOT_FOOTPRINT_RADIUS_M
            + object_item["footprint_radius_m"]
            + ROBOT_OBJECT_CLEARANCE_M
        ):
            return False
        for other in objects[index + 1 :]:
            if _distance(object_item["pose"], other["pose"]) < (
                object_item["footprint_radius_m"]
                + other["footprint_radius_m"]
                + OBJECT_OBJECT_CLEARANCE_M
            ):
                return False
    return True


def label_from_outcome(
    *,
    crossing_completed: bool,
    speed_tracking_ratio: float,
    safety_stop: bool,
    contact_impulse_ns: float,
) -> dict:
    """Build an initial outcome-grounded score and a conservative binary label.

    The score is a continuous motion-quality proxy in ``[0, 1]``.  A binary
    label is emitted only for clear outcomes; borderline runs stay unlabelled
    rather than contaminating the first training set.
    """
    score = round(max(0.0, min(1.0, speed_tracking_ratio)), 2)
    if safety_stop:
        return {"score": 0.0, "binary_label": 0, "reason": "safety_stop"}
    if contact_impulse_ns >= CONTACT_IMPULSE_THRESHOLD_NS:
        return {"score": 0.0, "binary_label": 0, "reason": "high_contact_impulse"}
    if crossing_completed and score >= 0.80:
        return {"score": score, "binary_label": 1, "reason": "normal_crossing"}
    if score <= 0.50:
        return {"score": score, "binary_label": 0, "reason": "poor_speed_tracking"}
    return {"score": score, "binary_label": None, "reason": "ambiguous_outcome"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Deterministic scenario seed")
    parser.add_argument("--output", type=Path, required=True, help="Manifest JSON output path")
    args = parser.parse_args()

    scenario = create_starter_scenario(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
