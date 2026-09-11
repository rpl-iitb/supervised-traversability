"""Create a labelled run manifest from a scenario plan and Nova Carter CSV log.

This bridges the first simulator artifacts into an inspectable dataset record.
It does not fabricate unavailable sensor signals: IMU/contact fields remain
explicit collection inputs until ROS 2 sensor recording is connected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scripts.starter_pipeline import label_from_outcome


REQUIRED_CSV_COLUMNS = {
    "elapsed_s",
    "x_m",
    "y_m",
    "yaw_rad",
    "speed_mps",
    "command_wheel_deg_s",
}
RUN_MANIFEST_SCHEMA_VERSION = "1.0"


def _load_csv_rows(csv_path: str | Path) -> list[dict[str, float]]:
    with Path(csv_path).open(newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_CSV_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Telemetry CSV is missing required columns: {sorted(missing)}")
        return [{field: float(row[field]) for field in REQUIRED_CSV_COLUMNS} for row in reader]


def telemetry_summary(rows: list[dict[str, float]]) -> dict[str, float | int]:
    """Summarize the observed pose log without assuming a wheel-speed model."""
    if not rows:
        raise ValueError("Telemetry CSV contains no samples")

    path_length_m = sum(
        math.hypot(current["x_m"] - previous["x_m"], current["y_m"] - previous["y_m"])
        for previous, current in zip(rows, rows[1:])
    )
    duration_s = max(0.0, rows[-1]["elapsed_s"] - rows[0]["elapsed_s"])
    mean_speed_mps = path_length_m / duration_s if duration_s > 0.0 else 0.0
    return {
        "sample_count": len(rows),
        "duration_s": round(duration_s, 3),
        "path_length_m": round(path_length_m, 3),
        "mean_speed_mps": round(mean_speed_mps, 3),
    }


def object_encounters(objects: list[dict], rows: list[dict[str, float]]) -> list[dict]:
    """Associate every known scene object with its closest logged robot pose."""
    encounters = []
    for object_item in objects:
        object_pose = object_item["pose"]
        closest = min(
            rows,
            key=lambda row: math.hypot(
                row["x_m"] - object_pose["x_m"], row["y_m"] - object_pose["y_m"]
            ),
        )
        encounters.append(
            {
                "instance_id": object_item["instance_id"],
                "asset_key": object_item["asset_key"],
                "closest_elapsed_s": round(closest["elapsed_s"], 3),
                "closest_distance_m": round(
                    math.hypot(
                        closest["x_m"] - object_pose["x_m"],
                        closest["y_m"] - object_pose["y_m"],
                    ),
                    3,
                ),
            }
        )
    return encounters


def build_run_manifest(
    *,
    scenario_path: str | Path,
    csv_path: str | Path,
    crossing_completed: bool,
    speed_tracking_ratio: float,
    safety_stop: bool,
    contact_impulse_ns: float,
) -> dict:
    """Combine reproducible scene data, measured telemetry, and an outcome label."""
    scenario_path = Path(scenario_path)
    csv_path = Path(csv_path)
    scenario = json.loads(scenario_path.read_text())
    rows = _load_csv_rows(csv_path)
    label = label_from_outcome(
        crossing_completed=crossing_completed,
        speed_tracking_ratio=speed_tracking_ratio,
        safety_stop=safety_stop,
        contact_impulse_ns=contact_impulse_ns,
    )
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "scenario": {
            "scenario_id": scenario["scenario_id"],
            "seed": scenario["seed"],
            "robot": scenario["robot"],
            "objects": scenario["objects"],
        },
        "telemetry": telemetry_summary(rows),
        "object_encounters": object_encounters(scenario["objects"], rows),
        "label_inputs": {
            "crossing_completed": crossing_completed,
            "speed_tracking_ratio": speed_tracking_ratio,
            "safety_stop": safety_stop,
            "contact_impulse_ns": contact_impulse_ns,
        },
        "traversability_label": label,
        "sources": {
            "scenario_manifest": scenario_path.name,
            "telemetry_csv": csv_path.name,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crossing-completed", action="store_true")
    parser.add_argument("--speed-tracking-ratio", type=float, required=True)
    parser.add_argument("--safety-stop", action="store_true")
    parser.add_argument("--contact-impulse-ns", type=float, default=0.0)
    args = parser.parse_args()

    manifest = build_run_manifest(
        scenario_path=args.scenario,
        csv_path=args.csv,
        crossing_completed=args.crossing_completed,
        speed_tracking_ratio=args.speed_tracking_ratio,
        safety_stop=args.safety_stop,
        contact_impulse_ns=args.contact_impulse_ns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
