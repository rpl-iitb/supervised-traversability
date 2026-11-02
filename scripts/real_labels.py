# UNVERIFIED: AI-written; its truth check and tests cover the logic, not independently reviewed.
"""Label the real-robot dataset and check the zones against the recorded placements.

    python3 -m scripts.real_labels --data /path/data/real/s1

Labels use only the robot's signals. Placements are used afterwards: to name which
object a photo shows (for evaluation) and in the truth check. Odometry reads long on
this robot, so a placement in true metres becomes ``ODOM_SCALE * metres`` along the
odometry line, plus one per-run offset (the robot was not always started exactly on
the mark) fitted to that run's own zones. Because of that fit the truth check shows
that zones land in the right PATTERN, not absolute positioning accuracy.
Writes labels.csv, labels.summary.json, labels.zones.json.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from scripts.episode_io import load_episode
from scripts.frame_labels_v2 import JACKAL_REAL, LABEL_COLUMNS, Settings, calibrate, drive_zones, episode_signals, label_episode, truth_check

ODOM_SCALE = 1.055  # median of end-of-line odometry / taped length over the complete runs
TRUTH_TOLERANCE_M = 0.25
OFFSET_RANGE_M = 1.0
VISIBLE_M = (0.4, 5.0)  # an object can be seen in the photo over this range ahead of the robot centre
ROI_M = (0.3, 2.8)  # an object can be inside the masked floor corridor over this range
EXTRA = ("run", "session", "direction", "condition_run", "objects_visible", "objects_in_roi", "object_ahead_m")


def fit_offset(placed_m: list[float], zone_centres: list[float], cap_m: float = 0.4) -> float:
    """Shift of a run's placements that best lines them up with its zones (robust: misses cost at most cap_m)."""
    if not placed_m or not zone_centres:
        return 0.0
    zones = np.array(zone_centres)
    candidates = np.arange(-OFFSET_RANGE_M, OFFSET_RANGE_M + 1e-9, 0.01)
    cost = [sum(min(float(np.abs(zones - (d + o)).min()), cap_m) for d in placed_m) + 0.01 * abs(o) for o in candidates]
    return round(float(candidates[int(np.argmin(cost))]), 2)


def place_objects(episode: dict, offset_m: float) -> None:
    route = episode["scenario"]["route"]["start_pose"]
    for item in episode["scenario"]["objects"]:
        d = item["measured_distance_m"] * ODOM_SCALE + offset_m
        item["pose"] = {"x_m": route["x_m"] + d * math.cos(route["yaw_rad"]), "y_m": route["y_m"] + d * math.sin(route["yaw_rad"])}


def load_runs(data: Path) -> list[dict]:
    with (data / "runs.csv").open(newline="") as file:
        runs = list(csv.DictReader(file))
    for run in runs:
        episode = load_episode(data / "episodes" / run["run"])
        place_objects(episode, 0.0)
        run["episode"] = episode
    return runs


def build(runs: list[dict], settings: Settings):
    clean = [r["episode"] for r in runs if r["condition"] == "clean"]
    thresholds = calibrate(clean, [], settings)
    rows, report = [], []
    for run in runs:
        episode = run["episode"]
        signals = episode_signals(episode, settings)
        zones = drive_zones(signals, thresholds, episode["meta"]["termination"], settings)
        offset = fit_offset([o["measured_distance_m"] * ODOM_SCALE for o in episode["scenario"]["objects"]],
                            [(z["lo"] + z["hi"]) / 2 for z in zones])
        place_objects(episode, offset)
        labelled = label_episode(episode, zones, settings)
        route = signals["route"]
        placed = [(o["asset_key"], route.progress_and_offset(o["pose"]["x_m"], o["pose"]["y_m"])[0]) for o in episode["scenario"]["objects"]]
        for row, frame in zip(labelled, episode["frames"]):
            s, _ = route.progress_and_offset(frame["x_m"], frame["y_m"])
            ahead = [(k, d - s) for k, d in placed if VISIBLE_M[0] <= d - s <= VISIBLE_M[1]]
            row.update(run=run["run"], session=run["session"], direction=run["direction"], condition_run=run["condition"],
                       objects_visible="+".join(sorted({k for k, _ in ahead})),
                       objects_in_roi="+".join(sorted({k for k, d in placed if ROI_M[0] <= d - s <= ROI_M[1]})),
                       object_ahead_m=round(min((d for _, d in ahead), default=-1.0), 3))
        rows.extend(labelled)
        report.append({"episode_id": run["run"], "condition": run["condition"], "zones": zones, "offset_m": offset,
                       "truth": truth_check(episode, zones, tolerance_m=TRUTH_TOLERANCE_M)})
    return rows, report, thresholds


def summarise(rows: list[dict], report: list[dict], thresholds: dict, settings: Settings) -> dict:
    moving = [r for r in rows if r["reason"] not in ("not_moving", "beyond_route")]
    by_class = collections.defaultdict(lambda: dict(n=0, found=0, unacceptable=0))
    for entry in report:
        for hit in entry["truth"]["hazards"]:
            c = by_class[hit["asset_key"]]
            c["n"] += 1; c["found"] += hit["found"]; c["unacceptable"] += hit["unacceptable"]
    frames_by_class = collections.defaultdict(collections.Counter)
    for r in moving:
        frames_by_class[r["patch_class"]][str(r["y"])] += 1
    return {
        "settings": {k: (None if v == math.inf else v) for k, v in asdict(settings).items()},
        "odom_scale": ODOM_SCALE, "truth_tolerance_m": TRUTH_TOLERANCE_M,
        "thresholds_from_clean": thresholds, "runs": len(report),
        "frames": {"recorded": len(rows), "moving": len(moving),
                   "by_reason": dict(collections.Counter(r["reason"] for r in rows)),
                   "by_label": dict(collections.Counter(str(r["y"]) for r in moving)),
                   "by_class_and_label": {k: dict(v) for k, v in sorted(frames_by_class.items())}},
        "truth": {"crossings": dict(sorted(by_class.items())),
                  "zones_away_from_any_object": sum(e["truth"]["false_zones"] for e in report if e["condition"] == "hazard"),
                  "unacceptable_zones_away_from_any_object": sum(
                      1 for e in report if e["condition"] == "hazard" for z in e["zones"] if z["unacceptable"] and not _object_overlap(e, z)),
                  "offsets_m": {e["episode_id"]: e["offset_m"] for e in report},
                  "zones_in_clean_runs": {e["episode_id"]: len(e["zones"]) for e in report if e["condition"] == "clean"},
                  "unacceptable_zones_in_clean_runs": sum(z["unacceptable"] for e in report if e["condition"] == "clean" for z in e["zones"])},
    }


def _object_overlap(entry: dict, zone: dict) -> bool:
    return any(zone["lo"] <= c + r and zone["hi"] >= c - r for c, r in entry["placed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: <data>/labels.csv")
    parser.add_argument("--multiple", type=float, default=2.0, help="unacceptable = this multiple of the clean-floor level")
    args = parser.parse_args()
    output = args.output or args.data / "labels.csv"
    settings = replace(Settings(**JACKAL_REAL), unacceptable_vib_clean_multiple=args.multiple, unacceptable_roll_clean_multiple=args.multiple)
    runs = load_runs(args.data)
    rows, report, thresholds = build(runs, settings)
    for entry, run in zip(report, runs):
        route = run["episode"]["scenario"]["route"]["start_pose"]
        entry["placed"] = [((o["pose"]["x_m"] - route["x_m"]) * math.cos(route["yaw_rad"]) + (o["pose"]["y_m"] - route["y_m"]) * math.sin(route["yaw_rad"]),
                            o["footprint_radius_m"] + TRUTH_TOLERANCE_M) for o in run["episode"]["scenario"]["objects"]]
    columns = [*LABEL_COLUMNS, *EXTRA]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, report, thresholds, settings)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output.with_suffix(".zones.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("thresholds_from_clean", "frames", "truth")}, indent=1))


if __name__ == "__main__":
    main()
