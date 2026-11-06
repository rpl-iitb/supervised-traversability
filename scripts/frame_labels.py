"""Turn recorded episodes into per-frame traversability labels.

Principle (GONet / Wild Visual Navigation style self-supervision): a frame is
labelled by what *actually happened* when the robot later drove over the
floor patch visible in it. For each frame at time ``t``:

1. The look-ahead patch is the corridor ``[lookahead_min_m, lookahead_max_m]``
   ahead of the robot, as wide as the robot.
2. The outcome window is the later interval during which the robot body
   overlaps that patch (found from the travelled arc length).
3. Speed tracking, vertical-acceleration vibration and hazard contact in that
   window give an outcome label: 1 traversable, 0 not, ``None`` ambiguous.

The simulator's knowledge of object poses is used only to *attribute* the
outcome to the hazard in the patch (for per-class evaluation), to draw its
image bounding box, to discard contaminated windows, and to provide a
separate operator-prior label. It is never silently substituted for the
measured outcome; the two are kept side by side.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scripts.episode_io import footprint_bbox, load_episode, world_to_base


# Placeholder operator knowledge: 1 = operator would drive over, 0 = avoid,
# None = no prior. These are assumptions, not an operator's actual policy.
DEFAULT_OPERATOR_PRIOR = {
    "clean_floor": 1,
    "plastic_wrapper": 1,
    "cable_hose": 0,
    "small_lip": 0,
    "oil_spill": 0,
}


@dataclass
class LabelConfig:
    lookahead_min_m: float = 0.8
    lookahead_max_m: float = 1.8
    corridor_half_width_m: float = 0.30
    robot_front_m: float = 0.40
    robot_rear_m: float = 0.35
    positive_tracking: float = 0.80
    negative_tracking: float = 0.50
    vibration_soft: float | None = None
    vibration_hard: float | None = None
    min_bbox_area_px: float = 30.0
    min_command_mps: float = 0.05
    policy: str = "outcome"
    operator_prior: dict = field(default_factory=lambda: dict(DEFAULT_OPERATOR_PRIOR))


class _StateIndex:
    """Arrays and cumulative arc length for fast window queries."""

    def __init__(self, state_rows: list[dict]):
        if not state_rows:
            raise ValueError("Episode has no state samples")
        self.rows = state_rows
        self.t = [row["t_s"] for row in state_rows]
        self.s = [0.0]
        for previous, current in zip(state_rows, state_rows[1:]):
            self.s.append(self.s[-1] + math.hypot(current["x_m"] - previous["x_m"], current["y_m"] - previous["y_m"]))
        self.az_ref = statistics.median(row["imu_az"] for row in state_rows)

    def arc_length_at(self, t_s: float) -> float:
        index = min(max(bisect.bisect_left(self.t, t_s), 0), len(self.t) - 1)
        return self.s[index]

    def window(self, t_start: float, s_begin: float, s_end: float) -> list[int]:
        """Indices after ``t_start`` whose arc length lies in ``[s_begin, s_end]``."""
        first = bisect.bisect_left(self.t, t_start)
        return [i for i in range(first, len(self.rows)) if s_begin <= self.s[i] <= s_end]


def window_metrics(index: _StateIndex, indices: list[int], config: LabelConfig) -> dict | None:
    rows = [index.rows[i] for i in indices]
    commanded = [row["cmd_v_mps"] for row in rows]
    if not rows or statistics.fmean(commanded) < config.min_command_mps:
        return None
    vertical = [row["imu_az"] - index.az_ref for row in rows]
    return {
        "tracking": round(statistics.fmean(row["v_fwd_mps"] for row in rows) / statistics.fmean(commanded), 3),
        "vibration_rms": round(math.sqrt(statistics.fmean(v * v for v in vertical)), 3),
        "peak_contact_n": round(max(row["hazard_contact_n"] for row in rows), 2),
        "duration_s": round(rows[-1]["t_s"] - rows[0]["t_s"], 3),
    }


def outcome_label(metrics: dict, config: LabelConfig) -> tuple[int | None, str]:
    """Conservative outcome rule; borderline windows stay unlabelled."""
    if metrics["tracking"] <= config.negative_tracking:
        return 0, "poor_speed_tracking"
    if config.vibration_hard is not None and metrics["vibration_rms"] >= config.vibration_hard:
        return 0, "severe_vibration"
    smooth = config.vibration_soft is None or metrics["vibration_rms"] <= config.vibration_soft
    if metrics["tracking"] >= config.positive_tracking and smooth:
        return 1, "normal_traversal"
    return None, "ambiguous_outcome"


def _target_hazards(scenario: dict) -> list[dict]:
    return [item for item in scenario.get("objects", []) if item.get("role", "target_hazard") == "target_hazard"]


def _hazard_in_patch(frame_pose: dict, hazards: list[dict], config: LabelConfig):
    best = None
    for item in hazards:
        forward, left = world_to_base(frame_pose, item["pose"]["x_m"], item["pose"]["y_m"])
        radius = item["footprint_radius_m"]
        in_patch = (
            config.lookahead_min_m - radius <= forward <= config.lookahead_max_m + radius
            and abs(left) <= config.corridor_half_width_m + radius
        )
        if in_patch and (best is None or forward < best[1]):
            best = (item, forward)
    return best


def _hazards_overlapping_robot(index: _StateIndex, indices: list[int], hazards: list[dict], config: LabelConfig) -> set[str]:
    touched = set()
    for i in indices:
        row = index.rows[i]
        pose = {"x_m": row["x_m"], "y_m": row["y_m"], "yaw_rad": row["yaw_rad"]}
        for item in hazards:
            forward, left = world_to_base(pose, item["pose"]["x_m"], item["pose"]["y_m"])
            radius = item["footprint_radius_m"]
            if -config.robot_rear_m - radius <= forward <= config.robot_front_m + radius and abs(left) <= config.corridor_half_width_m + radius:
                touched.add(item["instance_id"])
    return touched


def label_episode(episode: dict, config: LabelConfig) -> list[dict]:
    meta, scenario, camera = episode["meta"], episode["scenario"], episode["meta"]["camera"]
    index = _StateIndex(episode["state"])
    hazards = _target_hazards(scenario)
    stalled = meta.get("termination", {}).get("reason") == "stalled"
    records = []
    for frame in episode["frames"]:
        pose = {"x_m": frame["x_m"], "y_m": frame["y_m"], "yaw_rad": frame["yaw_rad"]}
        s_frame = index.arc_length_at(frame["t_s"])
        s_begin = s_frame + config.lookahead_min_m - config.robot_front_m
        s_end = s_frame + config.lookahead_max_m + config.robot_rear_m
        record = {
            "episode_id": meta["episode_id"],
            "scenario_id": scenario["scenario_id"],
            "seed": scenario["seed"],
            "condition": meta.get("condition", ""),
            "frame_idx": frame["frame_idx"],
            "t_s": frame["t_s"],
            "rgb_path": str(Path(meta["episode_id"]) / frame["rgb_path"]),
            "patch_class": "clean_floor",
            "hazard_instance": "",
            "hazard_forward_m": "",
            "bbox": "",
            "tracking": "",
            "vibration_rms": "",
            "peak_contact_n": "",
        }
        hit = _hazard_in_patch(pose, hazards, config)
        if hit is not None:
            item, forward = hit
            record.update(patch_class=item["asset_key"], hazard_instance=item["instance_id"], hazard_forward_m=round(forward, 3))
            bbox = footprint_bbox(camera, pose, item)
            if bbox is not None:
                record["bbox"] = " ".join(str(v) for v in bbox)
            if bbox is None or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < config.min_bbox_area_px:
                records.append(_finish(record, None, "hazard_not_visible", config))
                continue

        if index.s[-1] < s_end:
            blockers = _hazards_overlapping_robot(index, [len(index.rows) - 1], hazards, config) - {record["hazard_instance"]}
            if stalled and blockers:
                records.append(_finish(record, None, "stalled_on_other_hazard", config))
            elif stalled and index.s[-1] >= s_begin - 0.2:
                records.append(_finish(record, 0, "stalled_before_or_in_patch", config))
            else:
                records.append(_finish(record, None, "patch_not_covered", config))
            continue

        indices = index.window(frame["t_s"], s_begin, s_end)
        metrics = window_metrics(index, indices, config)
        if metrics is None:
            records.append(_finish(record, None, "no_command_in_window", config))
            continue
        record.update(tracking=metrics["tracking"], vibration_rms=metrics["vibration_rms"], peak_contact_n=metrics["peak_contact_n"])
        touched = _hazards_overlapping_robot(index, indices, hazards, config)
        if touched - {record["hazard_instance"]}:
            records.append(_finish(record, None, "window_contaminated_by_other_hazard", config))
            continue
        label, reason = outcome_label(metrics, config)
        records.append(_finish(record, label, reason, config))
    return records


def _finish(record: dict, outcome: int | None, reason: str, config: LabelConfig) -> dict:
    prior = config.operator_prior.get(record["patch_class"])
    if record["patch_class"] != "clean_floor" and reason == "hazard_not_visible":
        prior = None
    final = outcome if config.policy == "outcome" else prior
    return {**record, "y": "" if final is None else final, "y_outcome": "" if outcome is None else outcome,
            "y_prior": "" if prior is None else prior, "reason": reason}


def hazard_encounters(episode: dict, config: LabelConfig) -> list[dict]:
    """Per-hazard crossing summary: the object-level record."""
    index = _StateIndex(episode["state"])
    results = []
    for item in _target_hazards(episode["scenario"]):
        overlapping = [
            i for i in range(len(index.rows))
            if item["instance_id"] in _hazards_overlapping_robot(index, [i], [item], config)
        ]
        summary = {"instance_id": item["instance_id"], "asset_key": item["asset_key"], "reached": bool(overlapping)}
        if overlapping:
            metrics = window_metrics(index, overlapping, config)
            if metrics is not None:
                summary.update(metrics)
                summary["t_enter_s"] = round(index.t[overlapping[0]], 3)
                summary["outcome_label"], summary["outcome_reason"] = outcome_label(metrics, config)
        else:
            summary["outcome_label"], summary["outcome_reason"] = (
                (0, "stalled_before_hazard") if episode["meta"].get("termination", {}).get("reason") == "stalled" else (None, "not_reached")
            )
        results.append(summary)
    return results


MIN_VIBRATION_THRESHOLD = 0.05  # m/s^2 RMS; guards against a noise-free simulator
# Assumed payload-disturbance limit (~0.03 g RMS), a placeholder; the soft threshold, by contrast, is calibrated from clean-floor data.
DEFAULT_VIBRATION_HARD = 0.3


def calibrate_vibration(clean_episodes: list[dict], config: LabelConfig, quantile: float = 0.99) -> float:
    """Soft vibration threshold = high quantile of clean-floor windows (floored)."""
    values = []
    for episode in clean_episodes:
        for record in label_episode(episode, config):
            if record["vibration_rms"] != "":
                values.append(float(record["vibration_rms"]))
    if not values:
        raise ValueError("No clean-floor windows available for vibration calibration")
    values.sort()
    return round(max(MIN_VIBRATION_THRESHOLD, values[min(len(values) - 1, int(quantile * len(values)))]), 3)


LABEL_COLUMNS = (
    "episode_id", "scenario_id", "seed", "condition", "frame_idx", "t_s", "rgb_path", "patch_class",
    "hazard_instance", "hazard_forward_m", "bbox", "tracking", "vibration_rms", "peak_contact_n",
    "y", "y_outcome", "y_prior", "reason",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, required=True, help="Directory containing episode directories")
    parser.add_argument("--output", type=Path, required=True, help="labels.csv to write")
    parser.add_argument("--policy", choices=("outcome", "operator_prior"), default="outcome")
    parser.add_argument("--vibration-hard", type=float, default=DEFAULT_VIBRATION_HARD,
                        help="Vertical-acceleration RMS (m/s^2) treated as an unacceptable jolt")
    args = parser.parse_args()

    episodes = [load_episode(path) for path in sorted(p for p in args.episodes.iterdir() if (p / "meta.json").exists())]
    config = LabelConfig(policy=args.policy)
    clean = [episode for episode in episodes if episode["meta"].get("condition") == "clean"]
    if clean:
        config.vibration_soft = calibrate_vibration(clean, config)
    config.vibration_hard = args.vibration_hard

    records, encounters = [], []
    for episode in episodes:
        records.extend(label_episode(episode, config))
        for encounter in hazard_encounters(episode, config):
            encounters.append({"episode_id": episode["meta"]["episode_id"], **encounter})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "episodes": len(episodes),
        "frames": len(records),
        "config": asdict(config),
        "label_counts": _counts(records, "y"),
        "reason_counts": _counts(records, "reason"),
        "class_counts": _counts([r for r in records if r["y"] != ""], "patch_class"),
        "prior_vs_outcome_disagreements": sum(
            1 for r in records if r["y_outcome"] != "" and r["y_prior"] != "" and r["y_outcome"] != r["y_prior"]
        ),
        "hazard_encounters": encounters,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("episodes", "frames", "label_counts", "class_counts")}, indent=2))


def _counts(records: list[dict], key: str) -> dict:
    counts: dict = {}
    for record in records:
        counts[str(record[key])] = counts.get(str(record[key]), 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
