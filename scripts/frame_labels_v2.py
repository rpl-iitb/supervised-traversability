"""Version 2 labeller: label each photo by WHERE the robot had trouble.

Agreed design (see NOTES_labelling_and_gonet_2026-10-07.md):

1. Find bad moments from the robot's own 100 Hz signals: shaking (vertical
   acceleration), slow-down (smoothed speed / commanded speed), slip (wheel
   encoder speed vs ground speed), stalls, and long emergency stops in the
   separate brake-test drive. "Abnormal" thresholds are MEASURED from the
   clean-twin drives; "unacceptable" thresholds are stated requirements
   (placeholders to confirm with the professor / ATI).
2. Turn each bad moment into a bad piece of floor along the route: from the
   front wheel contact when it started to the rear caster contact when it
   ended. Bursts closer than one robot contact span (front wheel hits, then
   casters hit) are merged first, so a crossed cable maps onto the cable.
3. Label each photo by the floor strip it shows (0.6-1.8 m ahead of the
   robot centre): touches an unacceptable zone -> 0; touches an abnormal but
   acceptable zone -> gray; robot fully drove over it with nothing abnormal
   -> 1; never reached -> gray; taken beyond the route end -> excluded.

Simulator object poses are NOT used to make labels. They are used only to
attribute a frame to a hazard class (for evaluation) and in the truth check
written to the summary.

    python3 -m scripts.frame_labels_v2 --episodes data/episodes_v2 --output data/labels_v2/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from scripts.episode_io import footprint_bbox, load_episode, world_to_base
from scripts.route_follower import RAMP_S, Route


@dataclass
class Settings:
    # --- robot geometry (physics, from the simulated robot) -----------------
    front_contact_m: float = 0.20     # front of the drive wheels on a 3-4 cm obstacle
    rear_contact_m: float = -0.29     # back of the rear casters (-0.24 - 0.045)
    drive_wheel_x_m: float = 0.10     # drive-wheel ground contact (braking happens here)
    # --- strip shown in each photo -------------------------------------------
    strip_near_m: float = 0.6         # max(camera's nearest floor 0.59, stopping distance 0.47)
    strip_far_m: float = 1.8          # near + 0.5 m/s x 2.4 s (GONet's horizon) - a CHOICE
    strip_half_width_m: float = 0.275  # outer edge of the drive wheels (attribution only)
    # --- signal processing ---------------------------------------------------
    vib_window_s: float = 0.1         # vibration RMS window (= 5 cm of travel)
    speed_smooth_s: float = 0.3       # removes sub-0.3 s speed spikes
    # "abnormal" = a level that this fraction of clean-twin DRIVES never reach
    # anywhere (MEASURED). Per-drive, not per-sample: a per-sample 99.9 % level
    # is crossed ~1.5 times per clean drive by chance.
    abnormal_run_quantile: float = 0.95
    merge_gap_m: float = 0.49         # one robot contact span: front-wheel and caster bursts merge
    # --- unacceptable = requirements (PLACEHOLDERS, to confirm) --------------
    unacceptable_vib_mps2: float = 0.3
    unacceptable_delay_ratio: float = 1.5
    unacceptable_slip: float = 0.2
    braking_budget_mps2: float = 1.0  # same deceleration as the stopping-distance budget


# --------------------------------------------------------------------------- signals

def _moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, samples)
    kernel = np.ones(samples) / samples
    padded = np.pad(values, (samples // 2, samples - 1 - samples // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def episode_signals(episode: dict, settings: Settings) -> dict:
    state = episode["state"]
    route = Route.from_manifest(episode["scenario"])
    hz = episode["meta"]["rates"]["state_hz"]
    col = lambda name: np.array([row[name] for row in state], dtype=float)
    t = col("t_s")
    progress = np.array([route.progress_and_offset(row["x_m"], row["y_m"])[0] for row in state])
    az = col("imu_az")
    vib = np.sqrt(_moving_average((az - np.median(az)) ** 2, round(settings.vib_window_s * hz)))
    smooth = round(settings.speed_smooth_s * hz)
    v = _moving_average(col("v_fwd_mps"), smooth)
    cmd = _moving_average(col("cmd_v_mps"), smooth)
    wheel = col("wheel_v_mps") if "wheel_v_mps" in state[0] else np.full(len(state), np.nan)
    wheel = _moving_average(np.nan_to_num(wheel, nan=0.0), smooth) if not np.isnan(wheel).all() else v
    valid = (t > RAMP_S + 0.5) & (cmd > 0.2)
    ratio = np.where(valid, v / np.maximum(cmd, 1e-6), 1.0)
    slip = np.where(valid, np.abs(wheel - v) / np.maximum(np.abs(v), 0.1), 0.0)
    return {"t": t, "progress": progress, "vib": np.where(valid, vib, 0.0), "ratio": ratio, "slip": slip,
            "valid": valid, "cmd": cmd, "route": route}


# --------------------------------------------------------------------------- calibration

def calibrate(clean_main: list[dict], clean_brake: list[dict], settings: Settings) -> dict:
    """'Abnormal' thresholds measured from the clean-twin drives (per drive)."""
    vib_max, ratio_min, slip_max = [], [], []
    for episode in clean_main:
        s = episode_signals(episode, settings)
        v = s["valid"]
        vib_max.append(s["vib"][v].max()), ratio_min.append(s["ratio"][v].min()), slip_max.append(s["slip"][v].max())
    q = settings.abnormal_run_quantile
    stop_max = [max(stop["distance_m"] for stop in e["meta"]["brake_test"]["stops"]) for e in clean_brake
                if e["meta"]["brake_test"]["stops"]]
    return {
        "vib_mps2": float(np.quantile(vib_max, q)),
        "speed_ratio": float(np.quantile(ratio_min, 1.0 - q)),
        "slip": float(np.quantile(slip_max, q)),
        "brake_stop_m": float(np.quantile(stop_max, q)) if stop_max else math.inf,
        "clean_drives": len(vib_max),
        "clean_brake_drives": len(stop_max),
    }


# --------------------------------------------------------------------------- zones

def drive_zones(signals: dict, thresholds: dict, termination: dict, settings: Settings) -> list[dict]:
    p, valid = signals["progress"], signals["valid"]
    abnormal = valid & ((signals["vib"] > thresholds["vib_mps2"]) | (signals["ratio"] < thresholds["speed_ratio"])
                        | (signals["slip"] > thresholds["slip"]))
    bursts, start = [], None
    for i, flag in enumerate(abnormal):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            bursts.append([start, i - 1]); start = None
    if start is not None:
        bursts.append([start, len(abnormal) - 1])
    merged: list[list[int]] = []
    for burst in bursts:
        if merged and p[burst[0]] - p[merged[-1][1]] <= settings.merge_gap_m:
            merged[-1][1] = burst[1]
        else:
            merged.append(list(burst))
    zones = []
    for a, b in merged:
        lo, hi = p[a] + settings.front_contact_m, p[b] + settings.rear_contact_m
        if hi < lo:  # a single short burst: could be under the wheels or the casters
            lo, hi = p[a] + settings.rear_contact_m, p[b] + settings.front_contact_m
        window = (p >= lo - settings.front_contact_m) & (p <= hi - settings.rear_contact_m) & valid
        delay, crossing_vib = 1.0, float(signals["vib"][a:b + 1].max())
        if window.sum() > 2:
            idx = np.flatnonzero(window)
            travelled = p[idx[-1]] - p[idx[0]]
            expected = travelled / max(float(np.mean(signals["cmd"][idx])), 1e-6)
            delay = float((signals["t"][idx[-1]] - signals["t"][idx[0]]) / max(expected, 1e-6)) if travelled > 0.02 else 1.0
            # Severity uses vibration RMS over the whole crossing (how the
            # unacceptable level is defined); 0.1 s windows only localise.
            crossing_vib = float(np.sqrt(np.mean(signals["vib"][idx] ** 2)))
        stats = {"crossing_vib": round(crossing_vib, 3), "peak_vib": float(signals["vib"][a:b + 1].max()), "min_speed_ratio": float(signals["ratio"][a:b + 1].min()),
                 "max_slip": float(signals["slip"][a:b + 1].max()), "delay_ratio": round(delay, 3)}
        causes = [name for name, bad in (("vibration", stats["crossing_vib"] >= settings.unacceptable_vib_mps2),
                                         ("delay", delay >= settings.unacceptable_delay_ratio),
                                         ("slip", stats["max_slip"] >= settings.unacceptable_slip)) if bad]
        zones.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3), "source": "drive",
                      "unacceptable": bool(causes), "cause": "+".join(causes) or "abnormal", **stats})
    if termination.get("reason") in ("stalled", "timeout"):
        stop = float(p[-1]) + settings.front_contact_m
        zones.append({"lo": round(stop, 3), "hi": round(stop + 0.10, 3), "source": "drive", "unacceptable": True,
                      "cause": termination["reason"]})
    return zones


def brake_zones(brake_episodes: list[dict], thresholds: dict, settings: Settings) -> list[dict]:
    zones = []
    for stop in (s for e in brake_episodes for s in e["meta"]["brake_test"]["stops"]):
        if stop["distance_m"] <= thresholds["brake_stop_m"]:
            continue
        budget = stop["start_speed_mps"] ** 2 / (2.0 * settings.braking_budget_mps2)
        zones.append({"lo": round(stop["start_progress_m"] + settings.drive_wheel_x_m, 3),
                      "hi": round(stop["stop_progress_m"] + settings.drive_wheel_x_m, 3), "source": "brake",
                      "unacceptable": stop["distance_m"] > budget,
                      "cause": "long_stop" if stop["distance_m"] > budget else "abnormal_stop",
                      "stop_m": stop["distance_m"], "budget_m": round(budget, 4)})
    return zones


# --------------------------------------------------------------------------- photos

def _strip_hazard(pose: dict, hazards: list[dict], settings: Settings):
    best = None
    for item in hazards:
        forward, left = world_to_base(pose, item["pose"]["x_m"], item["pose"]["y_m"])
        r = item["footprint_radius_m"]
        if settings.strip_near_m - r <= forward <= settings.strip_far_m + r and abs(left) <= settings.strip_half_width_m + r:
            if best is None or forward < best[1]:
                best = (item, forward)
    return best


def label_episode(episode: dict, zones: list[dict], settings: Settings) -> list[dict]:
    meta, scenario = episode["meta"], episode["scenario"]
    signals = episode_signals(episode, settings)
    route: Route = signals["route"]
    reached = float(signals["progress"].max()) + settings.rear_contact_m
    hazards = [o for o in scenario.get("objects", []) if o.get("role", "target_hazard") == "target_hazard"]
    rows = []
    for frame in episode["frames"]:
        pose = {"x_m": frame["x_m"], "y_m": frame["y_m"], "yaw_rad": frame["yaw_rad"]}
        s, _ = route.progress_and_offset(frame["x_m"], frame["y_m"])
        lo, hi = s + settings.strip_near_m, s + settings.strip_far_m
        row = {"episode_id": meta["episode_id"], "scenario_id": scenario["scenario_id"], "seed": scenario["seed"],
               "condition": meta.get("condition", ""), "frame_idx": frame["frame_idx"], "t_s": frame["t_s"],
               "rgb_path": str(Path(meta["episode_id"]) / frame["rgb_path"]), "strip_lo_m": round(lo, 3),
               "strip_hi_m": round(hi, 3), "patch_class": "clean_floor", "hazard_instance": "", "hazard_forward_m": "",
               "bbox": "", "tracking": "", "vibration_rms": "", "peak_contact_n": "", "zone_cause": ""}
        hit = _strip_hazard(pose, hazards, settings)
        if hit is not None:
            item, forward = hit
            row.update(patch_class=item["asset_key"], hazard_instance=item["instance_id"], hazard_forward_m=round(forward, 3))
            bbox = footprint_bbox(meta["camera"], pose, item)
            row["bbox"] = " ".join(str(v) for v in bbox) if bbox else ""
        window = (signals["progress"] >= lo - settings.front_contact_m) & (signals["progress"] <= hi - settings.rear_contact_m) & signals["valid"]
        if window.any():
            row["tracking"] = round(float(signals["ratio"][window].mean()), 3)
            row["vibration_rms"] = round(float(np.sqrt(np.mean(signals["vib"][window] ** 2))), 3)
        touching = [z for z in zones if z["lo"] <= hi and z["hi"] >= lo]
        if s > route.length_m:
            y, reason = None, "beyond_route"
        elif any(z["unacceptable"] for z in touching):
            y, reason = 0, "unacceptable"
            row["zone_cause"] = "+".join(sorted({z["cause"] for z in touching if z["unacceptable"]}))
        elif touching:
            y, reason = None, "abnormal_but_acceptable"
            row["zone_cause"] = "+".join(sorted({z["cause"] for z in touching}))
        elif reached >= hi:
            y, reason = 1, "normal"
        else:
            y, reason = None, "not_reached"
        rows.append({**row, "y": "" if y is None else y, "reason": reason})
    return rows


# --------------------------------------------------------------------------- truth check

def truth_check(episode: dict, zones: list[dict], tolerance_m: float = 0.10) -> dict:
    """Compare zones with the simulator's real object positions (evaluation only)."""
    route = Route.from_manifest(episode["scenario"])
    hazards = []
    for item in episode["scenario"].get("objects", []):
        if item.get("role", "target_hazard") != "target_hazard":
            continue
        centre, offset = route.progress_and_offset(item["pose"]["x_m"], item["pose"]["y_m"])
        r = item["footprint_radius_m"]
        hazards.append({"asset_key": item["asset_key"], "lo": centre - r, "hi": centre + r, "centre": centre})
    out = {"hazards": [], "false_zones": 0, "zones": len(zones)}
    for h in hazards:
        overlapping = [z for z in zones if z["lo"] <= h["hi"] + tolerance_m and z["hi"] >= h["lo"] - tolerance_m]
        out["hazards"].append({"asset_key": h["asset_key"], "found": bool(overlapping),
                               "unacceptable": any(z["unacceptable"] for z in overlapping),
                               "centre_error_m": round(min((abs((z["lo"] + z["hi"]) / 2 - h["centre"]) for z in overlapping), default=float("nan")), 3)})
    for z in zones:
        if not any(z["lo"] <= h["hi"] + tolerance_m and z["hi"] >= h["lo"] - tolerance_m for h in hazards):
            out["false_zones"] += 1
    return out


# --------------------------------------------------------------------------- main

LABEL_COLUMNS = ("episode_id", "scenario_id", "seed", "condition", "frame_idx", "t_s", "rgb_path", "strip_lo_m",
                 "strip_hi_m", "patch_class", "hazard_instance", "hazard_forward_m", "bbox", "tracking", "vibration_rms",
                 "peak_contact_n", "zone_cause", "y", "reason")


def build(episodes_dir: Path, settings: Settings):
    main, brake = {}, {}
    for path in sorted(p for p in episodes_dir.iterdir() if (p / "meta.json").exists()):
        episode = load_episode(path)
        key = episode["meta"]["episode_id"].split("-braketest")[0]
        if "brake_test" in episode["meta"]:
            brake.setdefault(key, []).append(episode)
        else:
            main[key] = episode
    clean_ids = [k for k, e in main.items() if e["meta"].get("condition") == "clean"]
    thresholds = calibrate([main[k] for k in clean_ids], [b for k in clean_ids for b in brake.get(k, [])], settings)
    rows, report = [], []
    for key, episode in main.items():
        zones = drive_zones(episode_signals(episode, settings), thresholds, episode["meta"]["termination"], settings)
        zones += brake_zones(brake.get(key, []), thresholds, settings)
        rows.extend(label_episode(episode, zones, settings))
        report.append({"episode_id": key, "condition": episode["meta"].get("condition"), "zones": zones,
                       "truth": truth_check(episode, zones)})
    return rows, report, thresholds


def summarise(rows, report, thresholds, settings) -> dict:
    counts = lambda key, subset: dict(sorted(__import__("collections").Counter(str(r[key]) for r in subset).items()))
    hazard_rows = [h for e in report if e["condition"] == "hazard" for h in e["truth"]["hazards"]]
    by_class = {}
    for h in hazard_rows:
        c = by_class.setdefault(h["asset_key"], {"n": 0, "found": 0, "unacceptable": 0, "centre_errors": []})
        c["n"] += 1; c["found"] += h["found"]; c["unacceptable"] += h["unacceptable"]
        if h["found"]:
            c["centre_errors"].append(h["centre_error_m"])
    for c in by_class.values():
        errors = c.pop("centre_errors")
        c["median_centre_error_m"] = round(float(np.median(errors)), 3) if errors else None
    labelled = [r for r in rows if r["reason"] != "beyond_route"]
    return {
        "settings": asdict(settings), "thresholds_measured_from_clean": thresholds,
        "frames_total": len(rows), "frames_on_route": len(labelled),
        "label_counts_on_route": counts("y", labelled), "reason_counts_on_route": counts("reason", labelled),
        "gray_fraction_on_route": round(sum(1 for r in labelled if r["y"] == "") / max(len(labelled), 1), 4),
        "labelled_by_class": {k: counts("y", [r for r in labelled if r["patch_class"] == k])
                              for k in sorted({r["patch_class"] for r in labelled})},
        "truth_check": {"hazards_by_class": by_class,
                        "false_zones_in_hazard_runs": sum(e["truth"]["false_zones"] for e in report if e["condition"] == "hazard"),
                        "zones_in_clean_runs": sum(e["truth"]["zones"] for e in report if e["condition"] == "clean")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abnormal-run-quantile", type=float, default=Settings.abnormal_run_quantile)
    parser.add_argument("--strip-far-m", type=float, default=Settings.strip_far_m)
    parser.add_argument("--unacceptable-scale", type=float, default=1.0,
                        help="multiply the vibration/slip thresholds and (ratio-1) of the delay threshold (sensitivity)")
    args = parser.parse_args()
    settings = Settings(abnormal_run_quantile=args.abnormal_run_quantile, strip_far_m=args.strip_far_m)
    settings.unacceptable_vib_mps2 *= args.unacceptable_scale
    settings.unacceptable_slip *= args.unacceptable_scale
    settings.unacceptable_delay_ratio = 1.0 + (settings.unacceptable_delay_ratio - 1.0) * args.unacceptable_scale
    rows, report, thresholds = build(args.episodes, settings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, report, thresholds, settings)
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    args.output.with_suffix(".zones.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("frames_on_route", "label_counts_on_route", "gray_fraction_on_route",
                                              "thresholds_measured_from_clean", "truth_check")}, indent=2))


if __name__ == "__main__":
    main()
