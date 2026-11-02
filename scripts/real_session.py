"""Lab-day tool: convert a session's bags, label them, and check the labels landed on the hazards.

One command after copying the bags off the robot::

    python3 -m scripts.real_session --bags data/real_bags/session1 \\
        --camera-mount 0.20 0 0.30 --camera-pitch-deg 15 --out data/real/session1

The first run without ``drives.csv`` writes a template listing every bag in
``--bags`` (condition guessed from the name: ``clean_*`` is clean, anything
else is a hazard drive). Fill in where each hazard lay, then run it again::

    bag,condition,hazards,note
    clean_01_20261022_101500,clean,,
    cable_01_20261022_102000,hazard,cable:3.0:0.01:0.02,"2 cm cable"

``hazards`` is ``kind:distance_m[:radius_m[:height_m]]`` measured along the
taped line from the start mark, several separated by ``;``. These positions
are used only for the truth check, never for the labels.

Writes ``out/episodes/<bag>/``, ``out/labels.csv`` (+ ``.summary.json``,
``.zones.json``), one quick-look plot per drive in ``out/quicklook/`` and
prints a per-drive table: what was found, where, and how far from the tape.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.episode_io import load_episode
from scripts.frame_labels_v2 import JACKAL, LABEL_COLUMNS, Settings, build, episode_signals, summarise
from scripts.rosbag_to_episode import convert


def find_bags(bags_dir: Path) -> list[Path]:
    """ROS 2 bag directories (with metadata.yaml) and ROS 1 .bag files directly under ``bags_dir``."""
    found = [p for p in bags_dir.iterdir() if (p.is_dir() and (p / "metadata.yaml").exists()) or p.suffix == ".bag"]
    return sorted(found)


def write_template(bags: list[Path], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bag", "condition", "hazards", "note"])
        for bag in bags:
            writer.writerow([bag.name, "clean" if bag.name.lower().startswith("clean") else "hazard", "", ""])


def read_drives(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        drives = [row for row in csv.DictReader(file) if row.get("bag")]
    for row in drives:
        if row["condition"] not in ("clean", "hazard"):
            raise SystemExit(f"{path}: condition must be clean or hazard, got {row['condition']!r} for {row['bag']}")
        row["hazard_list"] = [h.strip() for h in (row.get("hazards") or "").split(";") if h.strip()]
        if row["condition"] == "hazard" and not row["hazard_list"]:
            print(f"warning: {row['bag']} is a hazard drive with no measured hazard; it gets labels but no truth check")
    if not any(row["condition"] == "clean" for row in drives):
        raise SystemExit("Need at least one clean drive: the 'unusual' levels are measured from clean drives.")
    return drives


def quicklook(episode_dir: Path, zones: list[dict], rows: list[dict], thresholds: dict, settings: Settings,
              output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episode = load_episode(episode_dir)
    s = episode_signals(episode, settings)
    p = s["progress"]
    hazards = [o for o in episode["scenario"].get("objects", [])]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 3, 3]})

    v = np.array([float(r["v_fwd_mps"]) for r in episode["state"]])
    axes[0].plot(p, s["cmd"], color="0.5", ls="--", lw=1.5, label="commanded")
    axes[0].plot(p, v, color="C0", lw=1, label="measured")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].plot(p, s["vib"], color="C0", lw=1)
    axes[1].axhline(thresholds["vib_mps2"], color="orange", lw=1, ls="--", label="unusual (from clean drives)")
    axes[1].axhline(settings.unacceptable_vib_mps2, color="red", lw=1, ls="--", label="unacceptable (placeholder)")
    axes[1].set_ylabel("shaking (m/s²)")
    axes[1].legend(loc="upper right", fontsize=8)
    if s["has_effort"]:  # motor current, counted only during steady straight driving
        effort_ax = axes[1].twinx()
        effort_ax.plot(p, s["effort"], color="C2", lw=1, label="motor effort (steady driving)")
        if "effort_a" in thresholds:
            effort_ax.axhline(thresholds["effort_a"], color="C2", lw=1, ls=":", label="unusual effort (from clean drives)")
        effort_ax.set_ylabel("motor current (A)")
        effort_ax.legend(loc="upper left", fontsize=8)

    for z in zones:
        for ax in axes[:2]:
            ax.axvspan(z["lo"], z["hi"], color="red" if z["unacceptable"] else "orange", alpha=0.2, lw=0)
    for ax in axes:
        for h in hazards:
            ax.axvline(h.get("measured_distance_m", np.nan), color="k", lw=1, ls=":")

    colour = {"1": "tab:green", "0": "tab:red", "": "0.7"}
    for row in rows:
        axes[2].plot([float(row["strip_lo_m"]), float(row["strip_hi_m"])], [row["frame_idx"]] * 2,
                     color=colour.get(str(row["y"]), "0.7"), lw=1.2)
    axes[2].set_ylabel("photo number")
    axes[2].set_xlabel("distance along the taped line (m)   ·   dotted = measured hazard")
    axes[2].set_title("floor each photo judged: green traversable, red not, gray unsure", fontsize=9)
    fig.suptitle(episode["meta"]["episode_id"], fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bags", type=Path, required=True, help="folder holding this session's bags")
    parser.add_argument("--drives", type=Path, help="drive list (default: <bags>/drives.csv)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera-mount", type=float, nargs=3, required=True, metavar=("FWD_M", "LEFT_M", "UP_M"))
    parser.add_argument("--camera-pitch-deg", type=float, required=True)
    parser.add_argument("--imu-frame", choices=("base", "optical"), default="base", help="'base' = the Jackal's IMU")
    parser.add_argument("--image-scale", type=float, default=0.5, help="saved resolution relative to the camera")
    parser.add_argument("--rotate-180", action="store_true", help="camera is mounted upside down (the lab's D555 is)")
    parser.add_argument("--reconvert", action="store_true", help="convert bags again even if the episode exists")
    for role in ("odom", "wheel_odom", "cmd", "imu", "image", "camera_info", "motors"):
        parser.add_argument(f"--{role.replace('_', '-')}", dest=f"topic_{role}", help=f"override the {role} topic")
    args = parser.parse_args()

    drives_path = args.drives or args.bags / "drives.csv"
    if not drives_path.exists():
        bags = find_bags(args.bags)
        if not bags:
            raise SystemExit(f"No bags found in {args.bags}")
        write_template(bags, drives_path)
        raise SystemExit(f"Wrote {drives_path} with {len(bags)} bags. Fill in conditions and hazards, then run again.")
    drives = read_drives(drives_path)

    episodes = args.out / "episodes"
    overrides = {role: getattr(args, f"topic_{role}") for role in ("odom", "wheel_odom", "cmd", "imu", "image", "camera_info", "motors")}
    for row in drives:
        out = episodes / row["bag"]
        if (out / "meta.json").exists() and not args.reconvert:
            continue
        result = convert(args.bags / row["bag"], out, episode_id=row["bag"], condition=row["condition"],
                         mount_xyz_m=args.camera_mount, pitch_down_deg=args.camera_pitch_deg, imu_frame=args.imu_frame,
                         frame_hz=10.0, image_scale=args.image_scale, hazards=row["hazard_list"], rotate_180=args.rotate_180,
                         termination="goal_reached", cmd_timeout_s=0.5, overrides=overrides)
        print(f"converted {row['bag']}: {result['frames']} photos, {result['route_length_m']} m, "
              f"nearest visible floor {result['nearest_visible_floor_m']} m")

    settings = Settings(**JACKAL)
    rows, report, thresholds = build(episodes, settings)
    labels = args.out / "labels.csv"
    with labels.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, report, thresholds, settings)
    labels.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    labels.with_suffix(".zones.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nunusual levels measured from {thresholds['clean_drives']} clean drives: "
          f"shaking {thresholds['vib_mps2']:.3f} m/s², speed ratio {thresholds['speed_ratio']:.3f}"
          + (f", motor effort {thresholds['effort_a']:.2f} A (steady clean median {thresholds['effort_baseline_a']:.2f} A)"
             if "effort_a" in thresholds else ", motor effort: no motor feedback recorded"))
    print(f"\n{'drive':<34} {'cond':<6} {'photos':>6} {'trav':>5} {'not':>5} {'gray':>5}  result")
    for entry in report:
        mine = [r for r in rows if r["episode_id"] == entry["episode_id"] and r["reason"] != "beyond_route"]
        count = lambda y: sum(1 for r in mine if str(r["y"]) == y)
        truth = entry["truth"]
        if entry["condition"] == "clean":
            bad = count("0")
            result = "OK, no false alarms" if bad == 0 else f"PROBLEM: {bad} photos marked not traversable"
        else:
            parts = [f"{h['asset_key']} {'found' if h['found'] else 'MISSED'}"
                     + (f" ({h['centre_error_m'] * 100:.0f} cm off, {'unacceptable' if h['unacceptable'] else 'only unusual'})"
                        if h["found"] else "") for h in truth["hazards"]]
            if truth["false_zones"]:
                parts.append(f"{truth['false_zones']} zone(s) away from any hazard")
            result = "; ".join(parts) or "no measured hazard"
        print(f"{entry['episode_id']:<34} {entry['condition']:<6} {len(mine):>6} {count('1'):>5} {count('0'):>5} "
              f"{count(''):>5}  {result}")
        quicklook(episodes / entry["episode_id"], entry["zones"], [r for r in rows if r["episode_id"] == entry["episode_id"]],
                  thresholds, settings, args.out / "quicklook" / f"{entry['episode_id']}.png")
    print(f"\nquick-look plots: {args.out / 'quicklook'}\nlabels: {labels}")


if __name__ == "__main__":
    main()
