# UNVERIFIED: AI-written; checked by its own output on the 30 Oct 2026 bags, not independently reviewed.
"""Assemble sessions s1a and s1c into one real-robot dataset and convert every bag.

    python3 -m scripts.real_dataset --bags /path/data/real_bags --plans /path/data/run_sheets --out /path/data/real/s1

Applies the placement corrections noted during the sessions, writes ``out/runs.csv``
(one row per run) and ``out/episodes/<run>/``. Placements stay in true metres along
the taped line; they are never used to make labels.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.rosbag_to_episode import convert

CAMERA = dict(mount_xyz_m=(0.22, 0.0, 0.26), pitch_down_deg=0.0, rotate_180=True)
LINE_M = {"s1a": 5.8, "s1c": 16.24}
PROBE = "probe_bundle_02"
# Corrections reported during the session: (run, kind) -> metres from A.
MOVED_FROM_A = {("s1a_r005", "deskpad"): 1.5, ("s1a_r006", "deskpad"): 1.5}
# Direction actually driven where it differs from the sheet (clean runs only).
DRIVEN = {"s1a_r002": "B->A", "s1c_r002": "B->A"}
CABLE_HEIGHT_M = 0.018  # the red cable was bundled from s1a run 3 on


def latest_bag(root: Path, run: str) -> Path | None:
    found = sorted(p for p in root.glob(f"{run}_*") if (p / "metadata.yaml").exists())
    return found[-1] if found else None


def plan_rows(bags: Path, plans: Path) -> list[dict]:
    rows = []
    for session in ("s1a", "s1c"):
        with (plans / session / "drives.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                run, line = row["bag"], LINE_M[session]
                bag = latest_bag(bags / session, run)
                if bag is None:
                    continue
                direction = DRIVEN.get(run, row["direction"])
                hazards = []
                for text in filter(None, row["hazards"].split(";")):
                    kind, start, radius, height = text.split(":")
                    from_a = float(start) if row["direction"] == "A->B" else line - float(start)
                    from_a = MOVED_FROM_A.get((run, kind), from_a)
                    from_start = from_a if direction == "A->B" else line - from_a
                    height = CABLE_HEIGHT_M if kind == "cable" else float(height)
                    hazards.append(f"{kind}:{from_start:.3f}:{radius}:{height}")
                rows.append({"run": run, "bag": str(bag), "session": session, "condition": row["condition"],
                             "direction": direction, "line_m": line, "hazards": ";".join(hazards)})
    probe = latest_bag(bags / "s1a", PROBE)
    if probe is not None:
        rows.append({"run": PROBE, "bag": str(probe), "session": "s1a", "condition": "hazard", "direction": "A->B",
                     "line_m": LINE_M["s1a"], "hazards": f"cable:2.000:0.01:{CABLE_HEIGHT_M}"})
    return rows


def convert_one(args: tuple[dict, Path, float]) -> str:
    row, out, scale = args
    target = out / "episodes" / row["run"]
    if (target / "meta.json").exists():
        return f"{row['run']}: already converted"
    result = convert(Path(row["bag"]), target, episode_id=row["run"], condition=row["condition"], imu_frame="base",
                     frame_hz=10.0, image_scale=scale, hazards=[h for h in row["hazards"].split(";") if h],
                     termination="goal_reached", cmd_timeout_s=0.5, overrides={}, **CAMERA)
    return f"{row['run']}: {result['frames']} frames, route {result['route_length_m']} m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bags", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    rows = plan_rows(args.bags, args.plans)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "runs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with ProcessPoolExecutor(args.workers) as pool:
        for line in pool.map(convert_one, [(r, args.out, args.image_scale) for r in rows]):
            print(line, flush=True)


if __name__ == "__main__":
    main()
