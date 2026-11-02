"""Plan a real-robot data session, then match the recorded bags back to the plan.

``make`` writes a printable run sheet (what to place where, which way to drive)
and a ``drives.csv`` already filled in, so nothing is typed after the session::

    python3 -m scripts.run_sheet make --session s1 --location eyantra --out data/run_sheets/s1

``match`` runs after the bags are copied off the robot. ``record_drive.sh``
appends a timestamp to each name (``s1_r007_20261029_101500``), so this maps
every planned run to its bag folder and writes ``<bags>/drives.csv`` in the
form ``scripts.real_session`` reads::

    python3 -m scripts.run_sheet match --plan data/run_sheets/s1/drives.csv --bags data/real_bags/s1

Design (see hardware/SESSION1_PLAN.md):

* A taped straight line ``--line-m`` long with start marks at both ends (A, B).
  The robot's centre starts on the start mark.
* Each hazard run carries one hazard per slot (``--slots``, default about 3 m and
  6 m from end A), all different. Slots are far enough apart (> look-ahead + robot
  length) that one hazard's shaking is never blamed on another.
* ``--objects`` adds your own objects from a JSON list of
  ``{"kind", "what", "along_m", "height_m"}``, usable in ``--kinds``.
* The hazard types are balanced over slots, the angle (straight across or
  diagonal) and the direction (A→B or B→A, which shows a new background).
* Clean runs calibrate "normal" on this floor. Two come first, the rest are
  spread through the session to catch drift (lighting, battery).
* The run order keeps one hazard in place between consecutive runs where
  possible, so each reset is quick.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

TRACK_M = 0.43  # Jackal outer track: the floor width the wheels sweep


@dataclass(frozen=True)
class Hazard:
    kind: str
    what: str            # what to put down, in words
    along_m: float       # size along the driving line when placed straight across
    height_m: float


HAZARDS = {
    "cable_thin": Hazard("cable_thin", "thin cable, about 1 cm (e.g. a phone or LAN cable)", 0.01, 0.01),
    "cable_thick": Hazard("cable_thick", "thick cable, 2-3 cm (e.g. an extension cord, or two bundled)", 0.025, 0.025),
    "mat": Hazard("mat", "rubber mat or doormat, about 40 cm along the line", 0.40, 0.01),
    "sheet": Hazard("sheet", "DISTRACTOR: flat plastic sheet, paper or bright floor tape, about 40 cm along the line, "
                    "taped flat. Looks unusual but is harmless: the model must learn NOT to flag it", 0.40, 0.001),
    "lip": Hazard("lip", "wooden plank or board, under 3 cm thick (the Jackal has about 6.5 cm clearance)", 0.15, 0.02),
    "debris": Hazard("debris", "small solid object, 2-4 cm high: a plug or connector, a wooden block, a small tool "
                     "handle (one item, or two or three within about 30 cm). Cheap things only, not lab equipment; "
                     "nothing sharp; under 4 cm so the robot can climb it", 0.30, 0.03),
    "bump": Hazard("bump", "rubber cable protector or a garden hose, 2-3 cm high", 0.08, 0.025),
    # Session 1 objects (photos 30 Oct; sizes estimated from the photos, not measured)
    "cable": Hazard("cable", "red extension-cord cable, single strand (about 7 mm), across the line; "
                    "the white socket box stays at least 1 m to the side of the line, same spot in EVERY run", 0.01, 0.007),
    "lid": Hazard("lid", "plastic box lid (about 12 cm square), rim down, taped at two corners, centred on a WHEEL TRACK (19 cm left or right of the line, alternate sides); on the line it passes under the chassis untouched", 0.12, 0.015),
    "pouch": Hazard("pouch", "green pencil pouch with its contents, taped at both ends, centred on a WHEEL TRACK (19 cm left or right of the line, alternate sides); on the line it passes under the chassis untouched",
                    0.10, 0.035),
    "deskpad": Hazard("deskpad", "DISTRACTOR: felt desk pad, flattened, all four edges taped. Looks unusual but is "
                      "harmless: the model must learn NOT to flag it", 0.30, 0.003),
}
# Default set: clearly bad (thick cable, lip, debris), borderline (thin cable), and a look-alike that is harmless
# (sheet). mat and bump are extras for --kinds.
DEFAULT_KINDS = ("cable_thin", "cable_thick", "lip", "debris", "sheet")


def along_line_radius(h: Hazard, angle_deg: float) -> float:
    """Half the hazard's extent along the line, as the truth check uses it.

    A long thin object crossed at an angle meets the left and right wheels at
    different points along the line: the extra length is track * tan(angle).
    """
    return round(h.along_m / 2 + TRACK_M * math.tan(math.radians(angle_deg)) / 2, 3)


def plan_runs(*, session: str, location: str, kinds: list[str], hazard_runs: int, clean_runs: int, line_m: float,
              slots: tuple[tuple[float, float], ...], angles: list[float], script_fraction: float,
              seed: int) -> list[dict]:
    if len(set(kinds)) < len(slots):
        raise ValueError(f"{len(slots)} slots need at least {len(slots)} different kinds, got {len(set(kinds))}")
    rng = random.Random(seed)
    # Balanced assignment: each kind fills each slot equally often (within one).
    n = hazard_runs
    slot_kinds = []
    for s in range(len(slots)):
        pool = [kinds[i % len(kinds)] for i in range(n)]
        rng.shuffle(pool)
        slot_kinds.append(pool)
    # Avoid the same kind twice in a run where a swap within the later slot fixes it.
    for t in range(1, len(slots)):
        earlier = lambda i: [slot_kinds[e][i] for e in range(t)]  # noqa: E731
        for i in range(n):
            if slot_kinds[t][i] in earlier(i):
                for j in range(n):
                    if slot_kinds[t][j] not in earlier(i) and slot_kinds[t][i] not in earlier(j):
                        slot_kinds[t][i], slot_kinds[t][j] = slot_kinds[t][j], slot_kinds[t][i]
                        break
    pairs = list(zip(*slot_kinds))

    # Order: greedy, keep one hazard in place from run to run where possible.
    ordered, remaining = [pairs.pop(0)], pairs
    while remaining:
        prev = ordered[-1]
        same = [p for p in remaining if any(a == b for a, b in zip(p, prev))]
        pick = same[0] if same else remaining[0]
        remaining.remove(pick)
        ordered.append(pick)

    runs = []
    for i, kinds_in_run in enumerate(ordered):
        hazards = []
        for slot, kind in zip(slots, kinds_in_run):
            angle = angles[rng.randrange(len(angles))]
            hazards.append({"kind": kind, "from_a_m": round(rng.uniform(*slot), 1), "angle_deg": angle})
        runs.append({"condition": "hazard", "hazards": hazards, "direction": "A->B" if i % 2 == 0 else "B->A"})

    # Clean runs: two at the start, the rest spread evenly through the session.
    later = max(clean_runs - 2, 0)
    positions = sorted({round((k + 1) * len(runs) / (later + 1)) for k in range(later)})
    for offset, pos in enumerate(positions):
        runs.insert(pos + offset, {"condition": "clean", "hazards": [], "direction": "A->B" if offset % 2 else "B->A"})
    for _ in range(min(2, clean_runs)):
        runs.insert(0, {"condition": "clean", "hazards": [], "direction": "A->B"})

    n_script = round(script_fraction * len(runs))
    script_idx = set(rng.sample(range(len(runs)), n_script)) if n_script else set()
    for i, run in enumerate(runs):
        run["run_id"] = f"{session}_r{i + 1:03d}"
        run["driver"] = "script" if i in script_idx else "teleop"
        run["location"] = location
        for h in run["hazards"]:
            h["from_start_m"] = h["from_a_m"] if run["direction"] == "A->B" else round(line_m - h["from_a_m"], 1)
            h["radius_m"] = along_line_radius(HAZARDS[h["kind"]], h["angle_deg"])
            h["height_m"] = HAZARDS[h["kind"]].height_m
    return runs


def hazard_field(run: dict) -> str:
    return ";".join(f"{h['kind']}:{h['from_start_m']}:{h['radius_m']}:{h['height_m']}" for h in run["hazards"])


def describe(h: dict) -> str:
    angle = "straight across" if h["angle_deg"] == 0 else f"diagonal {h['angle_deg']:.0f}°"
    return f"{h['kind']} at **{h['from_a_m']} m from A**, {angle}"


def write_plan(runs: list[dict], out: Path, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "drives.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["bag", "condition", "hazards", "note", "direction", "driver", "location"])
        for run in runs:
            note = " + ".join(f"{h['kind']} {'diag' if h['angle_deg'] else 'across'}" for h in run["hazards"]) or "clean"
            writer.writerow([run["run_id"], run["condition"], hazard_field(run), note, run["direction"], run["driver"],
                             run["location"]])
    (out / "plan.json").write_text(json.dumps({"args": vars(args), "runs": runs}, indent=2, default=str) + "\n")

    counts: dict = {}
    for run in runs:
        for h in run["hazards"]:
            counts[h["kind"]] = counts.get(h["kind"], 0) + 1
    lines = [
        f"# Run sheet: session {args.session}, {args.location}",
        "",
        f"{len(runs)} runs: {sum(r['condition'] == 'clean' for r in runs)} clean, "
        f"{sum(r['condition'] == 'hazard' for r in runs)} with {len(next(r for r in runs if r['hazards'])['hazards'])} hazards each. Crossings per type: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ".",
        "",
        "## Set up once",
        "",
        f"- Tape a straight line **{args.line_m} m** long. Mark end **A** and end **B**, and tape a tick every 0.5 m measured from A.",
        "- Robot's centre (middle of the wheels) on the start mark, pointing along the line.",
        "- Measure the camera once: forward / left / up of the robot centre on the floor (m) and its downward tilt (deg).",
        "  Write them here: fwd ____ left ____ up ____ tilt ____",
        "- Hazards to bring: " + "; ".join(f"**{h.kind}**: {h.what}" for k, h in HAZARDS.items() if k in counts) + ".",
        "- Driving: teleop = hold **L1**, stick fully forward (0.4 m/s), straight, all the way to the far end of the line.",
        "  Script = the scripted straight driver (see demo/RUNNING_ON_ROBOT.md) for the same distance.",
        "- One phone filming every run from the side.",
        "",
        "## After runs 1-3: stop and check",
        "",
        "Copy the bags off, run `scripts.run_sheet match` and `scripts.real_session` and look at the quick-look plots and a",
        "few photos. Continue only if the clean runs have no red and the camera image is sharp.",
        "",
        "## Runs",
        "",
        "Record each with `bash record_drive.sh <run id>`. Tick when done; note anything odd (bumped hazard, stopped early).",
        "",
        "| ✓ | Run | Drive | Driver | Put down (distances from end A) | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        what = "<br>".join(describe(h) for h in run["hazards"]) or "**clean** (remove everything)"
        lines.append(f"| ☐ | `{run['run_id']}` | {run['direction']} | {run['driver']} | {what} | |")
    (out / "run_sheet.md").write_text("\n".join(lines) + "\n")

    # One line per run for hardware/session_loop.sh on the robot: run id, then what to do, in plain text.
    with (out / "prompts.txt").open("w") as file:
        for run in runs:
            what = " | ".join(describe(h).replace("**", "") for h in run["hazards"]) or "CLEAN: remove everything"
            file.write(f"{run['run_id']}\tdrive {run['direction']} ({run['driver']}) | {what}\n")


def match(plan: Path, bags: Path) -> Path:
    with plan.open(newline="") as file:
        rows = list(csv.DictReader(file))
    folders = sorted(p.name for p in bags.iterdir() if p.is_dir() and (p / "metadata.yaml").exists())
    out_rows, missing = [], []
    for row in rows:
        hits = [f for f in folders if f == row["bag"] or f.startswith(row["bag"] + "_")]
        if not hits:
            missing.append(row["bag"])
            continue
        if len(hits) > 1:
            print(f"note: {row['bag']} was recorded {len(hits)} times; using the last one ({hits[-1]})")
        out_rows.append({**row, "bag": hits[-1]})
    unplanned = sorted(set(folders) - {r["bag"] for r in out_rows})
    target = bags / "drives.csv"
    with target.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"matched {len(out_rows)} of {len(rows)} planned runs -> {target}")
    if missing:
        print("not recorded: " + ", ".join(missing))
    if unplanned:
        print("bags not in the plan (left out): " + ", ".join(unplanned))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="write run_sheet.md, drives.csv and plan.json")
    make.add_argument("--session", required=True, help="short id used in run names, e.g. s1")
    make.add_argument("--location", required=True)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--kinds", default=",".join(DEFAULT_KINDS), help=f"comma list from {sorted(HAZARDS)}")
    make.add_argument("--objects", type=Path, help="JSON list of extra objects to use in --kinds")
    make.add_argument("--hazard-runs", type=int, default=40)
    make.add_argument("--clean-runs", type=int, default=8)
    make.add_argument("--line-m", type=float, default=9.0)
    make.add_argument("--slots", default="2.5-3.5,5.5-6.5",
                      help="the two object slots in m from A; keep about 2 m from each end and between them")
    make.add_argument("--angles", default="0,30", help="degrees from straight across")
    make.add_argument("--script-fraction", type=float, default=0.0,
                      help="share of runs driven by the scripted driver (0 until it is tested on the robot)")
    make.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("match", help="map recorded bags to planned runs, write <bags>/drives.csv")
    m.add_argument("--plan", type=Path, required=True)
    m.add_argument("--bags", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "match":
        match(args.plan, args.bags)
        return
    if args.objects:
        for o in json.loads(args.objects.read_text()):
            HAZARDS[o["kind"]] = Hazard(o["kind"], o["what"], float(o["along_m"]), float(o["height_m"]))
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = sorted(set(kinds) - set(HAZARDS))
    if unknown:
        raise SystemExit(f"unknown hazard kinds {unknown}; choose from {sorted(HAZARDS)}")
    runs = plan_runs(session=args.session, location=args.location, kinds=kinds, hazard_runs=args.hazard_runs,
                     clean_runs=args.clean_runs, line_m=args.line_m,
                     slots=tuple(tuple(float(x) for x in sl.split("-")) for sl in args.slots.split(",")),
                     angles=[float(a) for a in args.angles.split(",")], script_fraction=args.script_fraction,
                     seed=args.seed)
    write_plan(runs, args.out, args)
    print(f"wrote {args.out}/run_sheet.md, drives.csv, prompts.txt and plan.json ({len(runs)} runs)")


if __name__ == "__main__":
    main()
