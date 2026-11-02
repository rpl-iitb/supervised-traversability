import collections
import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.real_session import read_drives
from scripts.run_sheet import DEFAULT_KINDS, HAZARDS, along_line_radius, match, plan_runs

KINDS = list(DEFAULT_KINDS)


def plan(**overrides):
    args = dict(session="s1", location="lab", kinds=KINDS, hazard_runs=40, clean_runs=8, line_m=9.0,
                slots=((2.5, 3.5), (5.5, 6.5)), angles=[0.0, 30.0], script_fraction=0.0, seed=3)
    return plan_runs(**{**args, **overrides})


class PlanTest(unittest.TestCase):
    def test_counts_and_balance(self):
        runs = plan()
        self.assertEqual(len(runs), 48)
        self.assertEqual(sum(r["condition"] == "clean" for r in runs), 8)
        self.assertEqual([r["condition"] for r in runs[:2]], ["clean", "clean"])
        per_kind = collections.Counter(h["kind"] for r in runs for h in r["hazards"])
        self.assertEqual(set(per_kind.values()), {16})
        per_slot = collections.Counter((i, h["kind"]) for r in runs for i, h in enumerate(r["hazards"]))
        self.assertEqual(set(per_slot.values()), {8})

    def test_two_different_hazards_far_enough_apart(self):
        for run in plan():
            if run["condition"] == "clean":
                continue
            a, b = run["hazards"]
            self.assertNotEqual(a["kind"], b["kind"])
            # Further apart than the look-ahead strip (1.8 m) plus the robot's length.
            self.assertGreaterEqual(abs(a["from_a_m"] - b["from_a_m"]), 2.0)

    def test_more_slots_on_a_longer_line(self):
        kinds = [f"k{i}" for i in range(10)]
        hazards = {k: HAZARDS["debris"] for k in kinds}
        with mock.patch.dict(HAZARDS, hazards):
            runs = plan(kinds=kinds, hazard_runs=20, clean_runs=3, line_m=12.0,
                        slots=((1.9, 2.1), (3.9, 4.1), (5.9, 6.1), (7.9, 8.1), (9.9, 10.1)))
        crossings = collections.Counter()
        for run in runs:
            if run["condition"] == "clean":
                continue
            names = [h["kind"] for h in run["hazards"]]
            self.assertEqual(len(names), 5)
            self.assertEqual(len(set(names)), 5)
            where = [h["from_a_m"] for h in run["hazards"]]
            self.assertGreaterEqual(min(b - a for a, b in zip(where, where[1:])), 1.8 - 1e-9)
            crossings.update(names)
        self.assertEqual(set(crossings.values()), {10})

    def test_too_few_kinds_for_the_slots_is_refused(self):
        with self.assertRaises(ValueError):
            plan(kinds=["cable_thin"])

    def test_reverse_runs_measure_from_end_b(self):
        for run in plan():
            for h in run["hazards"]:
                expected = h["from_a_m"] if run["direction"] == "A->B" else round(9.0 - h["from_a_m"], 1)
                self.assertAlmostEqual(h["from_start_m"], expected)

    def test_diagonal_cable_spans_more_of_the_line(self):
        cable = HAZARDS["cable_thick"]
        self.assertAlmostEqual(along_line_radius(cable, 0.0), 0.0125, delta=0.001)  # rounded to mm
        self.assertGreater(along_line_radius(cable, 30.0), 0.12)

    def test_same_seed_same_plan_and_script_fraction(self):
        self.assertEqual(plan(), plan())
        runs = plan(script_fraction=0.5)
        self.assertEqual(sum(r["driver"] == "script" for r in runs), 24)


class MatchTest(unittest.TestCase):
    def test_maps_timestamped_bags_and_real_session_reads_the_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_csv = root / "drives.csv"
            with plan_csv.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["bag", "condition", "hazards", "note", "direction", "driver", "location"])
                writer.writerow(["s1_r001", "clean", "", "clean", "A->B", "teleop", "lab"])
                writer.writerow(["s1_r002", "hazard", "cable_thick:3.0:0.0125:0.025", "x", "A->B", "teleop", "lab"])
                writer.writerow(["s1_r003", "hazard", "mat:6.0:0.2:0.01", "x", "B->A", "teleop", "lab"])
            bags = root / "bags"
            for name in ("s1_r001_20261029_100000", "s1_r002_20261029_100200", "s1_r002_20261029_100500", "stray_test"):
                (bags / name).mkdir(parents=True)
                (bags / name / "metadata.yaml").write_text("x")
            target = match(plan_csv, bags)
            rows = read_drives(target)
            self.assertEqual([r["bag"] for r in rows], ["s1_r001_20261029_100000", "s1_r002_20261029_100500"])
            self.assertEqual(rows[1]["hazard_list"], ["cable_thick:3.0:0.0125:0.025"])


if __name__ == "__main__":
    unittest.main()
