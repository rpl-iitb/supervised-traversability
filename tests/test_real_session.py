import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from tests.test_rosbag_to_episode import CLEARPATH_JAZZY_LAYOUT, Writer, write_bag
except ImportError:  # pragma: no cover
    Writer = None

REPO = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, "-m", "scripts.real_session", *args], cwd=REPO, capture_output=True, text=True)


@unittest.skipIf(Writer is None, "rosbags not installed")
class RealSessionTest(unittest.TestCase):
    """Lab-day rehearsal on synthetic Clearpath Jazzy bags: template, fill in, convert, label, check."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.bags, cls.out = root / "bags", root / "out"
        cls.bags.mkdir()
        for name, cable in (("clean_01", False), ("clean_02", False), ("cable_01", True)):
            write_bag(cls.bags / name, cable, CLEARPATH_JAZZY_LAYOUT)
        common = ("--bags", str(cls.bags), "--out", str(cls.out), "--camera-mount", "0.2", "0", "0.3",
                  "--camera-pitch-deg", "15", "--image-scale", "1")
        cls.first = run(*common)
        drives = cls.bags / "drives.csv"
        with drives.open(newline="") as file:
            cls.template = list(csv.DictReader(file))
        rows = [{**r, "hazards": "cable:3.0:0.05:0.03" if r["bag"] == "cable_01" else ""} for r in cls.template]
        with drives.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["bag", "condition", "hazards", "note"])
            writer.writeheader()
            writer.writerows(rows)
        cls.second = run(*common)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_first_run_writes_a_template_guessing_conditions_from_names(self):
        self.assertNotEqual(self.first.returncode, 0)
        self.assertIn("Fill in", self.first.stderr)
        self.assertEqual({r["bag"]: r["condition"] for r in self.template},
                         {"cable_01": "hazard", "clean_01": "clean", "clean_02": "clean"})

    def test_second_run_finds_the_cable_and_keeps_clean_drives_clean(self):
        self.assertEqual(self.second.returncode, 0, self.second.stderr)
        lines = {line.split()[0]: line for line in self.second.stdout.splitlines() if line.startswith(("clean_", "cable_"))}
        self.assertIn("cable found", lines["cable_01"])
        self.assertIn("unacceptable", lines["cable_01"])
        self.assertIn("no false alarms", lines["clean_01"])
        self.assertIn("no false alarms", lines["clean_02"])

    def test_writes_labels_and_one_quicklook_per_drive(self):
        self.assertTrue((self.out / "labels.csv").exists())
        self.assertEqual(sorted(p.name for p in (self.out / "quicklook").glob("*.png")),
                         ["cable_01.png", "clean_01.png", "clean_02.png"])


if __name__ == "__main__":
    unittest.main()
