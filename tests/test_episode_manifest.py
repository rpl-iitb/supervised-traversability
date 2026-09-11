import csv
import json
import tempfile
import unittest
from pathlib import Path


class EpisodeManifestTest(unittest.TestCase):
    def _write_scenario(self, directory: Path) -> Path:
        path = directory / "starter-0007.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "scenario_id": "starter-0007",
                    "seed": 7,
                    "robot": {"model": "nova_carter", "start_pose": {"x_m": -4.5, "y_m": 0, "yaw_rad": 0}},
                    "objects": [
                        {
                            "instance_id": "cable_hose_01",
                            "asset_key": "cable_hose",
                            "semantic_class": "low_profile_cable",
                            "pose": {"x_m": 1.2, "y_m": 0.1, "yaw_rad": 0},
                        }
                    ],
                }
            )
        )
        return path

    def _write_csv(self, directory: Path) -> Path:
        path = directory / "run.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "elapsed_s",
                    "x_m",
                    "y_m",
                    "yaw_rad",
                    "speed_mps",
                    "command_wheel_deg_s",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"elapsed_s": 0.0, "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0, "speed_mps": 0.0, "command_wheel_deg_s": 120.0},
                    {"elapsed_s": 1.0, "x_m": 0.3, "y_m": 0.0, "yaw_rad": 0.0, "speed_mps": 0.3, "command_wheel_deg_s": 120.0},
                    {"elapsed_s": 2.0, "x_m": 0.6, "y_m": 0.0, "yaw_rad": 0.0, "speed_mps": 0.3, "command_wheel_deg_s": 120.0},
                ]
            )
        return path

    def test_build_run_manifest_retains_scene_metrics_and_positive_label(self):
        from scripts.episode_manifest import build_run_manifest

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = build_run_manifest(
                scenario_path=self._write_scenario(directory),
                csv_path=self._write_csv(directory),
                crossing_completed=True,
                speed_tracking_ratio=0.9,
                safety_stop=False,
                contact_impulse_ns=0.0,
            )

        self.assertEqual(manifest["scenario"]["scenario_id"], "starter-0007")
        self.assertEqual(manifest["scenario"]["objects"][0]["instance_id"], "cable_hose_01")
        self.assertEqual(
            manifest["telemetry"],
            {"sample_count": 3, "duration_s": 2.0, "path_length_m": 0.6, "mean_speed_mps": 0.3},
        )
        self.assertEqual(
            manifest["traversability_label"],
            {"score": 0.9, "binary_label": 1, "reason": "normal_crossing"},
        )
        self.assertEqual(
            manifest["object_encounters"],
            [
                {
                    "instance_id": "cable_hose_01",
                    "asset_key": "cable_hose",
                    "closest_elapsed_s": 2.0,
                    "closest_distance_m": 0.608,
                }
            ],
        )

    def test_safety_stop_is_recorded_as_non_traversable(self):
        from scripts.episode_manifest import build_run_manifest

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = build_run_manifest(
                scenario_path=self._write_scenario(directory),
                csv_path=self._write_csv(directory),
                crossing_completed=True,
                speed_tracking_ratio=0.95,
                safety_stop=True,
                contact_impulse_ns=0.0,
            )

        self.assertEqual(manifest["traversability_label"]["binary_label"], 0)
        self.assertEqual(manifest["traversability_label"]["reason"], "safety_stop")
