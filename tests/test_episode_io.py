import math
import tempfile
import unittest
from pathlib import Path

from scripts.episode_io import (
    STATE_COLUMNS,
    camera_model,
    footprint_bbox,
    load_episode,
    project_base_point,
    world_to_base,
    write_episode,
)


CAMERA = camera_model(320, 240, 90.0, (0.3, 0.0, 0.35), 15.0, 10.0)


class EpisodeIoTest(unittest.TestCase):
    def test_world_to_base_rotates_into_robot_frame(self):
        pose = {"x_m": 1.0, "y_m": 2.0, "yaw_rad": math.pi / 2}
        forward, left = world_to_base(pose, 1.0, 3.0)
        self.assertAlmostEqual(forward, 1.0)
        self.assertAlmostEqual(left, 0.0)

    def test_point_straight_ahead_projects_to_centre_column_below_horizon(self):
        horizon_v = CAMERA["cy"] - CAMERA["fy"] * math.tan(math.radians(CAMERA["pitch_down_deg"]))
        u, v = project_base_point(CAMERA, 2.0, 0.0)
        self.assertAlmostEqual(u, CAMERA["cx"])
        self.assertGreater(v, horizon_v)

    def test_optical_axis_hits_floor_at_image_centre(self):
        axis_forward = 0.3 + 0.35 / math.tan(math.radians(CAMERA["pitch_down_deg"]))
        u, v = project_base_point(CAMERA, axis_forward, 0.0)
        self.assertAlmostEqual(u, CAMERA["cx"])
        self.assertAlmostEqual(v, CAMERA["cy"], places=6)

    def test_point_on_left_projects_to_left_half(self):
        u, _ = project_base_point(CAMERA, 2.0, 0.5)
        self.assertLess(u, CAMERA["cx"])

    def test_point_behind_camera_is_not_projected(self):
        self.assertIsNone(project_base_point(CAMERA, 0.0, 0.0))

    def test_footprint_bbox_is_none_for_objects_behind_robot(self):
        robot = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
        item = {"pose": {"x_m": -2.0, "y_m": 0.0}, "footprint_radius_m": 0.2, "height_m": 0.03}
        self.assertIsNone(footprint_bbox(CAMERA, robot, item))

    def test_footprint_bbox_inside_image_for_object_ahead(self):
        robot = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
        item = {"pose": {"x_m": 2.0, "y_m": 0.0}, "footprint_radius_m": 0.2, "height_m": 0.03}
        u0, v0, u1, v1 = footprint_bbox(CAMERA, robot, item)
        self.assertLess(u0, CAMERA["cx"])
        self.assertGreater(u1, CAMERA["cx"])
        self.assertTrue(0 <= v0 < v1 <= CAMERA["height"])

    def test_round_trip(self):
        state = [{column: 0.0 for column in STATE_COLUMNS} | {"contact_object": ""}]
        frames = [{"frame_idx": 0, "t_s": 0.0, "x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0, "rgb_path": "rgb/000000.png"}]
        with tempfile.TemporaryDirectory() as directory:
            write_episode(
                Path(directory) / "ep",
                scenario={"scenario_id": "s", "objects": []},
                meta={"episode_id": "ep", "camera": CAMERA},
                state_rows=state,
                frame_rows=frames,
            )
            episode = load_episode(Path(directory) / "ep")
        self.assertEqual(episode["frames"][0]["frame_idx"], 0)
        self.assertEqual(episode["frames"][0]["rgb_path"], "rgb/000000.png")
        self.assertEqual(episode["state"][0]["contact_object"], "")
        self.assertEqual(episode["meta"]["format_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
