import unittest

from scripts.episode_io import camera_model
from scripts.frame_labels import LabelConfig, hazard_encounters, label_episode, outcome_label


CAMERA = camera_model(320, 240, 90.0, (0.3, 0.0, 0.35), 15.0, 10.0)
HAZARD = {
    "instance_id": "small_lip_01",
    "asset_key": "small_lip",
    "role": "target_hazard",
    "footprint_radius_m": 0.2,
    "height_m": 0.04,
    "pose": {"x_m": 3.0, "y_m": 0.0, "yaw_rad": 0.0},
}


def synthetic_episode(speed_near_hazard=0.5, stop_at_x=None, hazard=True, vib_near_hazard=0.0):
    """Robot drives along +x at 0.5 m/s, optionally slowing/stopping near x=3."""
    state, frames, x, t, dt = [], [], 0.0, 0.0, 0.01
    while x < 6.0 and t < 40.0:
        near = abs(x - 3.0) < 0.6
        speed = speed_near_hazard if near else 0.5
        if stop_at_x is not None and x >= stop_at_x:
            speed = 0.0
        state.append({
            "t_s": t, "x_m": x, "y_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0, "pitch_rad": 0.0, "roll_rad": 0.0,
            "v_fwd_mps": speed, "yaw_rate_rps": 0.0, "cmd_v_mps": 0.5, "cmd_w_rps": 0.0,
            "imu_ax": 0.0, "imu_ay": 0.0, "imu_az": 9.81 + (vib_near_hazard * (1 if int(t * 100) % 2 else -1) if near else 0.0),
            "imu_gx": 0.0, "imu_gy": 0.0, "imu_gz": 0.0, "hazard_contact_n": 0.0, "contact_object": "",
        })
        if round(t * 100) % 10 == 0:
            frames.append({"frame_idx": len(frames), "t_s": t, "x_m": x, "y_m": 0.0, "yaw_rad": 0.0,
                           "rgb_path": f"rgb/{len(frames):06d}.png"})
        x += speed * dt
        t += dt
    termination = "stalled" if stop_at_x is not None else "goal_reached"
    return {
        "meta": {"episode_id": "ep", "condition": "hazard" if hazard else "clean", "camera": CAMERA,
                 "termination": {"reason": termination}},
        "scenario": {"scenario_id": "warehouse-0001", "seed": 1, "objects": [HAZARD] if hazard else []},
        "state": state,
        "frames": frames,
    }


def by_x(records, episode, x):
    frame = min(episode["frames"], key=lambda f: abs(f["x_m"] - x))
    return next(r for r in records if r["frame_idx"] == frame["frame_idx"])


class OutcomeLabelTest(unittest.TestCase):
    def test_rule(self):
        config = LabelConfig(vibration_soft=1.0, vibration_hard=3.0)
        self.assertEqual(outcome_label({"tracking": 0.95, "vibration_rms": 0.2}, config), (1, "normal_traversal"))
        self.assertEqual(outcome_label({"tracking": 0.40, "vibration_rms": 0.2}, config), (0, "poor_speed_tracking"))
        self.assertEqual(outcome_label({"tracking": 0.95, "vibration_rms": 4.0}, config), (0, "severe_vibration"))
        self.assertEqual(outcome_label({"tracking": 0.95, "vibration_rms": 2.0}, config)[0], None)
        self.assertEqual(outcome_label({"tracking": 0.65, "vibration_rms": 0.2}, config)[0], None)


class FrameLabelTest(unittest.TestCase):
    def test_clean_episode_frames_are_positive_until_patch_runs_out(self):
        episode = synthetic_episode(hazard=False)
        records = label_episode(episode, LabelConfig())
        self.assertEqual(by_x(records, episode, 1.0)["y"], 1)
        self.assertEqual(by_x(records, episode, 1.0)["patch_class"], "clean_floor")
        self.assertEqual(by_x(records, episode, 5.5)["reason"], "patch_not_covered")

    def test_smooth_crossing_labels_hazard_frames_positive(self):
        episode = synthetic_episode(speed_near_hazard=0.5)
        record = by_x(label_episode(episode, LabelConfig()), episode, 1.8)
        self.assertEqual(record["patch_class"], "small_lip")
        self.assertEqual(record["y_outcome"], 1)
        self.assertEqual(record["y_prior"], 0)
        self.assertTrue(record["bbox"])

    def test_slow_crossing_labels_hazard_frames_negative(self):
        episode = synthetic_episode(speed_near_hazard=0.1)
        record = by_x(label_episode(episode, LabelConfig()), episode, 1.8)
        self.assertEqual(record["y"], 0)
        self.assertEqual(record["reason"], "poor_speed_tracking")

    def test_vibration_makes_crossing_negative(self):
        episode = synthetic_episode(vib_near_hazard=5.0)
        record = by_x(label_episode(episode, LabelConfig(vibration_soft=0.5, vibration_hard=1.5)), episode, 1.8)
        self.assertEqual((record["y"], record["reason"]), (0, "severe_vibration"))

    def test_stall_before_hazard_is_negative(self):
        episode = synthetic_episode(stop_at_x=2.6)
        record = by_x(label_episode(episode, LabelConfig()), episode, 1.8)
        self.assertEqual((record["y"], record["reason"]), (0, "stalled_before_or_in_patch"))

    def test_stall_on_hazard_is_not_blamed_on_clean_patch_ahead(self):
        episode = synthetic_episode(stop_at_x=3.0)
        record = by_x(label_episode(episode, LabelConfig()), episode, 2.95)
        self.assertEqual(record["patch_class"], "clean_floor")
        self.assertEqual((record["y"], record["reason"]), ("", "stalled_on_other_hazard"))

    def test_hazard_under_robot_contaminates_clean_patch_window(self):
        episode = synthetic_episode()
        record = by_x(label_episode(episode, LabelConfig()), episode, 2.8)
        self.assertEqual(record["patch_class"], "clean_floor")
        self.assertEqual((record["y"], record["reason"]), ("", "window_contaminated_by_other_hazard"))

    def test_operator_prior_policy_uses_prior(self):
        episode = synthetic_episode(speed_near_hazard=0.5)
        record = by_x(label_episode(episode, LabelConfig(policy="operator_prior")), episode, 1.8)
        self.assertEqual(record["y"], 0)

    def test_encounter_summary(self):
        smooth = hazard_encounters(synthetic_episode(), LabelConfig())[0]
        stalled = hazard_encounters(synthetic_episode(stop_at_x=2.0), LabelConfig())[0]
        self.assertTrue(smooth["reached"])
        self.assertEqual(smooth["outcome_label"], 1)
        self.assertEqual((stalled["reached"], stalled["outcome_label"]), (False, 0))


if __name__ == "__main__":
    unittest.main()
