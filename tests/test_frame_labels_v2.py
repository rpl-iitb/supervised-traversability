import unittest

import numpy as np

from scripts.episode_io import camera_model
from scripts.frame_labels_v2 import Settings, brake_zones, calibrate, drive_zones, episode_signals, label_episode


CAMERA = camera_model(320, 240, 90.0, (0.31, 0.0, 0.35), 15.0, 10.0)
SCENARIO = {
    "scenario_id": "warehouse-0001", "seed": 1,
    "route": {"start_pose": {"x_m": 0.0, "y_m": 0.0}, "goal_pose": {"x_m": 6.0, "y_m": 0.0}, "length_m": 6.0},
    "objects": [],
}
CABLE_AT = 3.0


def episode(bursts_at=(), stop_at=None, length=8.3, condition="hazard", vib=2.0):
    """Robot drives along +x at 0.5 m/s; vertical shaking bursts at given centre positions."""
    state, frames, x, t, dt = [], [], 0.0, 0.0, 0.01
    while x < length and t < 40:
        speed = 0.0 if (stop_at is not None and x >= stop_at) else 0.5
        shake = any(abs(x - b) < 0.03 for b in bursts_at)
        state.append({"t_s": t, "x_m": x, "y_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0, "pitch_rad": 0.0, "roll_rad": 0.0,
                      "v_fwd_mps": speed, "yaw_rate_rps": 0.0, "cmd_v_mps": 0.5, "cmd_w_rps": 0.0,
                      "imu_ax": 0.0, "imu_ay": 0.0, "imu_az": 9.81 + (vib * (1 if int(t * 100) % 2 else -1) if shake else 0.0),
                      "imu_gx": 0.0, "imu_gy": 0.0, "imu_gz": 0.0, "hazard_contact_n": 0.0, "contact_object": "",
                      "wheel_v_mps": speed})
        if round(t * 100) % 10 == 0:
            frames.append({"frame_idx": len(frames), "t_s": t, "x_m": x, "y_m": 0.0, "yaw_rad": 0.0,
                           "rgb_path": f"rgb/{len(frames):06d}.jpg"})
        x += speed * dt
        t += dt
    termination = {"reason": "stalled"} if stop_at is not None else {"reason": "goal_reached"}
    return {"meta": {"episode_id": f"ep-{condition}", "condition": condition, "camera": CAMERA,
                     "rates": {"state_hz": 100}, "termination": termination},
            "scenario": SCENARIO, "state": state, "frames": frames}


THRESHOLDS = {"vib_mps2": 0.05, "speed_ratio": 0.95, "slip": 0.05, "brake_stop_m": 0.03}


def cable_bursts(settings=Settings()):
    # Drive wheels hit the cable when the centre is 0.20 m before it; casters when 0.29 m past it.
    return (CABLE_AT - settings.front_contact_m, CABLE_AT - settings.rear_contact_m)


def frame_at(rows, ep, x):
    frame = min(ep["frames"], key=lambda f: abs(f["x_m"] - x))
    return next(r for r in rows if r["frame_idx"] == frame["frame_idx"])


class ZoneTest(unittest.TestCase):
    def test_wheel_and_caster_bursts_merge_onto_the_cable(self):
        ep = episode(bursts_at=cable_bursts())
        zones = drive_zones(episode_signals(ep, Settings()), THRESHOLDS, ep["meta"]["termination"], Settings())
        self.assertEqual(len(zones), 1)
        centre = (zones[0]["lo"] + zones[0]["hi"]) / 2
        self.assertAlmostEqual(centre, CABLE_AT, delta=0.08)
        self.assertLess(zones[0]["hi"] - zones[0]["lo"], 0.2)
        self.assertTrue(zones[0]["unacceptable"])

    def test_mild_shaking_is_abnormal_but_acceptable(self):
        ep = episode(bursts_at=cable_bursts(), vib=0.1)
        zone = drive_zones(episode_signals(ep, Settings()), THRESHOLDS, ep["meta"]["termination"], Settings())[0]
        self.assertFalse(zone["unacceptable"])

    def test_stall_marks_floor_ahead_unacceptable(self):
        ep = episode(stop_at=2.5)
        zones = drive_zones(episode_signals(ep, Settings()), THRESHOLDS, ep["meta"]["termination"], Settings())
        stall = [z for z in zones if z["cause"] == "stalled"][0]
        self.assertAlmostEqual(stall["lo"], 2.5 + Settings().front_contact_m, delta=0.02)

    def test_long_brake_stop_beyond_budget_is_unacceptable(self):
        brake = {"meta": {"brake_test": {"stops": [
            {"start_progress_m": 2.0, "stop_progress_m": 2.17, "distance_m": 0.17, "start_speed_mps": 0.5},
            {"start_progress_m": 2.4, "stop_progress_m": 2.47, "distance_m": 0.07, "start_speed_mps": 0.5},
            {"start_progress_m": 2.8, "stop_progress_m": 2.82, "distance_m": 0.02, "start_speed_mps": 0.5}]}}}
        zones = brake_zones([brake], THRESHOLDS, Settings())
        self.assertEqual([z["unacceptable"] for z in zones], [True, False])  # 0.02 m is normal -> no zone

    def test_calibration_is_per_drive(self):
        clean = [episode(condition="clean") for _ in range(3)]
        th = calibrate(clean, [], Settings())
        self.assertEqual(th["clean_drives"], 3)
        self.assertLess(th["vib_mps2"], 0.05)


class RealRobotSettingsTest(unittest.TestCase):
    REAL = Settings(steady_after_start_s=1.0, steady_before_stop_s=0.2, min_frame_speed_mps=0.2, severity="peak",
                    unacceptable_vib_clean_multiple=2.0)

    def teleop(self, bursts_at=(), vib=2.0, start_s=3.0, stop_x=5.0):
        """Parked, then a teleop drive to stop_x, then parked; jolts at both ends of the drive."""
        ep = episode(bursts_at=bursts_at, vib=vib)
        state, frames, x, dt = [], [], 0.0, 0.01
        for i in range(int(14.0 / dt)):
            t = i * dt
            driving = t >= start_s and x < stop_x
            speed = 0.5 if driving else 0.0
            x += speed * dt
            jolt = abs(t - start_s) < 0.05 or abs(x - stop_x) < 0.01
            shake = any(abs(x - b) < 0.03 for b in bursts_at)
            az = 9.81 + (3.0 if jolt else 0.0) * (1 if i % 2 else -1) + (vib * (1 if i % 2 else -1) if shake else 0.0)
            state.append({**ep["state"][0], "t_s": t, "x_m": x, "v_fwd_mps": speed, "cmd_v_mps": 0.5 if driving else 0.0,
                          "imu_az": az, "wheel_v_mps": speed})
            if i % 10 == 0:
                frames.append({"frame_idx": len(frames), "t_s": t, "x_m": x, "y_m": 0.0, "yaw_rad": 0.0, "rgb_path": "x.jpg"})
        return {**ep, "state": state, "frames": frames}

    def test_start_and_stop_jolts_make_no_zone(self):
        ep = self.teleop()
        zones = drive_zones(episode_signals(ep, self.REAL), THRESHOLDS, ep["meta"]["termination"], self.REAL)
        self.assertEqual(zones, [])

    def test_limit_is_relative_to_clean_level_and_uses_peak(self):
        settings_thresholds = {**THRESHOLDS, "vib_mps2": 0.5}
        mild, hard = self.teleop(bursts_at=[2.5], vib=0.8), self.teleop(bursts_at=[2.5], vib=2.0)
        for ep, expected in ((mild, False), (hard, True)):
            zones = drive_zones(episode_signals(ep, self.REAL), settings_thresholds, ep["meta"]["termination"], self.REAL)
            self.assertEqual(len(zones), 1)
            self.assertEqual(zones[0]["unacceptable"], expected)

    def test_roll_alone_makes_a_zone_and_can_be_unacceptable(self):
        real = Settings(steady_after_start_s=1.0, steady_before_stop_s=0.2, severity="peak", use_roll=True,
                        unacceptable_vib_clean_multiple=2.0, unacceptable_roll_clean_multiple=2.0)
        ep = self.teleop()
        ep["state"] = [{**row, "imu_gx": 0.5 if abs(row["x_m"] - 2.5) < 0.03 else 0.0} for row in ep["state"]]
        thresholds = {**THRESHOLDS, "vib_mps2": 0.5, "roll_rps": 0.04}
        zones = drive_zones(episode_signals(ep, real), thresholds, ep["meta"]["termination"], real)
        self.assertEqual([z["cause"] for z in zones], ["roll"])

    def test_parked_photos_are_dropped(self):
        ep = self.teleop()
        rows = label_episode(ep, [], self.REAL)
        self.assertEqual(rows[0]["reason"], "not_moving")
        self.assertEqual(rows[-1]["reason"], "not_moving")
        self.assertTrue(any(r["reason"] == "normal" for r in rows))


class PhotoLabelTest(unittest.TestCase):
    def setUp(self):
        self.ep = episode(bursts_at=cable_bursts())
        zones = drive_zones(episode_signals(self.ep, Settings()), THRESHOLDS, self.ep["meta"]["termination"], Settings())
        self.rows = label_episode(self.ep, zones, Settings())

    def test_photo_whose_strip_holds_the_cable_is_not_traversable(self):
        self.assertEqual(frame_at(self.rows, self.ep, 2.0)["y"], 0)    # strip 2.6-3.8

    def test_clean_photo_just_before_the_cable_is_traversable_not_gray(self):
        self.assertEqual(frame_at(self.rows, self.ep, 1.0)["y"], 1)    # strip 1.6-2.8

    def test_photos_beyond_route_end_are_excluded(self):
        self.assertEqual(frame_at(self.rows, self.ep, 7.0)["reason"], "beyond_route")

    def test_photo_whose_strip_was_never_driven_is_gray(self):
        ep = episode(length=6.5)
        rows = label_episode(ep, [], Settings())
        self.assertEqual(frame_at(rows, ep, 5.5)["reason"], "not_reached")


if __name__ == "__main__":
    unittest.main()
