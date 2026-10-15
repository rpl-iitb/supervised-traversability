import io
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
except ImportError:  # pragma: no cover
    Writer = None

from scripts.episode_io import load_episode
from scripts.frame_labels_v2 import JACKAL, Settings, build
from scripts.rosbag_to_episode import convert, optical_to_base

PITCH = 15.0
MOUNT = (0.2, 0.0, 0.3)
CABLE_AT = 3.0


def write_bag(path: Path, cable: bool) -> None:
    """Jackal driving straight along +x at 0.5 m/s; D435i IMU (optical frame) shakes over a cable."""
    from PIL import Image

    ts = get_typestore(Stores.ROS2_HUMBLE)
    T = ts.types
    Time, Header = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"]
    Vec, Quat, Point = T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Point"]

    def header(t, frame="odom"):
        return Header(stamp=Time(sec=int(t), nanosec=int(round((t % 1) * 1e9))), frame_id=frame)

    to_optical = optical_to_base(PITCH).T
    jpeg = io.BytesIO()
    Image.fromarray(np.full((48, 64, 3), 120, np.uint8)).save(jpeg, format="JPEG")
    jpeg = np.frombuffer(jpeg.getvalue(), dtype=np.uint8)
    t_start, x = 100.0, 0.0
    with Writer(path, version=8) as writer:
        c = {name: writer.add_connection(topic, msgtype, typestore=ts) for name, topic, msgtype in (
            ("odom", "/odometry/filtered", "nav_msgs/msg/Odometry"),
            ("wheel", "/jackal_velocity_controller/odom", "nav_msgs/msg/Odometry"),
            ("cmd", "/cmd_vel", "geometry_msgs/msg/Twist"),
            ("imu", "/camera/imu", "sensor_msgs/msg/Imu"),
            ("img", "/camera/color/image_raw/compressed", "sensor_msgs/msg/CompressedImage"),
            ("info", "/camera/color/camera_info", "sensor_msgs/msg/CameraInfo"))}

        def put(name, t, msg, msgtype):
            writer.write(c[name], int(t * 1e9), ts.serialize_cdr(msg, msgtype))

        cov36, cov9 = np.zeros(36), np.zeros(9)
        for k in range(int(12 * 200)):  # 200 Hz master clock, 12 s
            t = t_start + k / 200
            moving = t - t_start > 0.5
            v = 0.5 if moving else 0.0
            if k % 4 == 0:  # 50 Hz odometry
                twist = T["geometry_msgs/msg/TwistWithCovariance"](
                    twist=T["geometry_msgs/msg/Twist"](linear=Vec(x=v, y=0.0, z=0.0), angular=Vec(x=0.0, y=0.0, z=0.0)), covariance=cov36)
                pose = T["geometry_msgs/msg/PoseWithCovariance"](
                    pose=T["geometry_msgs/msg/Pose"](position=Point(x=x, y=0.0, z=0.0), orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0)), covariance=cov36)
                odom = T["nav_msgs/msg/Odometry"](header=header(t), child_frame_id="base_link", pose=pose, twist=twist)
                put("odom", t, odom, "nav_msgs/msg/Odometry")
                put("wheel", t, odom, "nav_msgs/msg/Odometry")
            if k % 10 == 0:
                put("cmd", t, T["geometry_msgs/msg/Twist"](linear=Vec(x=0.5, y=0.0, z=0.0), angular=Vec(x=0.0, y=0.0, z=0.0)), "geometry_msgs/msg/Twist")
            shake = cable and any(abs(x - b) < 0.03 for b in (CABLE_AT - 0.20, CABLE_AT + 0.20))
            az = 9.81 + (2.0 * (1 if k % 2 else -1) if shake else 0.0) + 0.01 * math.sin(k)
            a_opt = to_optical @ np.array([0.0, 0.0, az])
            imu = T["sensor_msgs/msg/Imu"](header=header(t, "camera_imu_optical_frame"), orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                                          orientation_covariance=cov9, angular_velocity=Vec(x=0.0, y=0.0, z=0.0),
                                          angular_velocity_covariance=cov9,
                                          linear_acceleration=Vec(x=float(a_opt[0]), y=float(a_opt[1]), z=float(a_opt[2])),
                                          linear_acceleration_covariance=cov9)
            put("imu", t, imu, "sensor_msgs/msg/Imu")
            if k % 7 == 0:  # ~29 Hz camera
                put("img", t, T["sensor_msgs/msg/CompressedImage"](header=header(t, "camera"), format="jpeg", data=jpeg),
                    "sensor_msgs/msg/CompressedImage")
            if k == 0:
                info = T["sensor_msgs/msg/CameraInfo"](
                    header=header(t, "camera"), height=48, width=64, distortion_model="plumb_bob", d=np.zeros(5),
                    k=np.array([40.0, 0, 32, 0, 40.0, 24, 0, 0, 1]), r=np.eye(3).ravel(), p=np.zeros(12), binning_x=0, binning_y=0,
                    roi=T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
                put("info", t, info, "sensor_msgs/msg/CameraInfo")
            x += v / 200


@unittest.skipIf(Writer is None, "rosbags not installed")
class ConvertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.episodes = root / "episodes"
        for name, cable in (("ep-clean-1", False), ("ep-clean-2", False), ("ep-cable-1", True)):
            write_bag(root / f"{name}_bag", cable)
            convert(root / f"{name}_bag", cls.episodes / name, episode_id=name, condition="hazard" if cable else "clean",
                    mount_xyz_m=MOUNT, pitch_down_deg=PITCH, imu_frame="optical", frame_hz=10.0, image_scale=1.0,
                    hazards=["cable:3.0:0.05:0.03"] if cable else [], termination="goal_reached", cmd_timeout_s=0.5,
                    overrides={})
        cls.cable = load_episode(cls.episodes / "ep-cable-1")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_state_is_100hz_and_optical_imu_is_rotated_to_vertical(self):
        state = self.cable["state"]
        self.assertAlmostEqual(state[1]["t_s"] - state[0]["t_s"], 0.01, places=4)
        self.assertAlmostEqual(float(np.median([r["imu_az"] for r in state])), 9.81, delta=0.02)
        self.assertLess(abs(float(np.median([r["imu_ax"] for r in state]))), 0.02)

    def test_frames_are_10hz_and_saved(self):
        frames = self.cable["frames"]
        gaps = np.diff([f["t_s"] for f in frames])
        self.assertAlmostEqual(float(np.median(gaps)), 0.1, delta=0.04)
        self.assertTrue((self.cable["dir"] / frames[0]["rgb_path"]).exists())

    def test_route_starts_where_the_robot_starts_moving(self):
        route = self.cable["scenario"]["route"]
        self.assertAlmostEqual(route["start_pose"]["x_m"], 0.0, delta=0.02)
        self.assertGreater(route["length_m"], 5.0)

    def test_labeller_finds_the_cable_with_the_jackal_preset(self):
        rows, report, _ = build(self.episodes, Settings(**JACKAL))
        truth = next(e for e in report if e["episode_id"] == "ep-cable-1")["truth"]
        self.assertTrue(truth["hazards"][0]["found"])
        self.assertTrue(truth["hazards"][0]["unacceptable"])
        self.assertLess(truth["hazards"][0]["centre_error_m"], 0.08)
        self.assertEqual(truth["false_zones"], 0)
        clean_zeros = [r for r in rows if r["condition"] == "clean" and r["y"] == 0]
        self.assertEqual(clean_zeros, [])
        before = min((r for r in rows if r["episode_id"] == "ep-cable-1"), key=lambda r: abs(r["strip_lo_m"] - 2.4))
        self.assertEqual(before["y"], 0)


if __name__ == "__main__":
    unittest.main()
