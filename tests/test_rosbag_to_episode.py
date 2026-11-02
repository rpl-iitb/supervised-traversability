import io
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore
except ImportError:  # pragma: no cover
    Writer = None

from scripts.episode_io import load_episode
from scripts.frame_labels_v2 import JACKAL, Settings, build
from scripts.rosbag_to_episode import clearpath_typestore, convert, optical_to_base, read_bag

PITCH = 15.0
MOUNT = (0.2, 0.0, 0.3)
CABLE_AT = 3.0

# ROS 1 style names, D435i IMU in the optical frame at 200 Hz, unstamped commands.
REALSENSE_LAYOUT = dict(
    odom="/odometry/filtered", wheel="/jackal_velocity_controller/odom", cmd="/cmd_vel", cmd_stamped=False,
    imu="/camera/imu", imu_frame="optical", imu_every=1, img="/camera/color/image_raw/compressed",
    info="/camera/color/camera_info", decoys=(), mcap=False, motors=None)
# What a Clearpath ROS 2 Jazzy Jackal publishes (namespaced, stamped commands, its own IMU at 50 Hz,
# camera under sensors/camera_0), recorded as MCAP, plus look-alike topics the converter must not pick.
CLEARPATH_JAZZY_LAYOUT = dict(
    odom="/j100_0434/platform/odom/filtered", wheel="/j100_0434/platform/odom", cmd="/j100_0434/cmd_vel",
    cmd_stamped=True, imu="/j100_0434/sensors/imu_0/data", imu_frame="base", imu_every=4,
    img="/j100_0434/sensors/camera_0/color/compressed", info="/j100_0434/sensors/camera_0/color/camera_info",
    decoys=(("/j100_0434/sensors/camera_0/depth/compressed", "sensor_msgs/msg/CompressedImage"),
            ("/j100_0434/sensors/camera_0/depth/camera_info", "sensor_msgs/msg/CameraInfo")),
    mcap=True, motors="/j100_0434/platform/motors/feedback")

# Clearpath's definitions verbatim (clearpath_msgs, branch jazzy), as the robot's MCAP embeds them.
UPSTREAM_MSGS = {
    "clearpath_platform_msgs/msg/DriveFeedback": """# This message represents feedback data from a single drive unit (driver + motor).

# Current flowing from battery into the MOSFET bridge.
float32 current

# Instantaneous duty cycle of MOSFET bridge.
float32 duty_cycle

# Temperatures measured in the MOSFET bridge and on the motor casing, in deg C.
float32 bridge_temperature
float32 motor_temperature

# Encoder data
float32 measured_velocity   # rad/s
float32 measured_travel     # rad

# True if the underlying driver chip reports a fault condition.
bool driver_fault
""",
    "clearpath_platform_msgs/msg/Feedback": """# This message represents high-frequency feedback from the MCU,
# as necessary to support closed-loop control and thermal monitoring.
# Default publish frequency is 50Hz.

std_msgs/Header header

DriveFeedback[2] drivers

# Commanded control mode, use the TYPE_ constants from jackal_msgs/Drive.
int8 commanded_mode

# Actual control mode. This may differ from the commanded in cases where
# the motor enable is off, the motors are in over-current, etc.
int8 actual_mode
""",
}
BASE_CURRENT_A, SPIKE_A, WHEEL_RADIUS_M = 2.0, 3.0, 0.098


def upstream_typestore():
    from rosbags.typesys import get_types_from_msg

    ts = get_typestore(Stores.ROS2_JAZZY)
    types = {}
    for name, text in UPSTREAM_MSGS.items():
        types.update(get_types_from_msg(text, name))
    ts.register(types)
    return ts


def write_bag(path: Path, cable: bool, layout: dict = REALSENSE_LAYOUT, *, shake: bool | None = None,
              current_spike: bool | None = None) -> None:
    """Jackal driving straight along +x at 0.5 m/s; over a cable the IMU shakes and, if the layout
    records motor feedback, both motors draw a current spike. ``shake``/``current_spike`` default to ``cable``."""
    from PIL import Image

    shake_on = cable if shake is None else shake
    spike_on = cable if current_spike is None else current_spike
    ts = upstream_typestore()
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
    L = layout
    cmd_type = "geometry_msgs/msg/TwistStamped" if L["cmd_stamped"] else "geometry_msgs/msg/Twist"
    storage = StoragePlugin.MCAP if L["mcap"] else StoragePlugin.SQLITE3
    with Writer(path, version=8, storage_plugin=storage) as writer:
        c = {name: writer.add_connection(topic, msgtype, typestore=ts) for name, topic, msgtype in (
            ("odom", L["odom"], "nav_msgs/msg/Odometry"),
            ("wheel", L["wheel"], "nav_msgs/msg/Odometry"),
            ("cmd", L["cmd"], cmd_type),
            ("imu", L["imu"], "sensor_msgs/msg/Imu"),
            ("img", L["img"], "sensor_msgs/msg/CompressedImage"),
            ("info", L["info"], "sensor_msgs/msg/CameraInfo"),
            *((f"decoy{i}", topic, msgtype) for i, (topic, msgtype) in enumerate(L["decoys"])),
            *((("motors", L["motors"], "clearpath_platform_msgs/msg/Feedback"),) if L.get("motors") else ()))}

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
                if L.get("motors"):  # 50 Hz motor feedback; the wheels climb the cable at +/-0.20 m
                    n = k // 4
                    spike = spike_on and any(abs(x - b) < 0.05 for b in (CABLE_AT - 0.20, CABLE_AT + 0.20))
                    current = (BASE_CURRENT_A if moving else 0.3) + (SPIKE_A if spike else 0.0) + 0.02 * math.sin(n)
                    drive = lambda sign: T["clearpath_platform_msgs/msg/DriveFeedback"](
                        current=current, duty_cycle=0.2, bridge_temperature=30.0, motor_temperature=30.0,
                        measured_velocity=sign * v / WHEEL_RADIUS_M, measured_travel=x / WHEEL_RADIUS_M, driver_fault=False)
                    feedback = T["clearpath_platform_msgs/msg/Feedback"](header=header(t, ""), drivers=[drive(1.0), drive(1.0)],
                                                                          commanded_mode=1, actual_mode=1)
                    put("motors", t, feedback, "clearpath_platform_msgs/msg/Feedback")
            if k % 10 == 0:
                twist = T["geometry_msgs/msg/Twist"](linear=Vec(x=0.5, y=0.0, z=0.0), angular=Vec(x=0.0, y=0.0, z=0.0))
                if L["cmd_stamped"]:
                    twist = T["geometry_msgs/msg/TwistStamped"](header=header(t, "base_link"), twist=twist)
                put("cmd", t, twist, cmd_type)
            if k % L["imu_every"] == 0:
                n = k // L["imu_every"]
                shake = shake_on and any(abs(x - b) < 0.03 for b in (CABLE_AT - 0.20, CABLE_AT + 0.20))
                az = 9.81 + (2.0 * (1 if n % 2 else -1) if shake else 0.0) + 0.01 * math.sin(n)
                a = to_optical @ np.array([0.0, 0.0, az]) if L["imu_frame"] == "optical" else np.array([0.0, 0.0, az])
                imu = T["sensor_msgs/msg/Imu"](header=header(t, "imu_link"), orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                                              orientation_covariance=cov9, angular_velocity=Vec(x=0.0, y=0.0, z=0.0),
                                              angular_velocity_covariance=cov9,
                                              linear_acceleration=Vec(x=float(a[0]), y=float(a[1]), z=float(a[2])),
                                              linear_acceleration_covariance=cov9)
                put("imu", t, imu, "sensor_msgs/msg/Imu")
            if k % 7 == 0:  # ~29 Hz camera
                image = T["sensor_msgs/msg/CompressedImage"](header=header(t, "camera"), format="jpeg", data=jpeg)
                put("img", t, image, "sensor_msgs/msg/CompressedImage")
                for i, (_, msgtype) in enumerate(L["decoys"]):
                    if msgtype.endswith("CompressedImage"):
                        put(f"decoy{i}", t, image, msgtype)
            if k == 0:
                info = T["sensor_msgs/msg/CameraInfo"](
                    header=header(t, "camera"), height=48, width=64, distortion_model="plumb_bob", d=np.zeros(5),
                    k=np.array([40.0, 0, 32, 0, 40.0, 24, 0, 0, 1]), r=np.eye(3).ravel(), p=np.zeros(12), binning_x=0, binning_y=0,
                    roi=T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
                put("info", t, info, "sensor_msgs/msg/CameraInfo")
                for i, (_, msgtype) in enumerate(L["decoys"]):
                    if msgtype.endswith("CameraInfo"):
                        put(f"decoy{i}", t, info, msgtype)
            x += v / 200


@unittest.skipIf(Writer is None, "rosbags not installed")
class ConvertTest(unittest.TestCase):
    LAYOUT = REALSENSE_LAYOUT

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.episodes = root / "episodes"
        for name, cable in (("ep-clean-1", False), ("ep-clean-2", False), ("ep-cable-1", True)):
            write_bag(root / f"{name}_bag", cable, cls.LAYOUT)
            convert(root / f"{name}_bag", cls.episodes / name, episode_id=name, condition="hazard" if cable else "clean",
                    mount_xyz_m=MOUNT, pitch_down_deg=PITCH, imu_frame=cls.LAYOUT["imu_frame"], frame_hz=10.0, image_scale=1.0,
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


class ClearpathJazzyConvertTest(ConvertTest):
    """Same checks on the layout the lab's Jackal should produce (ROS 2 Jazzy, MCAP, Jackal IMU at 50 Hz)."""

    LAYOUT = CLEARPATH_JAZZY_LAYOUT

    def test_picks_the_colour_camera_and_the_jackal_imu_not_the_look_alikes(self):
        topics = self.cable["meta"]["source"]["topics"]
        self.assertEqual(topics["image"], "/j100_0434/sensors/camera_0/color/compressed")
        self.assertEqual(topics["camera_info"], "/j100_0434/sensors/camera_0/color/camera_info")
        self.assertEqual(topics["imu"], "/j100_0434/sensors/imu_0/data")
        self.assertEqual(topics["odom"], "/j100_0434/platform/odom/filtered")
        self.assertEqual(topics["cmd"], "/j100_0434/cmd_vel")

    def test_labeller_finds_the_cable_with_the_jackal_preset(self):
        # With only two identical synthetic clean drives the "abnormal" level equals their exact
        # maximum, so a 0.25 % noise blip at the end of the 50 Hz drive becomes one gray zone.
        # What matters is that nothing false is judged unacceptable and clean photos stay clean.
        rows, report, _ = build(self.episodes, Settings(**JACKAL))
        cable = next(e for e in report if e["episode_id"] == "ep-cable-1")
        hazard = cable["truth"]["hazards"][0]
        self.assertTrue(hazard["found"])
        self.assertTrue(hazard["unacceptable"])
        self.assertLess(hazard["centre_error_m"], 0.08)
        self.assertEqual([z for z in cable["zones"] if z["unacceptable"] and not 2.8 <= (z["lo"] + z["hi"]) / 2 <= 3.2], [])
        self.assertEqual([r for r in rows if r["condition"] == "clean" and r["y"] == 0], [])

    def test_stamped_commands_are_read(self):
        moving = [r for r in self.cable["state"] if float(r["t_s"]) > 2.0]
        self.assertAlmostEqual(float(np.median([float(r["cmd_v_mps"]) for r in moving])), 0.5, places=3)


@unittest.skipIf(Writer is None, "rosbags not installed")
class MotorFeedbackTest(unittest.TestCase):
    """Clearpath motor current: decoded from the MCAP's embedded definition, resampled, used as 'effort'."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.episodes = root / "episodes"
        # ep-lip-1: the motors strain but the IMU stays quiet, so only the effort signal can find it.
        drives = (("ep-clean-1", False, {}), ("ep-clean-2", False, {}), ("ep-lip-1", True, {"shake": False}))
        for name, cable, extra in drives:
            write_bag(root / f"{name}_bag", cable, CLEARPATH_JAZZY_LAYOUT, **extra)
            convert(root / f"{name}_bag", cls.episodes / name, episode_id=name, condition="hazard" if cable else "clean",
                    mount_xyz_m=MOUNT, pitch_down_deg=PITCH, imu_frame="base", frame_hz=10.0, image_scale=1.0,
                    hazards=["lip:3.0:0.05:0.03"] if cable else [], termination="goal_reached", cmd_timeout_s=0.5,
                    overrides={})
        cls.bag = root / "ep-lip-1_bag"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_feedback_is_decoded_from_the_bag_and_resampled(self):
        lip = load_episode(self.episodes / "ep-lip-1")
        self.assertEqual(lip["meta"]["source"]["topics"]["motors"], CLEARPATH_JAZZY_LAYOUT["motors"])
        moving = [r for r in lip["state"] if r["t_s"] > 2.0]
        self.assertAlmostEqual(float(np.median([r["motor_current_l_a"] for r in moving])), BASE_CURRENT_A, delta=0.05)
        self.assertAlmostEqual(float(np.max([r["motor_current_r_a"] for r in moving])), BASE_CURRENT_A + SPIKE_A, delta=0.1)
        self.assertAlmostEqual(float(np.median([r["motor_vel_l_rps"] for r in moving])), 0.5 / WHEEL_RADIUS_M, delta=0.01)

    def test_bundled_definitions_decode_the_upstream_layout(self):
        upstream = upstream_typestore()
        T = upstream.types
        drive = T["clearpath_platform_msgs/msg/DriveFeedback"](current=1.25, duty_cycle=0.5, bridge_temperature=31.0,
                                                               motor_temperature=32.0, measured_velocity=-4.5,
                                                               measured_travel=7.0, driver_fault=True)
        msg = T["clearpath_platform_msgs/msg/Feedback"](
            header=T["std_msgs/msg/Header"](stamp=T["builtin_interfaces/msg/Time"](sec=3, nanosec=4), frame_id="x"),
            drivers=[drive, drive], commanded_mode=1, actual_mode=-1)
        raw = upstream.serialize_cdr(msg, "clearpath_platform_msgs/msg/Feedback")
        decoded = clearpath_typestore().deserialize_cdr(raw, "clearpath_platform_msgs/msg/Feedback")
        self.assertEqual((decoded.drivers[1].measured_velocity, decoded.drivers[0].driver_fault, decoded.actual_mode),
                         (-4.5, True, -1))

    def test_read_bag_returns_feedback_messages(self):
        msgs = read_bag(self.bag, {"motors": CLEARPATH_JAZZY_LAYOUT["motors"]})
        self.assertEqual(len(msgs["motors"]), 12 * 50)
        self.assertEqual(len(msgs["motors"][0][1].drivers), 2)

    def test_effort_finds_a_hazard_the_imu_misses(self):
        rows, report, thresholds = build(self.episodes, Settings(**JACKAL))
        self.assertEqual(thresholds["clean_drives_with_effort"], 2)
        self.assertAlmostEqual(thresholds["effort_baseline_a"], BASE_CURRENT_A, delta=0.05)
        lip = next(e for e in report if e["episode_id"] == "ep-lip-1")
        hazard = lip["truth"]["hazards"][0]
        self.assertTrue(hazard["found"])
        self.assertTrue(hazard["unacceptable"])
        self.assertLess(hazard["centre_error_m"], 0.08)
        zone = next(z for z in lip["zones"] if z["unacceptable"])
        self.assertEqual(zone["cause"], "effort")
        self.assertAlmostEqual(zone["effort_ratio"], (BASE_CURRENT_A + SPIKE_A) / BASE_CURRENT_A, delta=0.1)
        self.assertEqual([r for r in rows if r["condition"] == "clean" and r["y"] == 0], [])

    def test_effort_is_ignored_while_accelerating_or_turning(self):
        from scripts.frame_labels_v2 import effort_signal

        t = np.arange(0, 6, 0.01)
        cmd = np.where(t < 3, 0.2, 0.5)                    # command steps up at t = 3 s
        turn = np.where((t > 4.5) & (t < 5), 0.5, 0.0)
        current = np.full(len(t), 2.0)
        state = [{"motor_current_l_a": c, "motor_current_r_a": c, "cmd_w_rps": w} for c, w in zip(current, turn)]
        col = lambda name: np.array([row[name] for row in state], dtype=float)
        effort, steady = effort_signal(state, t, cmd, col, np.ones(len(t), bool), 100, Settings())
        self.assertTrue(steady[(t > 1.0) & (t < 2.9)].all())
        self.assertFalse(steady[(t > 3.0) & (t < 3.9)].any())   # within effort_settle_s of the step
        self.assertFalse(steady[(t > 4.6) & (t < 4.9)].any())   # turning
        self.assertTrue(steady[t > 5.6].all())
        self.assertEqual(float(effort[~steady].max()), 0.0)
        self.assertEqual(effort_signal([{"x": 1}], t[:1], cmd[:1], col, np.ones(1, bool), 100, Settings()), (None, None))


if __name__ == "__main__":
    unittest.main()
