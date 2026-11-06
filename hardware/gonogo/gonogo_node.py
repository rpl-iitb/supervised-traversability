# Checked on the robot, 5-6 Nov 2026 (live trials in the report, Section 5.10).
"""Live go/no-go on the robot: camera -> model -> STOP lock. Run it with run_gonogo.sh, not directly.

    bash run_gonogo.sh shadow     # shows GO/STOP and logs it; never touches the motors
    bash run_gonogo.sh gate       # also holds the robot's safety lock while the decision is STOP

How the gate stops the robot: twist_mux already has a lock input, /<ns>/platform/safety_stop (std_msgs/Bool,
priority 254, nobody else publishes it). True blocks every velocity command (joystick included); false lets them
through. The operator drives as always (hold L1). This node never sends a velocity command itself: it can only
stop the robot, never move it.

Gate mode:
- starts with the lock ON and releases it only once camera frames arrive and the model says GO;
- after 3 flagged frames in a row it locks and stays locked;
- holding the override button (default: triangle, button 2) releases the lock so the robot can back away; letting go
  re-arms the rule;
- no camera frame for 0.5 s -> lock ON;
- Ctrl+C releases the lock before exiting (normal joystick driving again). If the node is killed hard, release by
  hand: ros2 topic pub --once /j100_0952/platform/safety_stop std_msgs/msg/Bool "{data: false}"

Topics published: <ns>/gonogo/state (String), <ns>/gonogo/z (Float32), <ns>/gonogo/image/compressed (annotated
frame for Foxglove, 5 Hz). Every decision is also logged to a CSV in --log-dir.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gonogo_core import Model, StopRule, annotate  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rclpy.signals import SignalHandlerOptions  # noqa: E402
from sensor_msgs.msg import CompressedImage, Joy  # noqa: E402
from std_msgs.msg import Bool, Float32, String  # noqa: E402


class GoNoGo(Node):
    def __init__(self, args):
        super().__init__("gonogo")
        self.args = args
        self.gate = args.mode == "gate"
        self.model = Model(threads=args.threads)
        self.rule = StopRule(args.margin, args.consecutive, latch=self.gate)
        self.latest = None  # (receive time, msg)
        self.override = False
        self.buttons_seen = set()
        self.state, self.tick = "WAIT", 0
        ns = args.ns.rstrip("/")
        self.create_subscription(CompressedImage, args.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(Joy, f"{ns}/joy_teleop/joy", self.on_joy, qos_profile_sensor_data)
        self.pub_state = self.create_publisher(String, f"{ns}/gonogo/state", 10)
        self.pub_z = self.create_publisher(Float32, f"{ns}/gonogo/z", 10)
        self.pub_image = self.create_publisher(CompressedImage, f"{ns}/gonogo/image/compressed", 1)
        self.pub_lock = self.create_publisher(Bool, f"{ns}/platform/safety_stop", 10) if self.gate else None
        self.set_lock(True)
        log_dir = Path(args.log_dir).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"gonogo_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.log_file = path.open("w", newline="")
        self.log = csv.writer(self.log_file)
        self.log.writerow(["wall_time", "frame_age_ms", "infer_ms", "z", "z_folds", "flagged", "state", "lock",
                           "override"])
        self.last_print = 0.0
        self.create_timer(1.0 / args.rate, self.step)
        self.get_logger().info(f"mode {args.mode}; margin {args.margin}, {args.consecutive} frames; "
                               f"override button {args.override_button}; log {path}")

    def on_image(self, msg):
        self.latest = (time.monotonic(), msg)

    def on_joy(self, msg):
        pressed = {i for i, b in enumerate(msg.buttons) if b}
        for i in pressed - self.buttons_seen:
            self.get_logger().info(f"button {i} pressed")
        self.buttons_seen = pressed
        self.override = len(msg.buttons) > self.args.override_button and bool(msg.buttons[self.args.override_button])

    def set_lock(self, on: bool):
        if self.pub_lock is not None:
            self.pub_lock.publish(Bool(data=on))
        self.lock = on

    def step(self):
        if self.latest is None or time.monotonic() - self.latest[0] > self.args.max_age:
            self.report("NO CAMERA", float("nan"), None, 0.0, float("nan"), False)
            return
        received, msg = self.latest
        try:
            image, x = self.model.preprocess(bytes(msg.data))
            z, z_folds, ms = self.model.score(x)
        except Exception as error:  # a bad frame must never leave the robot unlocked
            self.get_logger().error(f"model failed: {error}")
            self.report("ERROR", float("nan"), None, 0.0, float("nan"), False)
            return
        if self.override:
            self.rule.reset()
            flagged, state = z < self.args.margin, "OVERRIDE"
        else:
            flagged = self.rule.update(z)
            state = "STOP" if self.rule.stopped else "GO"
        self.report(state, z, z_folds, ms, (time.monotonic() - received) * 1000.0, flagged)
        self.tick += 1
        if self.tick % max(1, round(self.args.rate / 5)) == 0:
            out = CompressedImage(format="jpeg", data=annotate(image, self.model.mask, state, z))
            out.header.stamp = msg.header.stamp
            self.pub_image.publish(out)

    def report(self, state, z, z_folds, ms, age_ms, flagged):
        self.set_lock(state not in ("GO", "OVERRIDE"))
        self.pub_state.publish(String(data=state))
        if z == z:
            self.pub_z.publish(Float32(data=z))
        self.log.writerow([f"{time.time():.3f}", f"{age_ms:.0f}", f"{ms:.1f}", f"{z:.3f}",
                           "" if z_folds is None else " ".join(f"{v:.2f}" for v in z_folds), int(flagged), state,
                           int(self.lock), int(self.override)])
        now = time.monotonic()
        if state != self.state or now - self.last_print > 1.0:
            mark = " <<<" if state != self.state else ""
            self.get_logger().info(f"{state:9s} z {z:+6.2f}  model {ms:4.0f} ms  frame age {age_ms:4.0f} ms  "
                                   f"lock {'ON ' if self.lock else 'off'}{mark}")
            self.state, self.last_print = state, now

    def release(self):
        if self.pub_lock is not None:
            for _ in range(5):
                self.pub_lock.publish(Bool(data=False))
                time.sleep(0.05)
            self.get_logger().info("lock released: normal joystick driving")
        self.log_file.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("shadow", "gate"))
    ap.add_argument("--ns", default="/j100_0952")
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw/compressed")
    ap.add_argument("--rate", type=float, default=10.0, help="decisions per second (training frames were 10 Hz)")
    ap.add_argument("--margin", type=float, default=-1.0)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--override-button", type=int, default=2)
    ap.add_argument("--max-age", type=float, default=0.5, help="seconds without a camera frame before locking")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log-dir", default="~/btp/gonogo_logs")
    args = ap.parse_args()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    def stop_signal(*_):  # SSH drop or kill: still release the lock on the way out
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGHUP, stop_signal)
    node = GoNoGo(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.release()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
