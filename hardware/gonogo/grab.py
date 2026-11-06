# Checked on the robot, 6 Nov 2026 (used for every live frame in the report).
"""Save one camera frame to /tmp/grab.jpg (default: the go/no-go's annotated view). Run on the robot:

    source /opt/ros/jazzy/setup.bash
    ~/btp/gonogo/venv/bin/python ~/btp/gonogo/grab.py [topic]
"""
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

topic = sys.argv[1] if len(sys.argv) > 1 else "/j100_0952/gonogo/image/compressed"
rclpy.init()
n = rclpy.create_node("grab")
got = []
n.create_subscription(CompressedImage, topic, lambda m: got.append(bytes(m.data)), qos_profile_sensor_data)
end = time.time() + 10
while not got and time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.2)
open("/tmp/grab.jpg", "wb").write(got[0])
