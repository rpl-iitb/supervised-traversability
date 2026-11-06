#!/usr/bin/env bash
# Checked on the robot, 5-6 Nov 2026.
#
# ON THE ROBOT: starts the live go/no-go node. Ctrl+C stops it (and releases the safety lock).
#
#     bash ~/btp/gonogo/run_gonogo.sh shadow            # decisions only, the motors are never touched
#     bash ~/btp/gonogo/run_gonogo.sh gate              # STOP holds the robot's safety lock
#     bash ~/btp/gonogo/run_gonogo.sh gate --margin -2  # extra options: see gonogo_node.py --help
#
# Watch in Foxglove (ws://<robot ip>:8765): /j100_0952/gonogo/image/compressed, /j100_0952/gonogo/state.
# If the robot ever stays locked after the node is gone:
#     ros2 topic pub --once /j100_0952/platform/safety_stop std_msgs/msg/Bool "{data: false}"

HERE="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/jazzy/setup.bash
[ -f /etc/clearpath/setup.bash ] && source /etc/clearpath/setup.bash
exec "$HERE/venv/bin/python" "$HERE/gonogo_node.py" "$@"
