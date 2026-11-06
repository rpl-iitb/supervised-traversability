#!/usr/bin/env bash
# Checked on the robot, 6 Nov 2026.
#
# ON THE ROBOT, next to a running gate node: saves the annotated frame at every new STOP to
# ~/btp/gonogo_logs/stop_frames/ (and the log line to stops.txt). Runs until killed.
#
#     tmux new-session -d -s frames "bash ~/btp/gonogo/stop_frames.sh"
source /opt/ros/jazzy/setup.bash
mkdir -p ~/btp/gonogo_logs/stop_frames
tail -n0 -F ~/btp/gonogo_logs/console_gate.txt | grep --line-buffered "STOP .*<<<" | while read -r line; do
  ~/btp/gonogo/venv/bin/python ~/btp/gonogo/grab.py && cp /tmp/grab.jpg ~/btp/gonogo_logs/stop_frames/stop_$(date +%H%M%S).jpg
  echo "$line" >> ~/btp/gonogo_logs/stop_frames/stops.txt
done
