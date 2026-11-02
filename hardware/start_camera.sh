#!/usr/bin/env bash
# Starts the D555 colour stream ON THE ROBOT in a tmux session (survives SSH drops).
# Verified 2026-10-28: D555 over USB 3.2, colour 640x480 @ 30 Hz, compressed topic at 30 Hz.
# The camera is not in /etc/clearpath/robot.yaml, so nothing starts it at boot: run this after every reboot.
#
#     bash start_camera.sh          # start (no-op if already running)
#     tmux attach -t btp_cam        # watch its log (detach: Ctrl+B then D)
#     tmux kill-session -t btp_cam  # stop
#
# Topics: /camera/camera/color/image_raw(/compressed), /camera/camera/color/camera_info.
# The camera is mounted upside down: convert with --rotate-180.

if tmux has-session -t btp_cam 2>/dev/null; then
  echo "camera already running (tmux session btp_cam)"
  exit 0
fi
mkdir -p "$HOME/btp"
tmux new-session -d -s btp_cam "bash -c 'set +u; source /opt/ros/jazzy/setup.bash; source /etc/clearpath/setup.bash; \
ros2 launch realsense2_camera rs_launch.py camera_namespace:=camera camera_name:=camera enable_color:=true \
enable_depth:=false enable_infra1:=false enable_infra2:=false pointcloud.enable:=false \
rgb_camera.color_profile:=640,480,30 2>&1 | tee \$HOME/btp/camera.log'"
echo "camera starting; check with: ros2 topic hz /camera/camera/color/image_raw/compressed"
