#!/usr/bin/env bash
# UNVERIFIED: written before we had access to the robot. It only reads; it never
# changes anything on the robot. Check its output on the first lab visit.
#
# Collects everything we need to know about the Jackal into one text file.
# Run ON THE ROBOT (after `ssh robot@<robot-ip>`), robot powered on and idle:
#
#     bash robot_info.sh            # ~2 minutes, writes ~/btp_robot_info_<time>.txt
#
# Copy it to the laptop afterwards:
#
#     scp robot@<robot-ip>:btp_robot_info_*.txt hardware/
#
# Drive the robot slowly with the joystick while the "rates" section runs if you
# also want the cmd_vel rates (they are silent while nobody is driving).

set -u
OUT="$HOME/btp_robot_info_$(date +%Y%m%d_%H%M%S).txt"
HZ_SECONDS="${HZ_SECONDS:-6}"
export PYTHONUNBUFFERED=1

section() { printf '\n\n===== %s =====\n' "$1"; }
run() {  # run "title" command...  (never stops the script on failure)
  printf '\n--- %s\n$ %s\n' "$1" "${*:2}"
  "${@:2}" 2>&1 || printf '(exit %s)\n' "$?"
}

# ROS environment: Clearpath's setup file sets the namespace, RMW and discovery.
# ROS setup files read unset variables, so relax `set -u` while sourcing them.
set +u
for f in /opt/ros/jazzy/setup.bash /etc/clearpath/setup.bash; do
  # shellcheck disable=SC1090
  [ -f "$f" ] && source "$f"
done
set -u

{
  echo "BTP robot info, $(date -Is), host $(hostname)"

  section "system"
  run "OS" cat /etc/os-release
  run "kernel" uname -a
  run "uptime" uptime
  run "time sync (matters once the laptop is in the loop)" timedatectl

  section "compute"
  run "CPU" lscpu
  run "memory" free -h
  run "disk" df -h / "$HOME"
  run "PCI devices (any GPU shows here)" sh -c "lspci | grep -i -E 'vga|3d|display|nvidia' || echo 'no GPU found'"
  command -v nvidia-smi >/dev/null && run "nvidia-smi" nvidia-smi
  run "USB devices" lsusb
  run "joystick devices" sh -c "ls -l /dev/input/js* 2>/dev/null || echo 'no /dev/input/js*'"
  run "python" python3 --version

  section "network"
  run "interfaces" ip -br addr
  run "routes" ip route
  command -v nmcli >/dev/null && run "wifi" nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device

  section "camera"
  command -v rs-enumerate-devices >/dev/null && run "RealSense devices (model, firmware, IMU?)" rs-enumerate-devices -s
  command -v rs-enumerate-devices >/dev/null && run "RealSense streams" sh -c "rs-enumerate-devices | grep -i -E 'name|serial|firmware|usb|accel|gyro|motion|color' | head -60"

  section "clearpath config"
  run "robot.yaml" cat /etc/clearpath/robot.yaml
  run "clearpath services" sh -c "systemctl list-units --all --no-pager | grep -i -E 'clearpath|ros|realsense|foxglove' || true"

  section "ROS 2 environment"
  run "ROS variables" sh -c "printenv | grep -E '^(ROS|RMW|FASTRTPS|FASTDDS|CYCLONE|ZENOH)' | sort"
  run "installed packages we may need" sh -c "ros2 pkg list | grep -E 'realsense|foxglove|rosbridge|twist_mux|teleop|nav2|robot_localization|image_transport|compressed|rosbag2_storage_mcap' || true"

  section "ROS 2 graph"
  run "nodes" ros2 node list
  run "topics with types" ros2 topic list -t
  run "cmd_vel wiring (who publishes / subscribes)" sh -c 'for t in $(ros2 topic list | grep -E "cmd_vel"); do echo "## $t"; ros2 topic info -v "$t" | grep -E "Type|Node name|Endpoint type|Publisher count|Subscription count"; done'
  for node in $(ros2 node list 2>/dev/null | grep -E 'twist_mux'); do
    run "twist_mux parameters ($node)" ros2 param dump "$node"
  done

  section "rates (${HZ_SECONDS} s each)"
  for t in $(ros2 topic list 2>/dev/null | grep -E 'imu|odom|cmd_vel|joy|camera_info|color/(image|compressed)|image_raw/compressed|gps|fix|navsat' | grep -v -E 'parameter_events|rosout'); do
    printf '\n--- %s\n' "$t"
    timeout -s INT "$HZ_SECONDS" ros2 topic hz "$t" 2>&1 | grep -E 'average rate|min:' | tail -n 2 || true
  done

  section "one message of each key topic"
  for t in $(ros2 topic list 2>/dev/null | grep -E 'sensors/imu_0/data$|/imu/data$|platform/odom/filtered$|color/camera_info$|navsat|gps.*fix'); do
    printf '\n--- %s\n' "$t"
    timeout 5 ros2 topic echo --once --no-arr "$t" 2>&1 | head -60
  done

  echo; echo "done: $(date -Is)"
} > "$OUT" 2>&1

echo "Wrote $OUT"
echo "Copy it to the laptop with:  scp robot@<robot-ip>:$(basename "$OUT") hardware/"
