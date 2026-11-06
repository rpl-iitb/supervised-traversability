#!/usr/bin/env bash
# Checked: used for every recorded drive (sessions s1a-s1c, 30 Oct 2026).
#
# Records one drive ON THE ROBOT into ~/btp_bags/<name>_<time>/ (MCAP format).
# It finds the topics we need from whatever the robot publishes and leaves out
# depth and point clouds, which we do not use and which fill the disk quickly.
#
#     bash record_drive.sh clean_01
#     bash record_drive.sh cable_01 "cable 2 cm thick at 3.0 m"
#     bash record_drive.sh --dry-run test     # only print what it would record
#
# Hands-free: DURATION=40 bash record_drive.sh s1_r001   stops by itself after 40 s.
# Stop with Ctrl+C at the end of the drive. It then prints `ros2 bag info` so
# you can check on the spot that every topic is there.

set -u
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
NAME="${1:?usage: record_drive.sh [--dry-run] NAME [\"note\"]}"
NOTE="${2:-}"
MIN_FREE_GB=5

# ROS setup files read unset variables, so relax `set -u` while sourcing them.
set +u
for f in /opt/ros/jazzy/setup.bash /etc/clearpath/setup.bash; do
  # shellcheck disable=SC1090
  [ -f "$f" ] && source "$f"
done
set -u

# What we want, as patterns over the robot's own topic list. Order does not matter.
WANT=(
  'platform/odom/filtered$'       # pose + speed (EKF)
  'platform/odom$'                # wheel odometry
  'platform/motors/feedback$'     # per-motor current and measured speed (effort spikes, stalls)
  'platform/joint_states$'        # wheel positions and velocities
  'platform/mcu/status/stop$'     # e-stop / motor cutoff state
  'platform/bms/state$'           # battery
  'cmd_vel'                       # every velocity command (joystick, mux output)
  'sensors/imu_[0-9]+/data$'      # the Jackal's own IMU
  '/imu/data$'
  'camera.*imu$'                  # the camera's IMU, if it has one (for comparison)
  'color/camera_info$'
  'joy$'                          # joystick buttons (marks deadman / takeover moments)
  'navsat|gps|fix$'               # GPS, if present
  '^/tf$|/tf$'
  '^/tf_static$|/tf_static$'
  'diagnostics_agg$'
  'gonogo/(state|z|image/compressed)$'  # live go/no-go decisions and annotated frames, when the node runs
  'platform/safety_stop$'         # the lock the go/no-go gate holds
)
# Colour images: prefer compressed, fall back to raw.
IMAGE_COMPRESSED='color/(image_raw/)?compressed$'
IMAGE_RAW='color/image(_raw)?$'

ALL=$(ros2 topic list 2>/dev/null)
if [ -z "$ALL" ]; then
  echo "No ROS topics visible. Is the robot software running? Try: source /etc/clearpath/setup.bash" >&2
  exit 1
fi

TOPICS=()
for p in "${WANT[@]}"; do
  while IFS= read -r t; do [ -n "$t" ] && TOPICS+=("$t"); done < <(echo "$ALL" | grep -E "$p" | grep -v -E 'depth|points|parameter_events|rosout')
done
IMG=$(echo "$ALL" | grep -E "$IMAGE_COMPRESSED" | grep -v depth | head -n 1)
[ -z "$IMG" ] && IMG=$(echo "$ALL" | grep -E "$IMAGE_RAW" | grep -v depth | head -n 1)
[ -n "$IMG" ] && TOPICS+=("$IMG")
SORTED=$(printf '%s\n' "${TOPICS[@]}" | sort -u)
TOPICS=()
while IFS= read -r t; do [ -n "$t" ] && TOPICS+=("$t"); done <<< "$SORTED"

echo "Recording ${#TOPICS[@]} topics:"; printf '  %s\n' "${TOPICS[@]}"
for need in 'odom' 'imu' 'color' 'camera_info'; do
  printf '%s\n' "${TOPICS[@]}" | grep -q "$need" || echo "WARNING: nothing matching '$need' - check 'ros2 topic list'" >&2
done
[ "$DRY" = 1 ] && exit 0

FREE_GB=$(df -BG --output=avail "$HOME" | tail -n 1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "Only ${FREE_GB} GB free in $HOME; copy old bags off first." >&2
  exit 1
fi

mkdir -p "$HOME/btp_bags"
DIR="$HOME/btp_bags/${NAME}_$(date +%Y%m%d_%H%M%S)"
echo
echo "Saving to $DIR"
trap ':' INT   # let ros2 bag finish writing on Ctrl+C instead of killing this script
if [ -n "${DURATION:-}" ]; then
  echo "Recording for ${DURATION} s, then it stops by itself (Ctrl+C stops it early). Drive now."
  timeout -s INT "$DURATION" ros2 bag record -s mcap -o "$DIR" "${TOPICS[@]}"
else
  echo "Drive now. Press Ctrl+C when the robot has stopped at the end."
  ros2 bag record -s mcap -o "$DIR" "${TOPICS[@]}"
fi
trap - INT

# A small sidecar note next to the bag (rosbags ignores extra files).
{
  echo "name: $NAME"
  echo "recorded: $(date -Is)"
  echo "host: $(hostname)"
  echo "note: $NOTE"
} > "$DIR/drive_notes.txt"

echo
ros2 bag info "$DIR"
echo
echo "Copy to the laptop with:  rsync -av robot@<robot-ip>:btp_bags/ data/real_bags/"
