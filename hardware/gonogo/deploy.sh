#!/usr/bin/env bash
# Checked: used on 5-6 Nov 2026. ON THE MAC: copies the go/no-go folder (model 223 MB + wheels) to ~/btp/gonogo
# on the robot, plus the updated record_drive.sh. Type the robot's password when asked.
#
#     bash hardware/gonogo/deploy.sh <robot ip>
set -eu
IP="${1:?usage: deploy.sh <robot ip>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
rsync -av --progress --exclude venv --exclude __pycache__ --exclude '*.csv' "$HERE/" "robot@$IP:btp/gonogo/"
rsync -av "$HERE/../record_drive.sh" "robot@$IP:btp/"
echo "copied. On the robot: bash ~/btp/gonogo/setup_robot.sh"
