#!/usr/bin/env bash
# Checked on the robot, 5-6 Nov 2026.
#
# ON THE ROBOT, once: installs the go/no-go packages (onnxruntime, Pillow) into ~/btp/gonogo/venv, offline, from the
# wheels copied next to this script. Nothing is installed system-wide. The venv sees the system's
# numpy and ROS (rclpy), so no numpy is installed: a second numpy would break ROS messages.
# Then it times the model on the robot's CPU.
#
#     bash ~/btp/gonogo/setup_robot.sh

set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/venv"
PIPWHL="$(ls "$HERE"/wheels/pip-*.whl | head -n 1)"
WHEELS=("$HERE"/wheels/onnxruntime-*.whl "$HERE"/wheels/pillow-*.whl "$HERE"/wheels/flatbuffers-*.whl
        "$HERE"/wheels/packaging-*.whl "$HERE"/wheels/protobuf-*.whl)

# python3.12-venv (ensurepip) is not installed on this robot, but a venv without pip still works; pip runs from its wheel.
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --without-pip --system-site-packages "$VENV"
fi
"$VENV/bin/python" "$PIPWHL/pip" install --quiet --no-index --no-deps --ignore-installed "${WHEELS[@]}"

set +u
source /opt/ros/jazzy/setup.bash
[ -f /etc/clearpath/setup.bash ] && source /etc/clearpath/setup.bash
set -u
cd "$HERE"
"$VENV/bin/python" - <<'EOF'
import time, numpy as np, onnxruntime, PIL, rclpy
from sensor_msgs.msg import CompressedImage
from gonogo_core import Model
print(f"numpy {np.__version__}, onnxruntime {onnxruntime.__version__}, Pillow {PIL.__version__}, rclpy ok")
m = Model()
x = np.full((112, 336, 3), 127, np.uint8)
m.score(x)
ms = [m.score(x)[2] for _ in range(30)]
print(f"model on this CPU: {np.median(ms):.0f} ms per frame (median of 30), slowest {max(ms):.0f} ms")
EOF
echo "setup OK. Next: bash $HERE/run_gonogo.sh shadow"
