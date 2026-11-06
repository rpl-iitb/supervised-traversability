# Live go/no-go on the Jackal

The 5-fold ResNet-18 ensemble (masked floor input) runs on the robot's CPU and stops the robot with its own safety
lock. It was run on the robot on 5-6 Nov 2026; the results are in `data/gonogo_logs/live_trials.csv` and in report
Section 5.10. Each file's first line says how it was checked.

## How it stops the robot
- The node scores 10 frames per second.
- After 3 frames in a row with `z < -1`, it publishes `true` on `/j100_0952/platform/safety_stop`. twist_mux already
  treats that topic as a lock with priority 254 and no timeout, so every drive command is dropped, the joystick
  included. Gate mode then stays locked until the override button is held.
- The node never sends a velocity command, so it cannot move the robot.
- With no camera frame for 0.5 s, or after a model error, it locks.
- On Ctrl+C or SIGTERM it publishes `false`. If it is killed hard, unlock by hand:
  `ros2 topic pub --once /j100_0952/platform/safety_stop std_msgs/msg/Bool "{data: false}"`

## Files
| File | Where it runs | What it does |
|---|---|---|
| `gonogo_resnet18.onnx` | robot | the ensemble: uint8 input 1x112x336x3; outputs `z` (mean over folds of logit - tau) and `z_folds` |
| `floor_mask.png` | robot | the floor strip (320x240) the model sees |
| `gonogo_core.py` | both | preprocessing (same as training), scoring, stop rule, annotated image |
| `gonogo_node.py` | robot | the ROS 2 node (`shadow` = decisions only; `gate` = holds the lock) |
| `setup_robot.sh` | robot, once | venv with onnxruntime and Pillow from `wheels/` (offline, nothing system-wide); times the model |
| `run_gonogo.sh` | robot | starts the node |
| `grab.py`, `stop_frames.sh` | robot | save the current frame or the frame at every STOP |
| `deploy.sh` | Mac | copies this folder to `~/btp/gonogo` on the robot |
| `replay_bag.py` | Mac | runs the stop rule over a recorded bag |
| `report_live_figures.py` | Mac | report figures and tables from the copied logs |
| `LAB_GONOGO.md` | - | the lab runbook |

## Run it
On the Mac:

    bash hardware/gonogo/deploy.sh <robot-ip>

On the robot (start the camera after every reboot):

    bash ~/btp/start_camera.sh
    bash ~/btp/gonogo/setup_robot.sh                                # once; prints ms per frame
    bash ~/btp/gonogo/run_gonogo.sh shadow                          # watch only
    bash ~/btp/gonogo/run_gonogo.sh gate --override-button 3        # holds the lock

To watch, open Foxglove at `ws://<robot-ip>:8765` and view `/j100_0952/gonogo/image/compressed`. Each run writes a
CSV of every decision to `~/btp/gonogo_logs/`.

## Lessons from the lab
- **Override button.** On our PS4 controller, triangle is `buttons[3]`, not 2 (the code's default is 2). The node logs
  `button N pressed` for every press, so you can check.
- **Controller mode.** The controller must be in controller mode. Otherwise `platform/emergency_stop` stays `true`
  and nothing drives.
- **Wi-Fi changes.** The robot can join ERTS_5G or ERTS_link_5G, and in the corridor it switches between them. If its
  IP address changes after boot, ROS 2 processes started before the change and after it no longer see each other.
  The node then logs NO CAMERA, or worse, its lock no longer reaches twist_mux. Reboot on the network you will use
  and start the node after that. Check with `ros2 topic info -v /j100_0952/platform/safety_stop`: it must list the
  `gonogo` publisher and the `twist_mux` subscriber.
- **Reaction time.** About 0.9 s from frame to braking. 0.5 s of that is the controller's `cmd_vel_timeout`. In our
  tests it stopped in time only up to about 1.2 m/s, so drive without turbo.
