# Supervised traversability from the robot's own IMU

Can a robot learn, from **one forward-facing camera**, whether the floor just ahead is safe to drive over? Here the
labels do not come from people. They come from what the robot later **felt** when it drove over that floor: vertical
shaking and roll rate from its IMU, compared with the levels measured on clean floor.

This repository holds the full pipeline:

1. **Simulation.** A MuJoCo proxy warehouse with cables, lips, spills and wrappers builds and checks the labeller
   against known object positions. It also includes a re-implementation of **GONet**, an earlier self-supervised
   baseline, checked against the authors' released model.
2. **Real robot.** Recordings from a **Clearpath Jackal J100** with an Intel RealSense D555, in 36 runs with nine
   everyday objects, are converted, labelled and used to train camera models. The best is a fine-tuned ResNet-18 on
   a masked floor strip: AUROC 0.967 on held-out runs, against 0.892 for GONet retrained on the same data.
3. **Live go/no-go.** The 5-fold ResNet-18 ensemble runs on the robot's CPU (about 37 ms per frame, no GPU) as a
   safety filter. It stops the robot through its own safety lock, for example 0.90 m before a lid and in front of a
   slipper it had never seen.

Results, limitations and figures are in the project report (B.Tech. Project, IIT Bombay, Robot Perception and
Learning Lab, November 2026).

## Repository layout

| Path | What it is |
|---|---|
| `scripts/` | data side: scenario planner, MuJoCo proxy simulator (`proxy_sim.py`), episode format (`episode_io.py`), simulation labellers (`frame_labels*.py`), ROS 2 bag converter (`rosbag_to_episode.py`), real-data labeller and dataset builder (`real_labels.py`, `real_dataset.py`), run-sheet tools |
| `learning/` | models: GONet port and training (`gonet*.py`), simulation baselines, real-data models (`real_frozen.py`, `real_finetune.py`, `real_gonet.py`), metrics, sim-to-real test |
| `hardware/` | scripts that run on the robot: recording (`record_drive.sh`), camera start, robot finder |
| `hardware/gonogo/` | the live go/no-go safety filter; see its [README](hardware/gonogo/README.md) |
| `presentation/` | scripts for result figures and run videos |
| `isaac/` | an Isaac Sim collector with the same episode format (experimental, not used for the results) |
| `scenarios/` | two seeded example scenario manifests |
| `results/` | figures and a label summary from the simulation study |
| `tests/` | unit tests (`python3 -m pytest tests`) |

## Setup

Python 3.12.

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python3 -m pytest tests        # 94 pass; 1 needs the official GONet weights

The large inputs are not in the repository:

- the recorded bags (about 4 GB);
- the converted episodes;
- the model checkpoints;
- the exported ONNX model (223 MB);
- the go/no-go wheel files.

Each command below says what it reads and writes.

## Simulation study (laptop, about 10 minutes)

    python3 -m scripts.proxy_sim --seeds 0-119 --output data/episodes
    python3 -m scripts.frame_labels --episodes data/episodes --output data/labels/labels.csv
    python3 -m learning.train_baselines --labels data/labels/labels.csv --episodes data/episodes --output runs/baselines
    python3 -m learning.figures --episodes data/episodes --labels data/labels/labels.csv --runs runs/baselines --output results/figures

The MuJoCo proxy checks the pipeline. It is not evidence about real robots.

## Real robot

**1. Record** (on the robot, ROS 2 Jazzy). After every boot, start the camera with `bash hardware/start_camera.sh`.
Then record each drive into `~/btp_bags/` with `DURATION=40 bash hardware/record_drive.sh s1_r001`.

**2. Convert and label** (laptop). The bags are read with the `rosbags` library, so ROS is not needed.

    python3 -m scripts.real_dataset --bags data/real_bags --plans data/run_sheets --out data/real/s1
    python3 -m scripts.real_labels --data data/real/s1

**3. Train and evaluate.** The fine-tuned models need a GPU; we used an NVIDIA L40S/A100.

    python3 -m learning.real_prep --data data/real/s1 --out runs/real_v1
    python3 -m learning.real_frozen   --data data/real/s1 --out runs/real_v1
    python3 -m learning.real_finetune --data data/real/s1 --out runs/real_v1 --stage cv
    python3 -m learning.real_gonet    --data data/real/s1 --out runs/real_v1 --official data/gonet/official_weights
    python3 -m learning.sim_to_real --help

**4. Run it live.** Export the ensemble with `python3 hardware/gonogo/export_onnx.py`. Then follow
[hardware/gonogo/README.md](hardware/gonogo/README.md).

## How the code was checked

- **Unit tests** (`tests/`) cover the episode format, scenario planning, both simulation labellers, the bag
  converter, the real-data labeller and splits, the metrics, the run-sheet tools and the GONet port.
- **The GONet port** reproduces the authors' released model on GO Stanford 1 (95.4% against their 94.25%).
- **The labeller** was compared with the recorded object placements: it marked 77 of 78 hard-object crossings as
  not traversable and flagged no clean runs.
- **The go/no-go node** was run on the robot. Its ONNX export matches the PyTorch scores to within 0.02.

## Known limitations

- All real training data comes from one corridor, recorded on one day.
- Live, the model stops for flat high-contrast items it could drive over, such as paper and printed floor mats.
- The safety filter reacts in about 0.9 s, so it stops in time only up to about 1.2 m/s.

## Licence

MIT; see [LICENSE](LICENSE).
