# Checked on recorded bags (laptop). Offline check of the go/no-go pipeline on a recorded bag (Mac or robot).
"""Feed a recorded drive through the live go/no-go logic and print when it would have stopped.

    python3 hardware/gonogo/replay_bag.py data/real_bags/s1c/s1c_r016_*          # shadow rule, timeline
    python3 hardware/gonogo/replay_bag.py --latch data/real_bags/s1c/s1c_r016_*  # gate rule: first stop only

Runs from s1 were used to train the ensemble, so on them this checks the pipeline (decoding, mask, timing, stop
rule), not how well the model generalises. Needs the ``rosbags`` package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gonogo_core import Model, StopRule  # noqa: E402

IMAGE_SUFFIX = "color/image_raw/compressed"
ODOM_SUFFIX = "platform/odom"


def stamp(msg, fallback_ns: int) -> float:
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9 if s.sec else fallback_ns * 1e-9


def read(bag: Path):
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    images, odom = [], []
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        conns = [c for c in reader.connections if c.topic.endswith(IMAGE_SUFFIX) or c.topic.endswith(ODOM_SUFFIX)]
        for c, t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, c.msgtype)
            if c.topic.endswith(IMAGE_SUFFIX):
                images.append((stamp(msg, t), bytes(msg.data)))
            else:
                p = msg.pose.pose.position
                odom.append((stamp(msg, t), p.x, p.y))
    images.sort(key=lambda m: m[0])
    odom = np.array(sorted(odom))
    return images, odom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+", type=Path)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=-1.0)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--latch", action="store_true", help="gate rule: stop once and stay stopped")
    args = ap.parse_args()
    model = Model()
    for bag in args.bags:
        images, odom = read(bag)
        if not images:
            print(f"{bag.name}: no compressed colour images")
            continue
        t0 = images[0][0]
        travelled = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(odom[:, 1]), np.diff(odom[:, 2])))])
        rule, state, stops, times, next_t = StopRule(args.margin, args.consecutive, args.latch), "GO", 0, [], t0
        for t, jpeg in images:
            if t + 1e-6 < next_t:
                continue
            next_t = t + 1.0 / args.rate
            _, x = model.preprocess(jpeg)
            z, _, ms = model.score(x)
            times.append(ms)
            rule.update(z)
            new = "STOP" if rule.stopped else "GO"
            if new != state:
                d = float(np.interp(t, odom[:, 0], travelled))
                print(f"  {bag.name}: {t - t0:5.1f} s  {d:5.2f} m driven  {state} -> {new}  (z {z:+.2f})")
                stops += new == "STOP"
                state = new
                if args.latch and new == "STOP":
                    break
        print(f"{bag.name}: {stops} stop(s); model {np.median(times):.0f} ms per frame (median)")


if __name__ == "__main__":
    main()
