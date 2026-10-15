"""GONet's own automatic labelling rule applied to our drives.

GONet (Hirose et al. 2018) calls a photo positive (safe) when the robot then
moved at >= 0.3 m/s continuously for 2.4 s. We apply the same rule to each
photo and call it negative when the robot did not (it slowed or stopped within
2.4 s). The start ramp (robot not yet up to speed) is left unlabelled.

Train and validation rows get the rule's label; TEST rows keep our physics
labels, so the model trained on GONet's labels is scored on the same test as
everything else:

    python3 -m learning.gonet_rule_labels --labels data/labels_v2/labels.csv \\
        --episodes data/episodes_v2 --output data/labels_gonet_rule/labels.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from learning.train_baselines import split_of
from scripts.route_follower import RAMP_S

MIN_SPEED_MPS = 0.3
HORIZON_S = 2.4


def rule_label(t: np.ndarray, v: np.ndarray, t_frame: float):
    if t_frame < RAMP_S:
        return None, "gonet_rule_ramp"
    window = (t >= t_frame) & (t <= t_frame + HORIZON_S)
    if t[-1] < t_frame + HORIZON_S:
        return None, "gonet_rule_too_close_to_end"
    return (1, "gonet_rule_moving") if (v[window] >= MIN_SPEED_MPS).all() else (0, "gonet_rule_slowed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.labels.open(newline="") as f:
        rows = list(csv.DictReader(f))
    speeds = {}
    stats = collections.defaultdict(collections.Counter)
    for row in rows:
        if split_of(int(row["seed"])) in ("train", "val"):
            ep = row["episode_id"]
            if ep not in speeds:
                with (args.episodes / ep / "state.csv").open(newline="") as f:
                    state = list(csv.DictReader(f))
                speeds[ep] = (np.array([float(s["t_s"]) for s in state]), np.array([float(s["v_fwd_mps"]) for s in state]))
            physics_y = row["y"]
            y, reason = rule_label(*speeds[ep], float(row["t_s"]))
            stats[row["patch_class"]][f"physics={physics_y or 'gray'} rule={'gray' if y is None else y}"] += 1
            row["y"], row["reason"] = ("" if y is None else y), reason
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {k: dict(sorted(v.items())) for k, v in sorted(stats.items())}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
