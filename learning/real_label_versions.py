# Checked: reproduces the label-version table of the report (Table 9).
"""Score saved held-out predictions under each way of treating mild-bump frames.

    python3 -m learning.real_label_versions --data DATA --scores runs/real_v1/finetune_scores --out OUT.json

Frames the robot never reached stay out in every version (no label exists for them).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from learning.real_data import load_rows

MODELS = {"trained_grey_left_out": "cv5__mask__resnet18", "trained_bumps_traversable": "mildsafe__mask__resnet18"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--scores", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
rows = load_rows(args.data)
y = np.array([-1 if r["y"] is None else r["y"] for r in rows])
bump = np.array([r["reason"] == "abnormal_but_acceptable" for r in rows])
truths = {"grey_left_out": y, "bumps_traversable": np.where(bump, 1, y), "bumps_not_traversable": np.where(bump, 0, y)}
out = {}
for name, file in MODELS.items():
    saved = np.load(args.scores / f"{file}.npz")
    for truth, t in truths.items():
        k = t >= 0
        flagged = saved["flag"][k] == 1
        keep, caught = float((~flagged)[t[k] == 1].mean()), float(flagged[t[k] == 0].mean())
        out[f"{name}/scored_{truth}"] = {"frames": int(k.sum()), "auroc": round(float(roc_auc_score(t[k], saved["score"][k])), 4),
                                         "balanced_accuracy": round((keep + caught) / 2, 4), "good_floor_kept": round(keep, 4),
                                         "bad_floor_flagged": round(caught, 4)}
args.out.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
