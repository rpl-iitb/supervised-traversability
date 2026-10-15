"""Score GONet on our warehouse test set with exactly the baselines' metrics.

    # A. GONet as released (authors' weights, no retraining)
    python3 -m learning.gonet_eval_ours --export data/gonet_ours --official data/gonet/official_weights \\
        --name gonet_official --output runs/gonet_ours/official.json
    # B. GONet retrained on our data (stage-3 checkpoint from learning.gonet_train)
    python3 -m learning.gonet_eval_ours --export data/gonet_ours --checkpoint runs/gonet_ours/train \\
        --name gonet_retrained --output runs/gonet_ours/retrained.json

Score = GONet's P(traversable). As in train_baselines, the threshold is chosen
on validation (max balanced accuracy); the result at GONet's own fixed 0.5
threshold is reported too.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from learning.gonet import GONet, load_official_weights, normalise
from learning.train_baselines import choose_threshold, evaluate


def load_manifest(export: Path) -> dict:
    with (export / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["y"] = None if row["y"] == "" else int(row["y"])
    return {split: [r for r in rows if r["split"] == split] for split in ("val", "test")}


@torch.no_grad()
def score(model: GONet, export: Path, rows: list[dict], dev, batch: int = 256) -> np.ndarray:
    out = []
    for start in range(0, len(rows), batch):
        images = np.stack([np.asarray(Image.open(export / r["image"]).convert("RGB")) for r in rows[start:start + batch]])
        x = normalise(torch.from_numpy(images).permute(0, 3, 1, 2)).to(dev)
        out.append(model(x).float().cpu().numpy().ravel())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--official", type=Path, help="directory with the authors' Chainer weights")
    group.add_argument("--checkpoint", type=Path, help="gonet_train output dir holding stage3_gonet.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.official:
        model = load_official_weights(GONet(), args.official)
    else:
        model = GONet()
        model.load_state_dict(torch.load(args.checkpoint / "stage3_gonet.pt", map_location="cpu"))
    model = model.to(dev).eval()

    rows = load_manifest(args.export)
    scores = {split: score(model, args.export, rows[split], dev) for split in rows}
    val_labelled = np.array([r["y"] is not None for r in rows["val"]])
    tau = choose_threshold(scores["val"][val_labelled], np.array([r["y"] for r in rows["val"] if r["y"] is not None]))
    result = {"tuned_on_val": evaluate(args.name, scores, rows, tau),
              "fixed_0.5": evaluate(args.name + "@0.5", scores, rows, 0.5)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output.with_suffix(".scores.npz"), **{k: v for k, v in scores.items()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
