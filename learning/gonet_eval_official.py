"""Check 1: does our PyTorch GONet reproduce the authors' trained model?

Loads the official Chainer weights into ``learning.gonet.GONet`` and scores
the hand-labelled GO Stanford 1 test set (400 positive + 400 negative per
camera). The paper reports 94.25 % accuracy for GONet on GS1.

    python3 -m learning.gonet_eval_official \
        --weights data/gonet/official_weights \
        --annotation data/gonet/gs1/data_test_annotation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from learning.gonet import GONet, load_official_weights, normalise


def load_folder(folder: Path) -> torch.Tensor:
    paths = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    images = np.stack([np.asarray(Image.open(p).convert("RGB").resize((128, 128))) for p in paths])
    return torch.from_numpy(images).permute(0, 3, 1, 2)


@torch.no_grad()
def score(model: GONet, images: torch.Tensor, batch: int = 100) -> np.ndarray:
    return np.concatenate([model(normalise(images[i:i + batch])).numpy() for i in range(0, len(images), batch)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True, help="data_test_annotation directory")
    args = parser.parse_args()

    model = load_official_weights(GONet(), args.weights).eval()
    results = {}
    for camera in ("L", "R"):
        positive = score(model, load_folder(args.annotation / f"positive_{camera}"))
        negative = score(model, load_folder(args.annotation / f"negative_{camera}"))
        correct = (positive >= 0.5).sum() + (negative < 0.5).sum()
        results[camera] = {
            "accuracy": round(float(correct / (len(positive) + len(negative))), 4),
            "positives_correct": round(float((positive >= 0.5).mean()), 4),
            "negatives_correct": round(float((negative < 0.5).mean()), 4),
            "n": int(len(positive) + len(negative)),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
