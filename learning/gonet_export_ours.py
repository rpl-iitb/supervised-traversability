"""Export our warehouse dataset in GO Stanford's folder layout, so GONet trains on it unchanged.

    python3 -m learning.gonet_export_ours --labels data/labels_v2/labels.csv \\
        --episodes data/episodes_v2 --out data/gonet_ours

Writes (images resized to GONet's 128x128, same splits as train_baselines):

    images/<split>/<episode>__<frame>.jpg        every train/val/test photo (gray ones too)
    data_train/positive_L/                       GAN + InvGen positives: our traversable train photos
    data_{train,vali,test}_annotation/{positive,negative}_L/
                                                 head data: traversable / not traversable photos.
                                                 train and vali are class-balanced like GONet's
                                                 hand-labelled sets; test keeps every labelled photo.
    manifest.csv                                 image path + label row, for gonet_eval_ours

Annotation folders hold relative symlinks into images/ (rsync -a keeps them).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

from PIL import Image

from learning.train_baselines import load_rows

SPLIT_DIR = {"train": "train", "val": "vali", "test": "test"}


def export(labels: Path, episodes: Path, out: Path, seed: int = 0) -> dict:
    rows = [r for r in load_rows(labels) if r["split"] in SPLIT_DIR]
    manifest = []
    for row in rows:
        name = f"{row['episode_id']}__{int(float(row['frame_idx'])):06d}.jpg"
        target = out / "images" / row["split"] / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.open(episodes / row["rgb_path"]).convert("RGB").resize((128, 128)).save(target, quality=95)
        manifest.append({**row, "image": str(target.relative_to(out))})

    def link(items, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        for item in items:
            dst = folder / Path(item["image"]).name
            if not dst.exists():
                dst.symlink_to(os.path.relpath(out / item["image"], folder))

    rng = random.Random(seed)
    counts = {}
    for split, folder in SPLIT_DIR.items():
        pos = [m for m in manifest if m["split"] == split and m["y"] == 1]
        neg = [m for m in manifest if m["split"] == split and m["y"] == 0]
        if split != "test":
            pos = rng.sample(pos, min(len(pos), len(neg)))
        link(pos, out / f"data_{folder}_annotation" / "positive_L")
        link(neg, out / f"data_{folder}_annotation" / "negative_L")
        counts[split] = {"positive": len(pos), "negative": len(neg)}
    gan = [m for m in manifest if m["split"] == "train" and m["y"] == 1]
    link(gan, out / "data_train" / "positive_L")
    counts["gan_positives"] = len(gan)

    columns = ["image", "split", "episode_id", "seed", "condition", "frame_idx", "patch_class", "y", "reason"]
    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**m, "y": "" if m["y"] is None else m["y"]} for m in manifest)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.labels, args.episodes, args.out))


if __name__ == "__main__":
    main()
