"""Camera-only traversability baselines on labelled episodes.

Three models, all scoring a single RGB frame (the company-style interface
``p(I_t) in [0, 1]``, higher = traversable):

1. ``frozen_logreg`` -- ImageNet ResNet-18 features + logistic regression.
2. ``finetuned_resnet18`` -- ResNet-18 fine-tuned end to end.
3. ``positive_only_mahalanobis`` -- a Gaussian fitted to features of frames
   labelled traversable *only*; the score is the negative Mahalanobis
   distance. It never sees a negative example, mirroring GONet's
   positive-only training and the positive-unlabelled nature of real
   deployment data (the safety layer prevents most failures from happening).

Splits are by scenario seed so no scene appears in two splits. The input is
the look-ahead region of interest (ROI): the image area where the labelled
floor corridor can appear, computed from the camera model.

Usage::

    python3 -m learning.train_baselines --labels data/labels/labels.csv \
        --episodes data/episodes --output runs/baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.episode_io import project_base_point
from scripts.frame_labels import LabelConfig


TRAIN_SEEDS = range(0, 80)
VAL_SEEDS = range(80, 100)
TEST_SEEDS = range(100, 120)
# Rows that say nothing about the visible patch are excluded from per-class
# flag rates (they are not "hard" examples, just unobservable ones).
UNOBSERVABLE_REASONS = {
    "patch_not_covered",
    "window_contaminated_by_other_hazard",
    "hazard_not_visible",
    "stalled_on_other_hazard",
    "no_command_in_window",
    "beyond_route",   # version 2 labeller
    "not_reached",
    "not_moving",     # real robot: parked or ramping photos
}
HAZARD_MARGIN_M = 0.45


def split_of(seed: int) -> str:
    if seed in TRAIN_SEEDS:
        return "train"
    if seed in VAL_SEEDS:
        return "val"
    if seed in TEST_SEEDS:
        return "test"
    return "unused"


def roi_box(camera: dict, config: LabelConfig) -> tuple[int, int, int, int]:
    """Pixel box covering the labelled look-ahead corridor plus a hazard margin."""
    points = []
    for forward in (config.lookahead_min_m - HAZARD_MARGIN_M, config.lookahead_max_m + HAZARD_MARGIN_M):
        for left in (-(config.corridor_half_width_m + HAZARD_MARGIN_M), config.corridor_half_width_m + HAZARD_MARGIN_M):
            for up in (0.0, 0.1):
                pixel = project_base_point(camera, max(forward, camera["mount_xyz_m"][0] + 0.05), left, up)
                if pixel is not None:
                    points.append(pixel)
    us, vs = [p[0] for p in points], [p[1] for p in points]
    return (
        int(max(0, min(us))),
        int(max(0, min(vs))),
        int(min(camera["width"], max(us))),
        int(min(camera["height"], max(vs))),
    )


def load_rows(labels_path: Path) -> list[dict]:
    with labels_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["split"] = split_of(row["seed"])
        row["y"] = None if row["y"] == "" else int(row["y"])
        row["observable"] = row["reason"] not in UNOBSERVABLE_REASONS
    return [row for row in rows if row["observable"]]


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _image_tensor(path: Path, box, train: bool = False):
    import torch
    from PIL import Image
    from torchvision.transforms import v2

    image = Image.open(path).convert("RGB").crop(box)
    ops = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((160, 288), antialias=True)]
    if train:
        ops.insert(0, v2.ColorJitter(0.3, 0.3, 0.2, 0.03))
    ops.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return v2.Compose(ops)(image)


def extract_features(rows: list[dict], episodes: Path, box, cache: Path) -> np.ndarray:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    if cache.exists():
        data = np.load(cache)
        if list(data["paths"]) == [row["rgb_path"] for row in rows]:
            return data["features"]
    device = _device()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            batch = torch.stack([_image_tensor(episodes / r["rgb_path"], box) for r in rows[start:start + 128]])
            features.append(model(batch.to(device)).cpu().numpy())
    features = np.concatenate(features).astype(np.float32)
    np.savez_compressed(cache, features=features, paths=np.array([row["rgb_path"] for row in rows]))
    return features


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising balanced accuracy on the validation split."""
    best, best_tau = -1.0, 0.5
    for tau in np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197))):
        predicted = scores >= tau
        tpr = predicted[labels == 1].mean()
        tnr = (~predicted[labels == 0]).mean()
        if (tpr + tnr) / 2 > best:
            best, best_tau = (tpr + tnr) / 2, float(tau)
    return best_tau


def evaluate(name: str, scores: dict, rows_by_split: dict, tau: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    test_rows, test_scores = rows_by_split["test"], scores["test"]
    labelled = np.array([row["y"] is not None for row in test_rows])
    y = np.array([row["y"] for row in test_rows if row["y"] is not None])
    s = test_scores[labelled]
    predicted = s >= tau
    result = {
        "model": name,
        "threshold": round(float(tau), 4),
        "test_frames_labelled": int(len(y)),
        "auroc": round(float(roc_auc_score(y, s)), 4),
        # AP for detecting the NON-traversable class, the one that matters for safety.
        "ap_non_traversable": round(float(average_precision_score(1 - y, -s)), 4),
        "balanced_accuracy": round(float((predicted[y == 1].mean() + (~predicted[y == 0]).mean()) / 2), 4),
        "tpr_traversable": round(float(predicted[y == 1].mean()), 4),
        "tnr_non_traversable": round(float((~predicted[y == 0]).mean()), 4),
        "flag_rate_by_class": {},
    }
    for patch_class in sorted({row["patch_class"] for row in test_rows}):
        mask = np.array([row["patch_class"] == patch_class for row in test_rows])
        result["flag_rate_by_class"][patch_class] = {
            "frames": int(mask.sum()),
            "flagged_non_traversable": round(float((test_scores[mask] < tau).mean()), 4),
        }
    return result


def fit_frozen_models(features: dict, rows_by_split: dict) -> dict:
    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def labelled(split):
        mask = np.array([row["y"] is not None for row in rows_by_split[split]])
        return features[split][mask], np.array([row["y"] for row in rows_by_split[split] if row["y"] is not None])

    x_train, y_train = labelled("train")
    logreg = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
    logreg.fit(x_train, y_train)

    positives = x_train[y_train == 1]
    mean = positives.mean(axis=0)
    precision = LedoitWolf().fit(positives - mean).precision_

    def mahalanobis_score(x):
        centred = x - mean
        return -np.sqrt(np.einsum("ij,jk,ik->i", centred, precision, centred))

    return {
        "frozen_logreg": lambda x: logreg.predict_proba(x)[:, 1],
        "positive_only_mahalanobis": mahalanobis_score,
    }


def finetune_resnet(rows_by_split: dict, episodes: Path, box, epochs: int, output: Path):
    """Fine-tune ResNet-18; return val/test scores and a scorer for other test sets."""
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision.models import ResNet18_Weights, resnet18

    class Frames(Dataset):
        def __init__(self, rows, train, root=episodes):
            self.rows, self.train, self.root = rows, train, root

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            return _image_tensor(self.root / row["rgb_path"], box, self.train), float(row["y"] if row["y"] is not None else -1)

    device = _device()
    train_rows = [row for row in rows_by_split["train"] if row["y"] is not None]
    labels = np.array([row["y"] for row in train_rows])
    weights = np.where(labels == 1, 0.5 / (labels == 1).sum(), 0.5 / (labels == 0).sum())
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=min(len(train_rows), 12000), replacement=True)
    workers = int(os.environ.get("TRAV_WORKERS", "0"))  # >0 on Linux clusters; 0 is safest on macOS
    loader = DataLoader(Frames(train_rows, True), batch_size=64, sampler=sampler, num_workers=workers)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(512, 1)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        total, started = 0.0, time.time()
        for images, targets in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(images.to(device)).squeeze(1), targets.float().to(device))
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(targets)
        print(f"  finetune epoch {epoch + 1}/{epochs}: loss {total / len(sampler):.4f} ({time.time() - started:.0f}s)", flush=True)
    torch.save(model.state_dict(), output / "finetuned_resnet18.pt")

    model.eval()

    def score(rows, root=episodes):
        with torch.no_grad():
            loader = DataLoader(Frames(rows, False, root), batch_size=128, num_workers=workers)
            return np.concatenate([torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy() for x, _ in loader])

    return {split: score(rows_by_split[split]) for split in ("val", "test")}, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--extra-test", action="append", default=[], metavar="NAME=LABELS_CSV:EPISODES_DIR",
                        help="Extra test set scored with the already-trained models and val thresholds")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.labels)
    first_episode = args.episodes / rows[0]["rgb_path"].split("/")[0]
    camera = json.loads((first_episode / "meta.json").read_text())["camera"]
    box = roi_box(camera, LabelConfig())
    rows_by_split = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    print("ROI box", box, {k: len(v) for k, v in rows_by_split.items()}, flush=True)

    all_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    started = time.time()
    all_features = extract_features(all_rows, args.episodes, box, args.output / "features_roi.npz")
    print(f"features {all_features.shape} in {time.time() - started:.0f}s", flush=True)
    offsets = np.cumsum([0] + [len(rows_by_split[s]) for s in ("train", "val", "test")])
    features = {s: all_features[offsets[i]:offsets[i + 1]] for i, s in enumerate(("train", "val", "test"))}

    results, predictions, thresholds, scorers = [], {}, {}, {}

    def run(name, scores):
        val_labelled = np.array([row["y"] is not None for row in rows_by_split["val"]])
        tau = choose_threshold(scores["val"][val_labelled], np.array([row["y"] for row in rows_by_split["val"] if row["y"] is not None]))
        thresholds[name] = tau
        results.append(evaluate(name, scores, rows_by_split, tau))
        predictions[name] = scores["test"]
        print(json.dumps({k: results[-1][k] for k in ("model", "auroc", "ap_non_traversable", "balanced_accuracy")}), flush=True)

    for name, score_fn in fit_frozen_models(features, rows_by_split).items():
        scorers[name] = lambda rows, root, fn=score_fn: fn(extract_features(rows, root, box, args.output / f"features_roi_{root.name}.npz"))
        run(name, {split: score_fn(features[split]) for split in ("val", "test")})

    # Unseen-hazard experiment: remove every cable frame from training, then
    # ask whether each model still flags cables at test time.
    held_out = "cable_hose"
    keep = np.array([row["patch_class"] != held_out for row in rows_by_split["train"]])
    reduced_rows = {**rows_by_split, "train": [row for row, k in zip(rows_by_split["train"], keep) if k]}
    for name, score_fn in fit_frozen_models({**features, "train": features["train"][keep]}, reduced_rows).items():
        scores = {split: score_fn(features[split]) for split in ("val", "test")}
        val_mask = np.array([row["y"] is not None and row["patch_class"] != held_out for row in rows_by_split["val"]])
        tau = choose_threshold(scores["val"][val_mask], np.array([row["y"] for row, m in zip(rows_by_split["val"], val_mask) if m]))
        result = evaluate(f"{name}__no_{held_out}_in_training", scores, rows_by_split, tau)
        result["held_out_class"] = held_out
        results.append(result)
        print(json.dumps({"model": result["model"], "cable_flag_rate": result["flag_rate_by_class"][held_out]}), flush=True)

    if not args.skip_finetune:
        finetuned_scores, scorers["finetuned_resnet18"] = finetune_resnet(rows_by_split, args.episodes, box, args.finetune_epochs, args.output)
        run("finetuned_resnet18", finetuned_scores)

    # Extra test sets (e.g. a held-out appearance regime): no refitting, the
    # thresholds chosen on the ordinary validation split are reused.
    for spec in args.extra_test:
        set_name, paths = spec.split("=", 1)
        labels_path, episodes_dir = (Path(v) for v in paths.split(":", 1))
        extra_rows = load_rows(labels_path)
        for name, scorer in scorers.items():
            result = evaluate(f"{name}__{set_name}", {"test": scorer(extra_rows, episodes_dir)}, {"test": extra_rows}, thresholds[name])
            result["test_set"] = set_name
            results.append(result)
            print(json.dumps({k: result[k] for k in ("model", "auroc", "ap_non_traversable", "balanced_accuracy")}), flush=True)

    (args.output / "metrics.json").write_text(json.dumps({"roi_box": box, "splits": {
        "train_seeds": [TRAIN_SEEDS.start, TRAIN_SEEDS.stop - 1], "val_seeds": [VAL_SEEDS.start, VAL_SEEDS.stop - 1],
        "test_seeds": [TEST_SEEDS.start, TEST_SEEDS.stop - 1]}, "results": results}, indent=2) + "\n")
    with (args.output / "predictions_test.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rgb_path", "patch_class", "y", "reason", *predictions])
        for index, row in enumerate(rows_by_split["test"]):
            writer.writerow([row["rgb_path"], row["patch_class"], "" if row["y"] is None else row["y"], row["reason"],
                             *(round(float(predictions[name][index]), 5) for name in predictions)])
    print(f"wrote {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
