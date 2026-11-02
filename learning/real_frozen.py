# UNVERIFIED: AI-written; checked through its outputs and the split/metric tests.
"""Frozen-feature experiments on real data: ResNet-18 features + logistic regression / Mahalanobis.

    python3 -m learning.real_frozen --data DATA --out runs/real_v1

Thresholds come from validation runs only. No test run touches training or tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from learning.real_data import INPUT_HW, RGB_MEAN, RGB_STD, build_cache, load_rows, run_folds, run_table, train_val_test
from learning.real_metrics import OBJECTS, bootstrap, choose_threshold, summary

FOLDS = 5


def device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def features(images: np.ndarray, cache: Path) -> np.ndarray:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    if cache.exists():
        out = np.load(cache)
        if len(out) == len(images):
            return out
    dev = device()
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval().to(dev)
    out = []
    with torch.no_grad():
        for start in range(0, len(images), 128):
            batch = torch.from_numpy(((images[start:start + 128] / 255.0 - RGB_MEAN) / RGB_STD).astype(np.float32)).permute(0, 3, 1, 2)
            out.append(model(batch.to(dev)).cpu().numpy())
    out = np.concatenate(out)
    np.save(cache, out)
    return out


def fit(x_train, y_train, name):
    from sklearn.covariance import LedoitWolf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
        model.fit(x_train, y_train)
        return lambda x: model.decision_function(x)
    positives = x_train[y_train == 1]
    mean = positives.mean(axis=0)
    precision = LedoitWolf().fit(positives - mean).precision_
    return lambda x: -np.sqrt(np.einsum("ij,jk,ik->i", x - mean, precision, x - mean))


class Experiment:
    def __init__(self, rows, x, y=None):
        self.rows, self.x = rows, x
        self.y = np.array([-1 if r["y"] is None else r["y"] for r in rows]) if y is None else y
        self.run = np.array([r["run"] for r in rows])

    def mask(self, names):
        return np.isin(self.run, list(names))

    def run_split(self, model, train, val, test, score, flag, drop=None):
        keep = np.ones(len(self.rows), bool) if drop is None else ~drop
        tr = self.mask(train) & (self.y >= 0) & keep
        va = self.mask(val) & (self.y >= 0) & keep
        scorer = fit(self.x[tr], self.y[tr], model)
        tau = choose_threshold(scorer(self.x[va]), self.y[va])
        te = self.mask(test)
        s = scorer(self.x[te])
        score[te], flag[te] = s, (s < tau).astype(float)


def blank(n):
    return np.full(n, np.nan), np.full(n, np.nan)


def report(rows, score, flag, ci=True):
    known = ~np.isnan(score)
    out = summary(rows, score, flag)
    if ci:
        out["ci95"] = bootstrap(rows, score, flag)
    out["test_frames_scored"] = int(known.sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data)
    for order, r in enumerate(rows):
        r["frame_order"] = order
    runs = run_table(rows)
    folds = run_folds(runs, FOLDS)
    (args.out / "folds.json").write_text(json.dumps([sorted(f) for f in folds], indent=1) + "\n")
    results = {}
    arrays = {}

    for mode in ("mask", "full"):
        images = build_cache(args.data, rows, mode, args.out / f"images_{mode}.npy")
        x = features(images, args.out / f"features_{mode}.npy")
        print(mode, x.shape, flush=True)
        exp = Experiment(rows, x)
        roi_key = "in_roi" if mode == "mask" else "visible"
        for model in ("logreg", "mahalanobis"):
            # (1) Five-fold over whole runs.
            score, flag = blank(len(rows))
            for i in range(FOLDS):
                exp.run_split(model, *train_val_test(runs, folds, i), score, flag)
            key = f"cv5/{mode}/{model}"
            results[key] = report(rows, score, flag)
            arrays[key.replace("/", "__")] = score
            arrays[key.replace("/", "__") + "__flag"] = flag
            print(key, {k: round(results[key][k], 3) for k in ("auroc", "balanced_accuracy", "clean_frame_flag_rate", "false_stops_per_100m", "bad_crossing_detected", "mild_crossing_flagged")}, flush=True)

            # (2) Drive one way, test the other: the background is seen from the other side.
            for name, value in (("A->B", "A->B"), ("B->A", "B->A")):
                train_all = sorted(k for k, v in runs.items() if v["direction"] == value)
                val = set(train_all[::5]); train = set(train_all) - val
                test = {k for k, v in runs.items() if v["direction"] != value}
                score, flag = blank(len(rows))
                exp.run_split(model, train, val, test, score, flag)
                results[f"direction/{mode}/{model}/train_{name}"] = report(rows, score, flag)

            # (3) One session trains, the other tests: objects sit in different spots.
            for session in ("s1a", "s1c"):
                train_all = sorted(k for k, v in runs.items() if v["session"] == session)
                val = set(train_all[::5]); train = set(train_all) - val
                test = {k for k, v in runs.items() if v["session"] != session}
                score, flag = blank(len(rows))
                exp.run_split(model, train, val, test, score, flag)
                results[f"session/{mode}/{model}/train_{session}"] = report(rows, score, flag)

            # (4) Leave one object out: no photo in which it could be seen is used to train or to tune.
            for obj in OBJECTS:
                drop = np.array([obj in r[roi_key] for r in rows])
                score, flag = blank(len(rows))
                for i in range(FOLDS):
                    exp.run_split(model, *train_val_test(runs, folds, i), score, flag, drop=drop)
                results[f"looo/{mode}/{model}/{obj}"] = report(rows, score, flag, ci=False)
            print(f"done looo {mode} {model}", flush=True)
            (args.out / "frozen_results.json").write_text(json.dumps(results, indent=1) + "\n")

    # (5) Learning curve and label sensitivity on the masked input, logistic regression.
    x = features(np.load(args.out / "images_mask.npy", mmap_mode="r"), args.out / "features_mask.npy")
    exp = Experiment(rows, x)
    rng = np.random.default_rng(0)
    for n in (2, 4, 8, 12, 16, 20):
        for rep in range(5):
            score, flag = blank(len(rows))
            for i in range(FOLDS):
                train, val, test = train_val_test(runs, folds, i)
                hazard = sorted(k for k in train if runs[k]["condition"] == "hazard")
                clean = [k for k in train if runs[k]["condition"] == "clean"]
                pick = set(rng.choice(hazard, min(n, len(hazard)), replace=False).tolist()) | set(clean[:1])
                exp.run_split("logreg", pick, val, test, score, flag)
            results[f"curve/mask/logreg/runs_{n}/rep_{rep}"] = report(rows, score, flag, ci=False)
        print("curve", n, flush=True)
    for multiple in ("1.5", "3.0"):
        path = args.data / f"labels_x{multiple}.csv"
        if path.exists():
            with path.open(newline="") as file:
                alt = {(r["run"], r["frame_idx"]): r["y"] for r in csv.DictReader(file)}
            y = np.array([-1 if alt[(r["run"], r["frame_idx"])] == "" else int(alt[(r["run"], r["frame_idx"])]) for r in rows])
            alt_rows = [{**r, "y": None if v < 0 else int(v)} for r, v in zip(rows, y)]
            alt_exp = Experiment(alt_rows, x, y)
            score, flag = blank(len(rows))
            for i in range(FOLDS):
                alt_exp.run_split("logreg", *train_val_test(runs, folds, i), score, flag)
            results[f"labels_x{multiple}/mask/logreg"] = report(alt_rows, score, flag, ci=False)
            print("labels", multiple, {k: round(results[f'labels_x{multiple}/mask/logreg'][k], 3) for k in ("auroc", "balanced_accuracy")}, flush=True)

    (args.out / "frozen_results.json").write_text(json.dumps(results, indent=1) + "\n")
    np.savez_compressed(args.out / "frozen_scores.npz", **arrays)
    print("wrote", args.out / "frozen_results.json")


if __name__ == "__main__":
    main()
