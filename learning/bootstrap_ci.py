"""Confidence intervals by resampling test SCENES (not photos).

Photos from one scene are strongly correlated, so the honest unit of
resampling is the scene: draw the 20 test scenes with replacement, recompute
every metric, repeat. Reports 95 % percentile intervals, and paired
differences against a reference model (same resampled scenes for both).

    python3 -m learning.bootstrap_ci --baselines runs/baselines_v2 \\
        --gonet gonet_retrained=runs/gonet_ours/eval_retrained --gonet-manifest data/gonet_ours/manifest.csv \\
        --reference finetuned_resnet18 --output runs/ci/ci.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CLASSES = ("cable_hose", "oil_spill", "small_lip", "plastic_wrapper", "clean_floor")


def load(baselines: Path, gonets: list[str], manifest_path: Path):
    with (baselines / "predictions_test.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    metrics = json.loads((baselines / "metrics.json").read_text())
    thresholds = {r["model"]: r["threshold"] for r in metrics["results"]}
    resnet_models = [c for c in rows[0] if c in thresholds]
    scores = {m: np.array([float(r[m]) for r in rows]) for m in resnet_models}
    index = {r["rgb_path"]: i for i, r in enumerate(rows)}
    if gonets:
        with manifest_path.open(newline="") as f:
            test = [m for m in csv.DictReader(f) if m["split"] == "test"]
        order = np.array([index[f"{m['episode_id']}/rgb/{int(float(m['frame_idx'])):06d}.jpg"] for m in test])
        for spec in gonets:
            name, stem = spec.split("=", 1)
            stem = Path(stem)
            s = np.full(len(rows), np.nan)
            s[order] = np.load(stem.with_suffix(".scores.npz"))["test"]
            scores[name] = s
            thresholds[name] = json.loads(stem.with_suffix(".json").read_text())["tuned_on_val"]["threshold"]
    y = np.array([np.nan if r["y"] == "" else float(r["y"]) for r in rows])
    seed = np.array([int(r["rgb_path"].split("-")[1]) for r in rows])
    cls = np.array([r["patch_class"] for r in rows])
    return scores, thresholds, y, seed, cls


def metrics(s, tau, y, cls, idx) -> dict:
    s, y, cls = s[idx], y[idx], cls[idx]
    lab = ~np.isnan(y)
    out = {"auroc": roc_auc_score(y[lab], s[lab]), "ap_non_traversable": average_precision_score(1 - y[lab], -s[lab])}
    for c in CLASSES:
        mask = cls == c
        out[f"flag_{c}"] = float((s[mask] < tau).mean()) if mask.any() else np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--gonet", action="append", default=[], metavar="NAME=EVAL_STEM")
    parser.add_argument("--gonet-manifest", type=Path)
    parser.add_argument("--reference", default="finetuned_resnet18")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scores, thresholds, y, seed, cls = load(args.baselines, args.gonet, args.gonet_manifest)
    scenes = np.unique(seed)
    by_scene = {sc: np.flatnonzero(seed == sc) for sc in scenes}
    rng = np.random.default_rng(0)
    draws = {m: [] for m in scores}
    for _ in range(args.reps):
        idx = np.concatenate([by_scene[sc] for sc in rng.choice(scenes, len(scenes), replace=True)])
        lab = idx[~np.isnan(y[idx])]
        if len(np.unique(y[lab])) < 2:
            continue
        for m in scores:
            draws[m].append(metrics(scores[m], thresholds[m], y, cls, idx))
    everything = np.arange(len(y))
    result = {"scenes": len(scenes), "reps": len(draws[args.reference]), "models": {}, f"difference_vs_{args.reference}": {}}
    for m in scores:
        point = metrics(scores[m], thresholds[m], y, cls, everything)
        result["models"][m] = {k: {"value": round(v, 4),
                                   "ci95": [round(float(np.nanpercentile([d[k] for d in draws[m]], q)), 4) for q in (2.5, 97.5)]}
                               for k, v in point.items()}
        if m != args.reference:
            diffs = {k: np.array([r[k] - d[k] for r, d in zip(draws[args.reference], draws[m])]) for k in point}
            result[f"difference_vs_{args.reference}"][m] = {
                k: {"ci95": [round(float(np.nanpercentile(v, q)), 4) for q in (2.5, 97.5)],
                    "reference_better_share": round(float(np.nanmean(v > 0)), 3)} for k, v in diffs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for m, vals in result["models"].items():
        print(f"{m:45s} AUROC {vals['auroc']['value']:.3f} {vals['auroc']['ci95']}  "
              f"lip {vals['flag_small_lip']['value']:.2f} {vals['flag_small_lip']['ci95']}  "
              f"wrapper {vals['flag_plastic_wrapper']['value']:.2f} {vals['flag_plastic_wrapper']['ci95']}")
    print(json.dumps(result[f"difference_vs_{args.reference}"], indent=1)[:3000])


if __name__ == "__main__":
    main()
