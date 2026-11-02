# UNVERIFIED: AI-written; checked through its outputs.
"""How far apart are the simulator and the real robot? Measure it, in both directions.

    python3 -m learning.sim_to_real --sim-labels data/labels_v2/labels.csv --sim-episodes data/episodes_v2 \\
        --sim-checkpoint runs/baselines_v2/finetuned_resnet18.pt --data DATA --out runs/real_v1

Sim models use the threshold chosen on simulator validation, as a sim-trained model deployed on the robot
would have to; AUROC does not depend on that threshold. Label rules differ between the two (see the report notes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from learning.real_data import RGB_MEAN, RGB_STD, load_rows, run_folds, run_table, train_val_test
from learning.real_frozen import FOLDS, fit
from learning.real_metrics import bootstrap, choose_threshold, summary
from learning.train_baselines import extract_features, load_rows as load_sim_rows


def frechet(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import linalg

    ma, mb, ca, cb = a.mean(0), b.mean(0), np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = linalg.sqrtm(ca @ cb)
    return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * np.real(root)))


def domain_auroc(sim_x, sim_groups, real_x, real_groups, sample=6000, seed=0) -> float:
    """Train a classifier to tell sim from real, on separate scenes/runs. 0.5 = indistinguishable, 1.0 = different worlds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    si = rng.choice(len(sim_x), min(sample, len(sim_x)), replace=False)
    ri = rng.choice(len(real_x), min(sample, len(real_x)), replace=False)
    x = np.concatenate([sim_x[si], real_x[ri]])
    y = np.concatenate([np.zeros(len(si)), np.ones(len(ri))])
    groups = np.concatenate([[f"s{g}" for g in sim_groups[si]], [f"r{g}" for g in real_groups[ri]]])
    out = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        model.fit(x[tr], y[tr])
        out[te] = model.decision_function(x[te])
    return float(roc_auc_score(y, out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-labels", type=Path, required=True)
    parser.add_argument("--sim-episodes", type=Path, required=True, help="sim frames; features are recomputed and cached")
    parser.add_argument("--sim-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from sklearn.metrics import roc_auc_score

    sim_rows = load_sim_rows(args.sim_labels)
    by_split = {s: [r for r in sim_rows if r["split"] == s] for s in ("train", "val", "test")}
    ordered = by_split["train"] + by_split["val"] + by_split["test"]
    box = tuple(json.loads((args.sim_checkpoint.parent / "metrics.json").read_text())["roi_box"])
    sim_x = extract_features(ordered, args.sim_episodes, box, args.out / "sim_features_roi.npz")
    offsets = np.cumsum([0, len(by_split["train"]), len(by_split["val"]), len(by_split["test"])])
    part = {s: sim_x[offsets[i]:offsets[i + 1]] for i, s in enumerate(("train", "val", "test"))}
    label = {s: np.array([-1 if r["y"] is None else r["y"] for r in by_split[s]]) for s in part}

    rows = load_rows(args.data)
    for order, r in enumerate(rows):
        r["frame_order"] = order
    real_x = np.load(args.out / "features_full.npy")
    y = np.array([-1 if r["y"] is None else r["y"] for r in rows])
    names = np.array([r["run"] for r in rows])
    runs = run_table(rows)
    folds = run_folds(runs, FOLDS)
    results = {}

    # 1. Same-world numbers, the reference points.
    def auroc(scores, truth):
        keep = truth >= 0
        return float(roc_auc_score(truth[keep], scores[keep]))

    def judge(key, score, tau):
        flag = (score < tau).astype(float)
        results[key] = {**summary(rows, score, flag), "ci95": bootstrap(rows, score, flag), "threshold_from": "simulator validation"}
        print(key, round(results[key]["auroc"], 3), round(results[key]["balanced_accuracy"], 3), flush=True)

    tr = label["train"] >= 0
    for model in ("logreg", "mahalanobis"):
        scorer = fit(part["train"][tr], label["train"][tr], model)
        va = label["val"] >= 0
        tau = choose_threshold(scorer(part["val"][va]), label["val"][va])
        results[f"sim_on_sim/{model}"] = {"auroc": auroc(scorer(part["test"]), label["test"])}
        judge(f"sim_on_real/{model}", scorer(real_x), tau)

    # 2. The sim fine-tuned network, on real photos.
    from torchvision.models import resnet18

    net = resnet18(weights=None)
    net.fc = torch.nn.Linear(512, 1)
    net.load_state_dict(torch.load(args.sim_checkpoint, map_location="cpu"))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net.eval().to(dev)
    images = np.load(args.out / "images_full.npy", mmap_mode="r")
    out = []
    with torch.no_grad():
        for start in range(0, len(images), 128):
            x = torch.from_numpy(((images[start:start + 128] / 255.0 - RGB_MEAN) / RGB_STD).astype(np.float32)).permute(0, 3, 1, 2)
            out.append(torch.sigmoid(net(x.to(dev))).squeeze(1).cpu().numpy())
    sim_tau = next((m["threshold"] for m in json.loads((args.sim_checkpoint.parent / "metrics.json").read_text())["results"] if m["model"] == "finetuned_resnet18"), 0.5)
    judge("sim_on_real/finetuned_resnet18", np.concatenate(out), sim_tau)
    results["sim_on_sim/finetuned_resnet18"] = {"auroc": next(m["auroc"] for m in json.loads((args.sim_checkpoint.parent / "metrics.json").read_text())["results"] if m["model"] == "finetuned_resnet18")}

    # 3. Real-trained on real (5-fold), real-trained on sim test, and sim + real together.
    rscore = np.full(len(rows), np.nan)
    mixed = np.full(len(rows), np.nan)
    sim_on = []
    for i in range(FOLDS):
        train, _, test = train_val_test(runs, folds, i)
        trm = np.isin(names, list(train)) & (y >= 0)
        te = np.isin(names, list(test))
        scorer = fit(real_x[trm], y[trm], "logreg")
        rscore[te] = scorer(real_x[te])
        sim_on.append(auroc(scorer(part["test"]), label["test"]))
        both = fit(np.concatenate([real_x[trm], part["train"][tr]]), np.concatenate([y[trm], label["train"][tr]]), "logreg")
        mixed[te] = both(real_x[te])
    results["real_on_real/logreg"] = {"auroc": auroc(rscore, np.where(np.isnan(rscore), -1, y))}
    results["real_on_sim/logreg"] = {"auroc": float(np.mean(sim_on)), "auroc_per_fold": sim_on}
    results["sim_plus_real_on_real/logreg"] = {"auroc": auroc(mixed, np.where(np.isnan(mixed), -1, y))}

    # 4. The shift itself, in ResNet-18 feature space.
    sim_groups = np.array([r["seed"] for r in ordered])
    results["shift"] = {
        "domain_classifier_auroc": domain_auroc(sim_x, sim_groups, real_x, names),
        "frechet_sim_vs_real": frechet(part["test"][::2], real_x[::2]),
        "frechet_sim_train_vs_sim_test": frechet(part["train"][::4], part["test"]),
        "frechet_real_half_vs_real_half": frechet(real_x[np.isin(names, [n for n in sorted(runs)[::2]])], real_x[np.isin(names, [n for n in sorted(runs)[1::2]])]),
    }
    keep = label["train"] == 1
    mean = part["train"][keep].mean(0)
    from sklearn.covariance import LedoitWolf

    precision = LedoitWolf().fit(part["train"][keep] - mean).precision_
    distance = lambda x: np.sqrt(np.einsum("ij,jk,ik->i", x - mean, precision, x - mean))
    limit = float(np.quantile(distance(part["val"][label["val"] == 1]), 0.95))
    clean = np.array([r["condition_run"] == "clean" for r in rows])
    results["shift"]["real_clean_floor_beyond_sim_95pct_distance"] = float(np.mean(distance(real_x[clean]) > limit))
    results["shift"]["sim_test_clean_beyond_sim_95pct_distance"] = float(np.mean(distance(part["test"][label["test"] == 1]) > limit))
    (args.out / "sim_to_real.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps({k: (v if k == "shift" else {m: round(v[m], 3) for m in ("auroc", "balanced_accuracy") if m in v}) for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
