# Checked through its outputs and the split/metric tests.
"""End-to-end fine-tuning on real data: ResNet-18 and the deployable MobileNetV3-Small.

    python3 -m learning.real_finetune --data DATA --out runs/real_v1 --stage cv

Stages: cv (5-fold over runs), direction, session, looo (leave one object out),
siminit (sim-trained start vs ImageNet start, with few real runs), seeds (cv repeated with
other run assignments and network seeds), mildsafe (photos of mild-bump floor, otherwise
unused, train as traversable: can a model tell a mild object from a hard one?). Epoch and threshold
come from validation runs. Results are written after every experiment, finished ones
are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from learning.real_data import RGB_MEAN, RGB_STD, build_cache, load_rows, run_folds, run_table, train_val_test
from learning.real_metrics import OBJECTS, bootstrap, choose_threshold, summary

FOLDS = 5
MEAN, STD = torch.tensor(RGB_MEAN, dtype=torch.float32).view(3, 1, 1), torch.tensor(RGB_STD, dtype=torch.float32).view(3, 1, 1)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def build(arch: str, sim_checkpoint: Path | None = None):
    from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights, mobilenet_v3_small, resnet18

    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(512, 1)
        if sim_checkpoint is not None:
            model.load_state_dict(torch.load(sim_checkpoint, map_location="cpu"))
    else:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    return model


def to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255.0)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Same object, different light and side: flip, brightness, contrast (per image)."""
    n = len(x)
    flip = torch.rand(n) < 0.5
    x[flip] = x[flip].flip(-1)
    bright = 0.7 + 0.6 * torch.rand(n, 1, 1, 1)
    contrast = 0.7 + 0.6 * torch.rand(n, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * contrast + mean * bright).clamp_(0.0, 1.0)


@torch.no_grad()
def predict(model, images, index, dev, batch=256) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(index), batch):
        x = (to_tensor(images[index[start:start + batch]]) - MEAN) / STD
        out.append(model(x.to(dev)).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_model(arch, images, y, train_idx, val_idx, *, epochs=5, seed=0, sim_checkpoint=None, per_epoch=6000):
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = build(arch, sim_checkpoint).to(dev)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    pos, neg = train_idx[y[train_idx] == 1], train_idx[y[train_idx] == 0]
    best, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        half = per_epoch // 2
        picks = np.concatenate([rng.choice(pos, half), rng.choice(neg, half)])
        rng.shuffle(picks)
        started, total = time.time(), 0.0
        for start in range(0, len(picks) - 63, 64):
            index = picks[start:start + 64]
            order = np.argsort(index)
            x = to_tensor(images[index[order]])
            targets = torch.from_numpy(y[index[order]].astype(np.float32))
            x = (augment(x) - MEAN) / STD
            optimiser.zero_grad()
            loss = loss_fn(model(x.to(dev)).squeeze(1), targets.to(dev))
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        value = roc_auc_score(y[val_idx], predict(model, images, val_idx, dev))
        if value > best:
            best, best_state, best_epoch = value, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch
        print(f"    epoch {epoch}: loss {total / max(1, len(picks) // 64):.3f} val AUROC {value:.4f} ({time.time() - started:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model, best_epoch


class Runner:
    def __init__(self, args):
        self.args = args
        self.rows = load_rows(args.data)
        for order, r in enumerate(self.rows):
            r["frame_order"] = order
        self.runs = run_table(self.rows)
        self.folds = run_folds(self.runs, FOLDS)
        self.y = np.array([-1 if r["y"] is None else r["y"] for r in self.rows])
        self.names = np.array([r["run"] for r in self.rows])
        self.path = args.out / f"finetune_results_{args.tag}.json"
        self.results = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.scores_path = args.out / "finetune_scores"
        self.scores_path.mkdir(parents=True, exist_ok=True)

    def images(self, mode):
        return build_cache(self.args.data, self.rows, mode, self.args.out / f"images_{mode}.npy")

    def split(self, images, arch, train, val, test, score, flag, drop=None, epochs=5, sim=None, save=None, seed=0, y=None):
        keep = np.ones(len(self.rows), bool) if drop is None else ~drop
        y = self.y if y is None else y
        mask = lambda names: np.isin(self.names, list(names))
        tr = np.flatnonzero(mask(train) & (y >= 0) & keep)
        va = np.flatnonzero(mask(val) & (y >= 0) & keep)
        te = np.flatnonzero(mask(test))
        model, best_epoch = train_model(arch, images, y, tr, va, epochs=epochs, sim_checkpoint=sim, seed=seed)
        tau = choose_threshold(predict(model, images, va, device()), y[va])
        s = predict(model, images, te, device())
        score[te], flag[te] = s, (s < tau).astype(float)
        if save is not None:
            torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "tau": tau, "arch": arch}, save)
        return best_epoch

    def record(self, key, score, flag, ci=True):
        out = summary(self.rows, score, flag)
        if ci:
            out["ci95"] = bootstrap(self.rows, score, flag)
        out["test_frames_scored"] = int((~np.isnan(score)).sum())
        self.results[key] = out
        self.path.write_text(json.dumps(self.results, indent=1) + "\n")
        np.savez_compressed(self.scores_path / (key.replace("/", "__") + ".npz"), score=score, flag=flag)
        print(key, {k: round(out[k], 3) for k in ("auroc", "balanced_accuracy", "clean_frame_flag_rate", "false_stops_per_100m", "bad_crossing_detected", "mild_crossing_flagged")}, flush=True)

    def blank(self):
        return np.full(len(self.rows), np.nan), np.full(len(self.rows), np.nan)

    def cv(self):
        for mode, arch in (("mask", "resnet18"), ("mask", "mobilenet"), ("full", "resnet18")):
            key = f"cv5/{mode}/{arch}"
            if key in self.results:
                continue
            images, (score, flag) = self.images(mode), self.blank()
            for i in range(FOLDS):
                print(key, "fold", i, flush=True)
                self.split(images, arch, *train_val_test(self.runs, self.folds, i), score, flag,
                           save=self.args.out / f"finetuned_{mode}_{arch}_fold{i}.pt")
            self.record(key, score, flag)

    def holdouts(self, field, values, name):
        images = self.images("mask")
        for arch in ("resnet18", "mobilenet"):
            for value in values:
                key = f"{name}/mask/{arch}/train_{value}"
                if key in self.results:
                    continue
                train_all = sorted(k for k, v in self.runs.items() if v[field] == value)
                val = set(train_all[::5])
                test = {k for k, v in self.runs.items() if v[field] != value}
                score, flag = self.blank()
                self.split(images, arch, set(train_all) - val, val, test, score, flag)
                self.record(key, score, flag)

    def looo(self):
        images = self.images("mask")
        for arch in ("resnet18",):
            for obj in OBJECTS:
                key = f"looo/mask/{arch}/{obj}"
                if key in self.results:
                    continue
                drop = np.array([obj in r["in_roi"] for r in self.rows])
                score, flag = self.blank()
                for i in range(FOLDS):
                    print(key, "fold", i, flush=True)
                    self.split(images, arch, *train_val_test(self.runs, self.folds, i), score, flag, drop=drop, epochs=3)
                self.record(key, score, flag, ci=False)

    def seeds(self):
        images = self.images("mask")
        for seed in (1, 2, 3):
            folds = run_folds(self.runs, FOLDS, seed=seed)
            for arch in ("resnet18", "mobilenet"):
                key = f"seed{seed}/mask/{arch}"
                if key in self.results:
                    continue
                score, flag = self.blank()
                for i in range(FOLDS):
                    self.split(images, arch, *train_val_test(self.runs, folds, i), score, flag, seed=seed)
                self.record(key, score, flag, ci=False)

    def mildsafe(self):
        images = self.images("mask")
        y = np.array([1 if r["reason"] == "abnormal_but_acceptable" else v for r, v in zip(self.rows, self.y)])
        for arch in ("resnet18", "mobilenet"):
            key = f"mildsafe/mask/{arch}"
            if key in self.results:
                continue
            score, flag = self.blank()
            for i in range(FOLDS):
                self.split(images, arch, *train_val_test(self.runs, self.folds, i), score, flag, y=y)
            self.record(key, score, flag)

    def siminit(self):
        images = self.images("full")
        checkpoint = self.args.sim_checkpoint
        for n in (2, 4, 8):
            for start, sim in (("imagenet", None), ("sim", checkpoint)):
                key = f"siminit/full/resnet18/{start}/hazard_runs_{n}"
                if key in self.results:
                    continue
                rng = np.random.default_rng(n)
                score, flag = self.blank()
                for i in range(FOLDS):
                    train, val, test = train_val_test(self.runs, self.folds, i)
                    hazard = sorted(k for k in train if self.runs[k]["condition"] == "hazard")
                    clean = sorted(k for k in train if self.runs[k]["condition"] == "clean")
                    pick = set(rng.choice(hazard, min(n, len(hazard)), replace=False).tolist()) | set(clean[:1])
                    print(key, "fold", i, flush=True)
                    self.split(images, "resnet18", pick, val, test, score, flag, epochs=3, sim=sim)
                self.record(key, score, flag, ci=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", nargs="+", default=["cv"], choices=["cv", "direction", "session", "looo", "siminit", "seeds", "mildsafe"])
    parser.add_argument("--sim-checkpoint", type=Path, default=None)
    parser.add_argument("--tag", default="main", help="results file suffix, so parallel jobs do not overwrite each other")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    for stage in args.stage:
        if stage == "cv":
            runner.cv()
        elif stage == "direction":
            runner.holdouts("direction", ["A->B", "B->A"], "direction")
        elif stage == "session":
            runner.holdouts("session", ["s1a", "s1c"], "session")
        elif stage == "looo":
            runner.looo()
        elif stage == "seeds":
            runner.seeds()
        elif stage == "mildsafe":
            runner.mildsafe()
        else:
            runner.siminit()


if __name__ == "__main__":
    main()
