# Checked: reuses learning.gonet_train stages (tested), wrapper checked through its outputs.
"""GONet on real data: the authors' released weights, and retrained on our runs with the paper's three stages.

    python3 -m learning.real_gonet --data DATA --out runs/real_v1 --official data/gonet/official_weights

Full 128x128 frames, as GONet expects. Positive photos of the training runs train the GAN and InvGen; a
class-balanced labelled set trains the small head, early-stopped on validation runs.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from learning.gonet import GONet, load_official_weights, normalise
from learning.gonet_train import device, train_gan, train_invgen
from learning.real_data import load_rows, run_folds, run_table, train_val_test
from learning.real_metrics import bootstrap, choose_threshold, summary

FOLDS = 5


def cache_128(data: Path, rows: list[dict], cache: Path) -> np.ndarray:
    if cache.exists():
        images = np.load(cache, mmap_mode="r")
        if len(images) == len(rows):
            return images
    images = np.zeros((len(rows), 128, 128, 3), np.uint8)
    for i, r in enumerate(rows):
        images[i] = np.asarray(Image.open(data / "episodes" / r["rgb_path"]).convert("RGB").resize((128, 128)))
    np.save(cache, images)
    return images


@torch.no_grad()
def embed(model: GONet, images, index, dev, batch=128):
    parts = ([], [], [])
    for start in range(0, len(index), batch):
        x = normalise(torch.from_numpy(np.ascontiguousarray(images[index[start:start + batch]])).permute(0, 3, 1, 2)).to(dev)
        rec = model.reconstruct(x)
        real, fake = model.dis(x), model.dis(rec)
        for part, value in zip(parts, (x - rec, real - fake, real)):
            part.append(value.cpu())
    return tuple(torch.cat(p) for p in parts)


def train_head(model: GONet, train, y_train, val, y_val, seed=0):
    torch.manual_seed(seed)
    head = model.head.cpu().train()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else torch.nn.init.zeros_(p)
    optimiser = torch.optim.Adam(head.parameters(), lr=1e-3)
    labels, val_labels = torch.from_numpy(y_train.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32))
    best, state, patience = -1.0, None, 0
    for _ in range(500):
        order = torch.randperm(len(labels))
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            loss = F.mse_loss(head(*(f[idx] for f in train)), labels[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            acc = float(((head(*val) >= 0.5).float() == val_labels).float().mean())
        if acc > best:
            best, state, patience = acc, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 30:
                break
    head.load_state_dict(state)
    model.head = head
    return best


def score_all(model: GONet, images, index, dev, batch=256):
    feats = embed(model, images, index, dev, batch)
    model.head.eval()
    with torch.no_grad():
        out = []
        for start in range(0, len(index), batch):
            out.append(model.head.cpu()(*(f[start:start + batch] for f in feats)).numpy().ravel())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--gan-epochs", type=int, default=40)
    parser.add_argument("--invgen-epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(FOLDS)))
    args = parser.parse_args()
    rows = load_rows(args.data)
    for order, r in enumerate(rows):
        r["frame_order"] = order
    runs = run_table(rows)
    folds = run_folds(runs, FOLDS)
    y = np.array([-1 if r["y"] is None else r["y"] for r in rows])
    names = np.array([r["run"] for r in rows])
    images = cache_128(args.data, rows, args.out / "images_128.npy")
    dev = device()
    path = args.out / "gonet_results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    mask = lambda s: np.isin(names, list(s))

    # A. Released weights, trained on GO Stanford: no training on our data, threshold from validation runs and at 0.5.
    official = load_official_weights(GONet(), args.official).to(dev).eval()
    score, flag = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    every = score_all(official, images, np.arange(len(rows)), dev)
    official.to(dev)
    for i in range(FOLDS):
        _, val, test = train_val_test(runs, folds, i)
        va, te = np.flatnonzero(mask(val) & (y >= 0)), np.flatnonzero(mask(test))
        tau = choose_threshold(every[va], y[va])
        score[te], flag[te] = every[te], (every[te] < tau).astype(float)
    results["cv5/full128/gonet_official"] = {**summary(rows, score, flag), "ci95": bootstrap(rows, score, flag)}
    results["cv5/full128/gonet_official@0.5"] = summary(rows, score, (score < 0.5).astype(float))
    path.write_text(json.dumps(results, indent=1) + "\n")
    print("official", round(results["cv5/full128/gonet_official"]["auroc"], 3), flush=True)

    # B. Retrained per fold.
    score, flag = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for i in args.folds:
        out = args.out / f"gonet_fold{i}"
        train, val, test = train_val_test(runs, folds, i)
        tr, va, te = np.flatnonzero(mask(train) & (y >= 0)), np.flatnonzero(mask(val) & (y >= 0)), np.flatnonzero(mask(test))
        positives = out / "positives"
        if not positives.exists():
            positives.mkdir(parents=True)
            for index in tr[y[tr] == 1]:
                Image.fromarray(np.array(images[index])).save(positives / f"{index:06d}.jpg", quality=95)
        common = dict(positives=positives, out=out, batch=64, lr=2e-4, limit=None, workers=0, seed=i)
        if not (out / "stage1_gan.pt").exists():
            train_gan(Namespace(**common, epochs=args.gan_epochs, snapshot_epochs=set()))
        if not (out / "stage2_invgen.pt").exists():
            train_invgen(Namespace(**common, epochs=args.invgen_epochs, lam=0.1, gan_checkpoint=None))
        model = GONet()
        model.load_state_dict(torch.load(out / "stage2_invgen.pt", map_location="cpu"))
        model = model.to(dev).eval()
        rng = np.random.default_rng(i)
        pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
        n = min(len(pos), len(neg))
        pick = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
        train_feats, val_feats = embed(model, images, pick, dev), embed(model, images, va, dev)
        acc = train_head(model, train_feats, y[pick], val_feats, y[va], seed=i)
        model = model.to(dev).eval()
        val_score = score_all(model, images, va, dev)
        model = model.to(dev).eval()
        tau = choose_threshold(val_score, y[va])
        s = score_all(model, images, te, dev)
        score[te], flag[te] = s, (s < tau).astype(float)
        print("gonet fold", i, "val acc", round(acc, 3), flush=True)
        np.savez_compressed(args.out / "gonet_scores.npz", score=score, flag=flag)
        if not np.isnan(score).any() or i == args.folds[-1]:
            results["cv5/full128/gonet_retrained"] = {**summary(rows, score, flag), "ci95": bootstrap(rows, score, flag), "folds_done": args.folds[: args.folds.index(i) + 1]}
            path.write_text(json.dumps(results, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
