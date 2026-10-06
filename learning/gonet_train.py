"""Train GONet from scratch with the paper's three-stage recipe.

    # 1. DCGAN on positive (traversable) images only
    python3 -m learning.gonet_train gan    --positives data/gonet/gs1/data_train/positive_L --out runs/gonet
    # 2. InvGen with the generator and discriminator frozen
    python3 -m learning.gonet_train invgen --positives data/gonet/gs1/data_train/positive_L --out runs/gonet
    # 3. the small classification head on the labelled set, early-stopped on validation
    python3 -m learning.gonet_train head   --annotation-root data/gonet/gs1 --out runs/gonet

Stated in the paper: positives-only GAN; InvGen loss
``(1 - lam) * |I - G(E(I))| + lam * |f(I) - f(G(E(I)))|`` with lam tuned on
validation; MSE loss and early stopping for the head; threshold 0.5.

NOT stated in the paper, so these are ASSUMPTIONS taken from the standard
DCGAN / AnoGAN recipes (recorded in each run's config.json): Adam
lr 2e-4, beta1 0.5, batch 64, lam 0.1, epoch counts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from learning.gonet import GONet, NZ, normalise


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ImageFolder(Dataset):
    def __init__(self, folder: Path, limit: int | None = None):
        self.paths = sorted(Path(folder).glob("*.jpg"))[:limit]
        if not self.paths:
            raise FileNotFoundError(f"No .jpg images in {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = np.asarray(Image.open(self.paths[index]).convert("RGB").resize((128, 128)))
        return normalise(torch.from_numpy(image.copy()).permute(2, 0, 1))


def _save(model: GONet, out: Path, name: str, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{name}.pt")
    (out / f"{name}_config.json").write_text(json.dumps(config, indent=2) + "\n")


def _load(out: Path, name: str) -> GONet:
    model = GONet()
    model.load_state_dict(torch.load(out / f"{name}.pt", map_location="cpu"))
    return model


def _sample_grid(images: torch.Tensor, path: Path) -> None:
    grid = ((images[:16].clamp(-1, 1) * 128 + 128).byte().permute(0, 2, 3, 1).cpu().numpy())
    rows = [np.concatenate(list(grid[r * 4:(r + 1) * 4]), axis=1) for r in range(4)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def train_gan(args) -> None:
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = device()
    data = DataLoader(ImageFolder(args.positives, args.limit), batch_size=args.batch, shuffle=True, drop_last=True,
                      num_workers=args.workers)
    model = GONet().to(dev)
    gen, dis = model.gen, model.dis
    gen_params = list(gen.parameters())
    dis_params = [p for n, p in dis.named_parameters() if not n.startswith("bn0")]
    opt_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(dis_params, lr=args.lr, betas=(0.5, 0.999))
    fixed_z = torch.randn(16, NZ, device=dev)
    config = {"stage": "gan", **{k: str(v) for k, v in vars(args).items() if k != "func"}, "images": len(data.dataset)}
    print(json.dumps(config), flush=True)
    for epoch in range(1, args.epochs + 1):
        started, d_total, g_total = time.time(), 0.0, 0.0
        gen.train(), dis.train()
        for real in data:
            real = real.to(dev)
            z = torch.randn(real.shape[0], NZ, device=dev)
            fake = gen(z)
            # Discriminator: class 1 = real, class 0 = generated (2-way head as in the release).
            ones = torch.ones(real.shape[0], dtype=torch.long, device=dev)
            loss_d = F.cross_entropy(dis.logits(real), ones) + F.cross_entropy(dis.logits(fake.detach()), 1 - ones)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            loss_g = F.cross_entropy(dis.logits(fake), ones)
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            d_total += float(loss_d.detach())
            g_total += float(loss_g.detach())
        gen.eval()
        with torch.no_grad():
            _sample_grid(gen(fixed_z), Path(args.out) / f"gan_samples_epoch{epoch:03d}.png")
        print(f"gan epoch {epoch}/{args.epochs}: D {d_total / len(data):.3f}  G {g_total / len(data):.3f}  "
              f"({time.time() - started:.0f}s)", flush=True)
        _save(model, Path(args.out), "stage1_gan", config | {"epochs_done": epoch})
        if epoch in args.snapshot_epochs:
            _save(model, Path(args.out), f"stage1_gan_e{epoch:03d}", config | {"epochs_done": epoch})


def train_invgen(args) -> None:
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = device()
    data = DataLoader(ImageFolder(args.positives, args.limit), batch_size=args.batch, shuffle=True, drop_last=True,
                      num_workers=args.workers)
    model = GONet()
    model.load_state_dict(torch.load(args.gan_checkpoint or Path(args.out) / "stage1_gan.pt", map_location="cpu"))
    model = model.to(dev)
    model.gen.eval(), model.dis.eval()
    for p in list(model.gen.parameters()) + list(model.dis.parameters()):
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.invg.parameters(), lr=args.lr, betas=(0.5, 0.999))
    config = {"stage": "invgen", **{k: str(v) for k, v in vars(args).items() if k != "func"}}
    print(json.dumps(config), flush=True)
    for epoch in range(1, args.epochs + 1):
        started, total = time.time(), 0.0
        model.invg.train()
        for real in data:
            real = real.to(dev)
            rec = model.reconstruct(real)
            loss = (1 - args.lam) * (real - rec).abs().mean() + args.lam * (model.dis(real) - model.dis(rec)).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        model.invg.eval()
        with torch.no_grad():
            batch = next(iter(data))[:8].to(dev)
            _sample_grid(torch.cat([batch, model.reconstruct(batch)]), Path(args.out) / f"invgen_recon_epoch{epoch:03d}.png")
        print(f"invgen epoch {epoch}/{args.epochs}: loss {total / len(data):.4f} ({time.time() - started:.0f}s)", flush=True)
        _save(model, Path(args.out), "stage2_invgen", config | {"epochs_done": epoch})


@torch.no_grad()
def _features(model: GONet, folder: Path, dev) -> tuple[torch.Tensor, ...]:
    images = torch.stack([ImageFolder(folder)[i] for i in range(len(ImageFolder(folder)))]).to(dev)
    rec = model.reconstruct(images)
    dis_real, dis_rec = model.dis(images), model.dis(rec)
    return (images - rec).cpu(), (dis_real - dis_rec).cpu(), dis_real.cpu()


def _split(model, root: Path, split: str, camera: str, dev):
    folder = root / f"data_{split}_annotation"
    pos, neg = _features(model, folder / f"positive_{camera}", dev), _features(model, folder / f"negative_{camera}", dev)
    feats = tuple(torch.cat([p, n]) for p, n in zip(pos, neg))
    labels = torch.cat([torch.ones(len(pos[0])), torch.zeros(len(neg[0]))])
    return feats, labels


def _accuracy(head, feats, labels) -> float:
    with torch.no_grad():
        return float(((head(*feats) >= 0.5).float() == labels).float().mean())


def train_head(args) -> None:
    torch.manual_seed(args.seed)
    dev = device()
    model = _load(Path(args.out), args.from_stage).to(dev).eval()
    root = Path(args.annotation_root)
    train, val, test = (_split(model, root, s, args.camera, dev) for s in ("train", "vali", "test"))
    head = model.head.cpu().train()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else torch.nn.init.zeros_(p)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    best, best_state, patience = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train[1]))
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            loss = F.mse_loss(head(*(f[idx] for f in train[0])), train[1][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_acc = _accuracy(head, *val)
        if val_acc > best:
            best, best_state, patience = val_acc, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    head.load_state_dict(best_state)
    result = {"camera": args.camera, "from_stage": args.from_stage, "epochs_run": epoch, "val_accuracy": round(best, 4),
              "test_accuracy": round(_accuracy(head, *test), 4), "paper_test_accuracy": 0.9425}
    print(json.dumps(result, indent=2))
    model.head = head
    _save(model.cpu(), Path(args.out), "stage3_gonet", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    for name, func, epochs, batch, lr in (("gan", train_gan, 20, 64, 2e-4), ("invgen", train_invgen, 10, 64, 2e-4)):
        p = sub.add_parser(name)
        p.add_argument("--positives", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--epochs", type=int, default=epochs)
        p.add_argument("--batch", type=int, default=batch)
        p.add_argument("--lr", type=float, default=lr)
        p.add_argument("--limit", type=int, default=None, help="use only the first N images (smoke tests)")
        p.add_argument("--workers", type=int, default=0)
        p.add_argument("--seed", type=int, default=0)
        if name == "gan":
            p.add_argument("--snapshot-epochs", type=lambda t: {int(v) for v in t.split(",") if v}, default=set(),
                           help="also keep checkpoints at these epochs, e.g. 10,20,30,40")
        if name == "invgen":
            p.add_argument("--lam", type=float, default=0.1)
            p.add_argument("--gan-checkpoint", type=Path, default=None, help="default: <out>/stage1_gan.pt")
        p.set_defaults(func=func)
    p = sub.add_parser("head")
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--from-stage", default="stage2_invgen")
    p.add_argument("--camera", default="L")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=train_head)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
