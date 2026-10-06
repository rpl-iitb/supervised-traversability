"""PyTorch re-implementation of GONet (Hirose et al., IROS 2018).

Architecture follows the authors' released inference code
(github.com/NHirose/GONET, Chainer 4.1) layer for layer, so their trained
weights can be loaded with :func:`load_official_weights` to verify this port
before it is retrained:

* ``Generator``   z (100) -> 3x128x128 image; DCGAN, ReLU, *no* output tanh.
* ``InvGen``      image -> z; the generator's mirror, trained so that
                  ``Generator(InvGen(x))`` reconstructs safe images.
* ``Discriminator`` image -> 512x8x8 feature map (ELU); a 2-way real/fake
                  head ``l4l`` is used only while training the GAN.
* ``GONetHead``   three linear scores on |x - x'|, |f(x) - f(x')| and f(x),
                  combined by a 3->1 linear layer and a sigmoid.

Images are RGB, 128x128, scaled as ``(pixel - 128) / 128``.

The paper's semi-supervised recipe (implemented in ``gonet_train.py``):
1. train the DCGAN on positive (traversable) images only;
2. train InvGen on positives with the generator frozen;
3. train only GONetHead on a small labelled positive + negative set.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


NZ = 100
IMAGE_SIZE = 128
# Chainer's BatchNormalization default; PyTorch's is 1e-5. Matching it makes
# the official weights reproduce the original outputs exactly.
BN_EPS = 2e-5


def _bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=BN_EPS)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0z = nn.Linear(NZ, 8 * 8 * 512)
        self.bn0l = nn.BatchNorm1d(8 * 8 * 512, eps=BN_EPS)
        self.dc1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dc2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dc3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dc4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
        # bn0 exists in the released checkpoint but is unused in forward().
        self.bn0, self.bn1, self.bn2, self.bn3 = _bn(512), _bn(256), _bn(128), _bn(64)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn0l(self.l0z(z))).view(-1, 512, 8, 8)
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        return self.dc4(h)


class _Encoder(nn.Module):
    """Shared 4-conv downsampling trunk of InvGen and the Discriminator."""

    def __init__(self, activation):
        super().__init__()
        self.c0 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        # bn0 exists in the released checkpoints but is unused in forward().
        self.bn0, self.bn1, self.bn2, self.bn3 = _bn(64), _bn(128), _bn(256), _bn(512)
        self.act = activation

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c0(x))
        h = self.act(self.bn1(self.c1(h)))
        h = self.act(self.bn2(self.c2(h)))
        return self.act(self.bn3(self.c3(h)))


class InvGen(_Encoder):
    def __init__(self):
        super().__init__(F.relu)
        self.l4l = nn.Linear(8 * 8 * 512, NZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l4l(self.features(x).flatten(1))


class Discriminator(_Encoder):
    def __init__(self):
        super().__init__(F.elu)
        self.l4l = nn.Linear(8 * 8 * 512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """512x8x8 feature map (what GONet's classifier consumes)."""
        return self.features(x)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Real/fake logits, used only when training the GAN."""
        return self.l4l(self.features(x).flatten(1))


class GONetHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_img = nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 1)
        self.l_dis = nn.Linear(512 * 8 * 8, 1)
        self.l_fdis = nn.Linear(512 * 8 * 8, 1)
        self.l_FL = nn.Linear(3, 1)

    def forward(self, img_error, dis_error, dis_real) -> torch.Tensor:
        h = self.l_img(img_error.abs().flatten(1))
        g = self.l_dis(dis_error.abs().flatten(1))
        f = self.l_fdis(dis_real.flatten(1))
        return torch.sigmoid(self.l_FL(torch.cat((h, g, f), dim=1))).squeeze(1)


class GONet(nn.Module):
    """Traversable probability for a batch of normalised 128x128 images."""

    def __init__(self):
        super().__init__()
        self.gen, self.invg, self.dis, self.head = Generator(), InvGen(), Discriminator(), GONetHead()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.gen(self.invg(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rec = self.reconstruct(x)
        dis_real, dis_rec = self.dis(x), self.dis(x_rec)
        return self.head(x - x_rec, dis_real - dis_rec, dis_real)


def normalise(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW RGB -> float as in the original code: (x - 128) / 128."""
    return (images_uint8.float() - 128.0) / 128.0


# --- Loading the authors' Chainer checkpoints -------------------------------

def _chainer_state(path: Path) -> dict[str, torch.Tensor]:
    import h5py

    state = {}
    with h5py.File(path, "r") as file:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.shape != ():
                state[name] = torch.from_numpy(obj[()])
        file.visititems(visit)
    return state


_CHAINER_TO_TORCH = {"W": "weight", "b": "bias", "gamma": "weight", "beta": "bias",
                     "avg_mean": "running_mean", "avg_var": "running_var"}


def _load_into(module: nn.Module, path: Path) -> None:
    state = {}
    for name, tensor in _chainer_state(path).items():
        layer, param = name.split("/")
        state[f"{layer}.{_CHAINER_TO_TORCH[param]}"] = tensor
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [m for m in missing if not m.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"{path.name}: missing {missing}, unexpected {unexpected}")


def load_official_weights(model: GONet, directory: str | Path, variant: str = "single") -> GONet:
    """Load featlayer_{gen,invg,dis}_<variant>.h5 and classlayer_core.h5."""
    directory = Path(directory)
    _load_into(model.gen, directory / f"featlayer_gen_{variant}.h5")
    _load_into(model.invg, directory / f"featlayer_invg_{variant}.h5")
    _load_into(model.dis, directory / f"featlayer_dis_{variant}.h5")
    _load_into(model.head, directory / "classlayer_core.h5")
    return model
