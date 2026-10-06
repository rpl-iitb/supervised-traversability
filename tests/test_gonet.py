import unittest
from pathlib import Path

import torch

from learning.gonet import GONet, load_official_weights, normalise


OFFICIAL = Path(__file__).resolve().parents[1] / "data" / "gonet" / "official_weights"


class GONetShapeTest(unittest.TestCase):
    def test_forward_shapes(self):
        model = GONet().eval()
        x = normalise(torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.uint8))
        with torch.no_grad():
            self.assertEqual(model.invg(x).shape, (2, 100))
            self.assertEqual(model.reconstruct(x).shape, (2, 3, 128, 128))
            self.assertEqual(model.dis(x).shape, (2, 512, 8, 8))
            self.assertEqual(model.dis.logits(x).shape, (2, 2))
            p = model(x)
        self.assertEqual(p.shape, (2,))
        self.assertTrue(bool(((p >= 0) & (p <= 1)).all()))

    def test_normalise_matches_original_scaling(self):
        x = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8)
        self.assertTrue(torch.allclose(normalise(x), torch.tensor([[[[-1.0, 0.0, 127 / 128]]]])))


@unittest.skipUnless((OFFICIAL / "classlayer_core.h5").exists(), "official GONet weights not downloaded")
class OfficialWeightsTest(unittest.TestCase):
    def test_every_checkpoint_tensor_loads(self):
        model = load_official_weights(GONet(), OFFICIAL).eval()
        x = normalise(torch.full((1, 3, 128, 128), 128, dtype=torch.uint8))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(x)).all())


if __name__ == "__main__":
    unittest.main()
