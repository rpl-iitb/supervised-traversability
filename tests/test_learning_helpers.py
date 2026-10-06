import unittest

import numpy as np

from learning.train_baselines import choose_threshold, roi_box, split_of
from scripts.episode_io import camera_model
from scripts.frame_labels import LabelConfig


class LearningHelpersTest(unittest.TestCase):
    def test_splits_are_disjoint_by_seed(self):
        self.assertEqual({split_of(s) for s in range(0, 80)}, {"train"})
        self.assertEqual({split_of(s) for s in range(80, 100)}, {"val"})
        self.assertEqual({split_of(s) for s in range(100, 120)}, {"test"})

    def test_roi_box_is_lower_image_region_inside_bounds(self):
        camera = camera_model(320, 240, 90.0, (0.31, 0.0, 0.35), 15.0, 10.0)
        u0, v0, u1, v1 = roi_box(camera, LabelConfig())
        self.assertTrue(0 <= u0 < u1 <= 320 and 0 <= v0 < v1 <= 240)
        self.assertGreater(v1, 200)

    def test_threshold_separates_perfectly_separable_scores(self):
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        tau = choose_threshold(scores, labels)
        self.assertTrue(0.3 < tau <= 0.7)


if __name__ == "__main__":
    unittest.main()
