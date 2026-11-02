import unittest

from scripts.real_labels import fit_offset


class FitOffsetTest(unittest.TestCase):
    def test_recovers_a_late_start_even_when_one_object_made_no_zone(self):
        placed = [2.7, 5.2, 7.6, 10.1, 12.6, 15.0]
        zones = [d - 0.7 for d in placed if d != 7.6]
        self.assertAlmostEqual(fit_offset(placed, zones), -0.7, delta=0.011)

    def test_no_zones_or_no_objects_leaves_the_placements_alone(self):
        self.assertEqual(fit_offset([2.0], []), 0.0)
        self.assertEqual(fit_offset([], [2.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
