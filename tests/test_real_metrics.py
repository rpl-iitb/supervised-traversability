import unittest

import numpy as np

from learning.real_metrics import clean_run_metrics, crossing_metrics, false_alarms, frame_metrics, has_run


def row(run, order, y=1, instance="", kind="clean_floor", condition="hazard"):
    return {"run": run, "frame_order": order, "y": y, "hazard_instance": instance, "patch_class": kind, "condition_run": condition}


class MetricsTest(unittest.TestCase):
    def test_single_flags_do_not_count_as_a_stop(self):
        self.assertFalse(has_run(np.array([1, 0, 1, 0, 1, 1], bool)))
        self.assertTrue(has_run(np.array([0, 1, 1, 1], bool)))
        self.assertEqual(false_alarms(np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 1], bool)), 2)

    def test_false_stops_per_100m_on_clean_runs_only(self):
        rows = [row("c", i, condition="clean") for i in range(100)] + [row("h", i) for i in range(10)]
        flag = np.zeros(110); flag[10:14] = 1; flag[100:110] = 1
        out = clean_run_metrics(rows, flag)
        self.assertAlmostEqual(out["clean_frame_flag_rate"], 0.04)
        self.assertAlmostEqual(out["false_stops_per_100m"], 25.0)  # 1 stop in 4 m

    def test_a_crossing_is_bad_when_any_strip_photo_is_labelled_non_traversable(self):
        rows = [row("h", i, y=0, instance="lid-0", kind="lid") for i in range(4)] + [row("h", 4 + i, y=None, instance="cloth-1", kind="cloth") for i in range(4)]
        flag = np.array([1, 1, 1, 0, 0, 1, 0, 0], float)
        out = crossing_metrics(rows, flag)
        self.assertEqual(out["bad_crossing_detected"], 1.0)
        self.assertEqual(out["mild_crossing_flagged"], 0.0)
        self.assertEqual(out["per_object"]["lid"]["bad_crossings"], 1)

    def test_frame_metrics_ignore_gray_and_unscored_rows(self):
        rows = [row("a", 0, y=1), row("a", 1, y=0), row("a", 2, y=None), row("b", 0, y=1)]
        score = np.array([0.9, 0.1, 0.5, np.nan]); flag = np.array([0, 1, 0, np.nan])
        out = frame_metrics(rows, score, flag)
        self.assertEqual(out["frames"], 2)
        self.assertEqual(out["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
