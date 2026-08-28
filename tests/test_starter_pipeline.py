import unittest


class StarterPipelineTest(unittest.TestCase):
    def test_seeded_scenario_is_reproducible(self):
        from scripts.starter_pipeline import create_starter_scenario

        first = create_starter_scenario(seed=17)
        second = create_starter_scenario(seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first["scenario_id"], "starter-0017")
        self.assertEqual(first["robot"]["model"], "nova_carter")
        self.assertEqual(
            [item["asset_key"] for item in first["objects"]],
            ["cable_hose", "low_debris", "small_lip"],
        )

    def test_scenario_keeps_hazards_clear_of_robot_and_each_other(self):
        from scripts.starter_pipeline import create_starter_scenario, minimum_clearance_ok

        scenario = create_starter_scenario(seed=23)

        self.assertTrue(minimum_clearance_ok(scenario))

    def test_label_marks_normal_crossing_as_traversable(self):
        from scripts.starter_pipeline import label_from_outcome

        self.assertEqual(
            label_from_outcome(
                crossing_completed=True,
                speed_tracking_ratio=0.92,
                safety_stop=False,
                contact_impulse_ns=0.0,
            ),
            {"score": 0.92, "binary_label": 1, "reason": "normal_crossing"},
        )

    def test_label_marks_safety_stop_as_not_traversable(self):
        from scripts.starter_pipeline import label_from_outcome

        self.assertEqual(
            label_from_outcome(
                crossing_completed=False,
                speed_tracking_ratio=0.75,
                safety_stop=True,
                contact_impulse_ns=0.0,
            )["binary_label"],
            0,
        )

    def test_label_keeps_borderline_outcomes_unlabelled(self):
        from scripts.starter_pipeline import label_from_outcome

        self.assertIsNone(
            label_from_outcome(
                crossing_completed=False,
                speed_tracking_ratio=0.7,
                safety_stop=False,
                contact_impulse_ns=0.0,
            )["binary_label"]
        )
