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

    def test_warehouse_creator_places_anomalies_on_a_reproducible_route(self):
        from scripts.starter_pipeline import create_warehouse_scenario

        first = create_warehouse_scenario(seed=23, origin_x_m=-18.8, origin_y_m=0.0)
        second = create_warehouse_scenario(seed=23, origin_x_m=-18.8, origin_y_m=0.0)

        self.assertEqual(first, second)
        self.assertEqual(first["scenario_kind"], "warehouse_traversability")
        self.assertEqual(first["route"]["length_m"], 6.0)
        self.assertTrue(any(item["role"] == "target_hazard" for item in first["objects"]))
        self.assertTrue(
            all(
                item["asset_key"]
                in {"cable_hose", "plastic_wrapper", "oil_spill", "small_lip", "cardboard_box"}
                for item in first["objects"]
            )
        )

    def test_hazard_catalog_distinguishes_collision_and_visual_only_hazards(self):
        from scripts.starter_pipeline import WAREHOUSE_OBJECT_CATALOG

        self.assertEqual(WAREHOUSE_OBJECT_CATALOG["cable_hose"]["spawn_mode"], "procedural_collision")
        self.assertEqual(WAREHOUSE_OBJECT_CATALOG["plastic_wrapper"]["spawn_mode"], "procedural_collision")
        self.assertEqual(WAREHOUSE_OBJECT_CATALOG["oil_spill"]["spawn_mode"], "procedural_visual_only")
        self.assertEqual(WAREHOUSE_OBJECT_CATALOG["cardboard_box"]["spawn_mode"], "isaac_usd_reference")
