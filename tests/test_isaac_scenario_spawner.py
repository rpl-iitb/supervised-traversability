import unittest


CABLE = {
    "asset_key": "cable_hose",
    "instance_id": "cable_hose_01",
    "footprint_radius_m": 0.18,
    "height_m": 0.03,
    "pose": {"x_m": 1.0, "y_m": -0.5, "yaw_rad": 1.571},
}

DEBRIS = {
    "asset_key": "low_debris",
    "instance_id": "low_debris_02",
    "footprint_radius_m": 0.20,
    "height_m": 0.08,
    "pose": {"x_m": 2.0, "y_m": 0.0, "yaw_rad": 0.0},
}

PLASTIC_WRAPPER = {
    "asset_key": "plastic_wrapper",
    "instance_id": "plastic_wrapper_01",
    "footprint_radius_m": 0.22,
    "height_m": 0.025,
    "spawn_mode": "procedural_collision",
    "pose": {"x_m": 2.0, "y_m": 0.0, "yaw_rad": 0.0},
}

OIL_SPILL = {
    "asset_key": "oil_spill",
    "instance_id": "oil_spill_01",
    "footprint_radius_m": 0.45,
    "height_m": 0.002,
    "spawn_mode": "procedural_visual_only",
    "pose": {"x_m": 2.0, "y_m": 0.0, "yaw_rad": 0.0},
}

CARDBOARD_BOX = {
    "asset_key": "cardboard_box",
    "instance_id": "cardboard_box_01",
    "footprint_radius_m": 0.28,
    "height_m": 0.35,
    "spawn_mode": "isaac_usd_reference",
    "usd_relative_path": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
    "pose": {"x_m": 2.0, "y_m": 0.0, "yaw_rad": 0.0},
}


class IsaacScenarioSpawnerTest(unittest.TestCase):
    def test_proxy_spec_maps_cable_to_oriented_capsule(self):
        from scripts.isaac_scenario_spawner import proxy_spec

        self.assertEqual(
            proxy_spec(CABLE),
            {
                "primitive": "capsule",
                "translation_m": (1.0, -0.5, 0.03),
                "rotation_deg": (0.0, 90.0, 90.01),
                "scale": (1.0, 1.0, 1.0),
                "color": (0.95, 0.45, 0.05),
            },
        )

    def test_proxy_spec_maps_low_debris_to_cube(self):
        from scripts.isaac_scenario_spawner import proxy_spec

        self.assertEqual(
            proxy_spec(DEBRIS),
            {
                "primitive": "cube",
                "translation_m": (2.0, 0.0, 0.04),
                "rotation_deg": (0.0, 0.0, 0.0),
                "scale": (0.4, 0.4, 0.08),
                "color": (0.85, 0.15, 0.15),
            },
        )

    def test_proxy_spec_rejects_unknown_asset(self):
        from scripts.isaac_scenario_spawner import proxy_spec

        with self.assertRaisesRegex(ValueError, "Unsupported starter asset"):
            proxy_spec({**DEBRIS, "asset_key": "mystery"})

    def test_spawn_spec_uses_semantic_procedural_models_for_anomalies(self):
        from scripts.isaac_scenario_spawner import spawn_spec

        self.assertEqual(spawn_spec(CABLE)["kind"], "curved_cable")
        self.assertEqual(spawn_spec(PLASTIC_WRAPPER)["kind"], "crumpled_wrapper")
        self.assertEqual(spawn_spec(OIL_SPILL)["kind"], "irregular_spill")
        self.assertFalse(spawn_spec(OIL_SPILL)["has_collision"])

    def test_spawn_spec_uses_an_isaac_reference_for_normal_clutter(self):
        from scripts.isaac_scenario_spawner import spawn_spec

        spec = spawn_spec(CARDBOARD_BOX)

        self.assertEqual(spec["kind"], "isaac_usd_reference")
        self.assertEqual(
            spec["usd_relative_path"],
            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd",
        )
