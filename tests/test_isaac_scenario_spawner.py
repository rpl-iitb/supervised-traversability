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
