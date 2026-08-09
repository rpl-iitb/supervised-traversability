import unittest


class WarehouseFixedRunTest(unittest.TestCase):
    def test_wheel_joint_paths_returns_only_the_two_drive_wheels(self):
        from scripts.warehouse_fixed_run import wheel_joint_paths

        names = ["joint_caster_left", "joint_wheel_left", "joint_swing_right", "joint_wheel_right"]
        self.assertEqual(wheel_joint_paths(names), ["joint_wheel_left", "joint_wheel_right"])

    def test_csv_row_uses_the_stable_log_schema(self):
        from scripts.warehouse_fixed_run import csv_row

        self.assertEqual(
            csv_row(0.1, 1.0, 2.0, 0.3, 0.25, 120.0),
            {
                "elapsed_s": 0.1,
                "x_m": 1.0,
                "y_m": 2.0,
                "yaw_rad": 0.3,
                "speed_mps": 0.25,
                "command_wheel_deg_s": 120.0,
            },
        )

    def test_motion_prim_path_uses_the_dynamic_chassis_link(self):
        from scripts.warehouse_fixed_run import motion_prim_path

        paths = [
            "/World/nova_carter_reference",
            "/World/nova_carter_reference/chassis_link",
            "/World/nova_carter_reference/chassis_link/visual",
        ]

        self.assertEqual(
            motion_prim_path(paths),
            "/World/nova_carter_reference/chassis_link",
        )
