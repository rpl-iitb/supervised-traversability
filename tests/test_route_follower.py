import math
import unittest

from scripts.route_follower import CRUISE_SPEED_MPS, LOOKAHEAD_TAIL_M, STALL_TIME_S, Route, TerminationMonitor, route_command


ROUTE = Route(0.0, 0.0, 0.0, 6.0)


class RouteFollowerTest(unittest.TestCase):
    def test_from_manifest_uses_start_to_goal_heading(self):
        route = Route.from_manifest({"route": {"start_pose": {"x_m": 1.0, "y_m": 1.0}, "goal_pose": {"x_m": 1.0, "y_m": 7.0}, "length_m": 6.0}})
        self.assertAlmostEqual(route.yaw_rad, math.pi / 2)

    def test_speed_ramps_then_cruises(self):
        self.assertAlmostEqual(route_command(ROUTE, 0.5, 0, 0, 0)[0], CRUISE_SPEED_MPS / 2)
        self.assertAlmostEqual(route_command(ROUTE, 5.0, 0, 0, 0)[0], CRUISE_SPEED_MPS)

    def test_steers_back_toward_route(self):
        self.assertLess(route_command(ROUTE, 5.0, 1.0, 0.3, 0.0)[1], 0.0)
        self.assertGreater(route_command(ROUTE, 5.0, 1.0, -0.3, 0.0)[1], 0.0)

    def test_goal_and_stall(self):
        monitor = TerminationMonitor(ROUTE)
        self.assertIsNone(monitor.update(10.0, 6.1, 0.0, 0.5, 0.5))
        self.assertEqual(monitor.update(10.0, 6.1 + LOOKAHEAD_TAIL_M, 0.0, 0.5, 0.5)["reason"], "goal_reached")
        monitor = TerminationMonitor(ROUTE)
        self.assertIsNone(monitor.update(3.0, 2.0, 0.0, 0.0, 0.5))
        self.assertEqual(monitor.update(3.0 + STALL_TIME_S, 2.0, 0.0, 0.0, 0.5)["reason"], "stalled")

    def test_motion_resets_stall_timer(self):
        monitor = TerminationMonitor(ROUTE)
        monitor.update(3.0, 2.0, 0.0, 0.0, 0.5)
        monitor.update(4.0, 2.0, 0.0, 0.4, 0.5)
        self.assertIsNone(monitor.update(3.0 + STALL_TIME_S + 0.5, 2.0, 0.0, 0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
