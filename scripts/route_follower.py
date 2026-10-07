"""Scripted route follower shared by every collector.

It stands in for the teleoperator in automated collection: ramp to a cruise
speed and steer back onto the straight manifest route. Keeping it in one
place guarantees the MuJoCo proxy and the Isaac collector issue identical
commands, so their episodes are comparable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


CRUISE_SPEED_MPS = 0.5
RAMP_S = 1.0
HEADING_GAIN = 1.5
CROSS_TRACK_GAIN = 1.2
MAX_YAW_RATE_RPS = 0.6
STALL_SPEED_MPS = 0.03
STALL_TIME_S = 2.0
TIMEOUT_S = 30.0
# Keep driving past the manifest goal so frames seen near the goal still get
# their look-ahead patch traversed (the labeller needs ~2.2 m of future).
LOOKAHEAD_TAIL_M = 2.3


@dataclass(frozen=True)
class Route:
    start_x_m: float
    start_y_m: float
    yaw_rad: float
    length_m: float

    @classmethod
    def from_manifest(cls, scenario: dict) -> "Route":
        start, goal = scenario["route"]["start_pose"], scenario["route"]["goal_pose"]
        return cls(
            start["x_m"],
            start["y_m"],
            math.atan2(goal["y_m"] - start["y_m"], goal["x_m"] - start["x_m"]),
            scenario["route"]["length_m"],
        )

    def progress_and_offset(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Distance along the route and signed lateral offset (left positive)."""
        dx, dy = x_m - self.start_x_m, y_m - self.start_y_m
        cos_yaw, sin_yaw = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw


def route_command(route: Route, t_s: float, x_m: float, y_m: float, yaw_rad: float) -> tuple[float, float]:
    """Return ``(v_mps, yaw_rate_rps)`` for the current pose."""
    _, offset = route.progress_and_offset(x_m, y_m)
    v = CRUISE_SPEED_MPS * min(1.0, max(0.0, t_s) / RAMP_S)
    heading_error = math.atan2(math.sin(route.yaw_rad - yaw_rad), math.cos(route.yaw_rad - yaw_rad))
    w = HEADING_GAIN * heading_error - CROSS_TRACK_GAIN * offset
    return v, max(-MAX_YAW_RATE_RPS, min(MAX_YAW_RATE_RPS, w))


class TerminationMonitor:
    """Goal / stall / timeout rules applied identically in every simulator."""

    def __init__(self, route: Route, tail_m: float = LOOKAHEAD_TAIL_M, timeout_s: float = TIMEOUT_S):
        self.route = route
        self.tail_m = tail_m
        self.timeout_s = timeout_s
        self._stall_since: float | None = None

    def update(self, t_s: float, x_m: float, y_m: float, v_fwd_mps: float, cmd_v_mps: float) -> dict | None:
        progress, _ = self.route.progress_and_offset(x_m, y_m)
        if progress >= self.route.length_m + self.tail_m:
            return {"reason": "goal_reached", "t_s": round(t_s, 3)}
        if cmd_v_mps > 0.2 and abs(v_fwd_mps) < STALL_SPEED_MPS:
            self._stall_since = t_s if self._stall_since is None else self._stall_since
            if t_s - self._stall_since >= STALL_TIME_S:
                return {"reason": "stalled", "t_s": round(t_s, 3), "progress_m": round(progress, 3)}
        else:
            self._stall_since = None
        if t_s >= self.timeout_s:
            return {"reason": "timeout", "t_s": round(t_s, 3), "progress_m": round(progress, 3)}
        return None


# Small, smoothly varying command disturbances imitating real controller and
# floor imperfections, so low-friction patches can actually cause slip.
# ASSUMED magnitudes (to confirm against a real robot's logs).
YAW_WOBBLE_SIGMA_RPS = 0.08
YAW_WOBBLE_TAU_S = 0.5
SPEED_WOBBLE_SIGMA_MPS = 0.03
SPEED_WOBBLE_TAU_S = 1.0


class DrivingDisturbance:
    """Ornstein-Uhlenbeck wobble added to (v, w) commands.

    Seeded per scene, so a scene's hazard run and its clean twin receive the
    identical disturbance sequence and remain a fair pair.
    """

    def __init__(self, seed: int, enabled: bool = True):
        self.rng = random.Random(1_000_003 * (seed + 1))
        self.enabled = enabled
        self.dv = 0.0
        self.dw = 0.0

    def step(self, dt_s: float) -> tuple[float, float]:
        if not self.enabled:
            return 0.0, 0.0
        for attr, sigma, tau in (("dv", SPEED_WOBBLE_SIGMA_MPS, SPEED_WOBBLE_TAU_S),
                                 ("dw", YAW_WOBBLE_SIGMA_RPS, YAW_WOBBLE_TAU_S)):
            decay = math.exp(-dt_s / tau)
            value = getattr(self, attr) * decay + sigma * math.sqrt(1.0 - decay * decay) * self.rng.gauss(0.0, 1.0)
            setattr(self, attr, value)
        return self.dv, self.dw

    @staticmethod
    def config() -> dict:
        return {"yaw_sigma_rps": YAW_WOBBLE_SIGMA_RPS, "yaw_tau_s": YAW_WOBBLE_TAU_S,
                "speed_sigma_mps": SPEED_WOBBLE_SIGMA_MPS, "speed_tau_s": SPEED_WOBBLE_TAU_S}
