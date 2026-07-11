"""
Synthetic rover telemetry generator.

This simulates what a real rover's sensor stream would look like:
- battery percentage (slowly drains, drops faster under strain)
- motor temperature per wheel (4 wheels)
- tilt (pitch/roll, reacts to terrain slope later in Week 2)
- comms signal strength (mostly stable, can be disrupted)
- position (x, y) - simple 2D for now, ties into terrain in Week 2

Faults can be injected on demand (for live demo control) or left to
occur "naturally" via small random walk noise.
"""

import random
import time
from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Fault:
    """An active fault being simulated. Decays back to normal over time
    unless it's a step-change fault."""
    fault_type: str          # e.g. "motor_temp_spike", "comms_dropout", "battery_drain"
    target: str = "general"  # e.g. "front_left" for a specific wheel
    magnitude: float = 1.0   # severity multiplier
    ticks_remaining: int = 20  # how many ticks the fault stays elevated


class RoverTelemetry:
    def __init__(self):
        self.battery_pct = 95.0
        self.motor_temp = {
            "front_left": 35.0,
            "front_right": 35.0,
            "rear_left": 35.0,
            "rear_right": 35.0,
        }
        self.tilt_deg = 0.0
        self.comms_signal = 95.0  # percentage signal strength
        self.position = {"x": 0.0, "y": 0.0}
        self.tick = 0
        self.active_faults: list[Fault] = []
        self.mode = "NOMINAL"  # "NOMINAL" | "COOL_DOWN" | "HIBERNATION"
        self.goal_pos = None
        self.replan_requested = False

    def inject_fault(self, fault_type: str, target: str = "general",
                      magnitude: float = 1.0, duration_ticks: int = 20):
        """Called by the API when the demo operator clicks 'inject fault'."""
        self.active_faults.append(
            Fault(fault_type=fault_type, target=target,
                  magnitude=magnitude, ticks_remaining=duration_ticks)
        )

    def _apply_faults(self):
        still_active = []
        for fault in self.active_faults:
            if fault.fault_type == "motor_temp_spike":
                wheel = fault.target if fault.target in self.motor_temp else "front_left"
                self.motor_temp[wheel] += 4.0 * fault.magnitude
            elif fault.fault_type == "battery_drain":
                self.battery_pct -= 0.8 * fault.magnitude
                self.battery_pct = max(0.0, self.battery_pct)
            elif fault.fault_type == "comms_dropout":
                self.comms_signal -= 15.0 * fault.magnitude
                self.comms_signal = max(0.0, self.comms_signal)  # signal can't go negative
            elif fault.fault_type == "tilt_spike":
                self.tilt_deg += 6.0 * fault.magnitude

            fault.ticks_remaining -= 1
            if fault.ticks_remaining > 0:
                still_active.append(fault)
        self.active_faults = still_active

    def _normal_drift(self):
        """Small random-walk noise so telemetry looks alive even with no faults."""
        if self.mode != "HIBERNATION":
            self.battery_pct -= random.uniform(0.01, 0.05)
        else:
            self.battery_pct -= random.uniform(0.001, 0.003)
        self.battery_pct = max(0.0, min(100.0, self.battery_pct))

        for wheel in self.motor_temp:
            pull_rate = 0.20 if self.mode == "COOL_DOWN" else 0.05
            baseline_pull = (35.0 - self.motor_temp[wheel]) * pull_rate
            self.motor_temp[wheel] += baseline_pull + random.uniform(-0.3, 0.3)

        self.tilt_deg += (0.0 - self.tilt_deg) * 0.1 + random.uniform(-0.5, 0.5)

        self.comms_signal += (95.0 - self.comms_signal) * 0.1 + random.uniform(-1, 1)
        self.comms_signal = max(0.0, min(100.0, self.comms_signal))

        # simple forward crawl, Week 2 will replace this with real path following
        self.position["x"] += random.uniform(0.05, 0.15)
        self.position["y"] += random.uniform(-0.05, 0.05)

    def next_reading(self) -> dict:
        self.tick += 1
        self._normal_drift()
        self._apply_faults()

        return {
            "tick": self.tick,
            "timestamp": time.time(),
            "battery_pct": round(self.battery_pct, 2),
            "motor_temp": {k: round(v, 2) for k, v in self.motor_temp.items()},
            "tilt_deg": round(self.tilt_deg, 2),
            "comms_signal": round(self.comms_signal, 2),
            "position": {k: round(v, 3) for k, v in self.position.items()},
            "active_fault_count": len(self.active_faults),
            "mode": self.mode,
        }