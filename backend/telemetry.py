"""
Synthetic rover telemetry generator.

This simulates what a real rover's sensor stream would look like:
- battery percentage (slowly drains, drops faster under strain)
- motor temperature per wheel (4 wheels)
- tilt (pitch/roll, driven by real DEM slope)
- comms signal strength (mostly stable, can be disrupted)
- wheel_slip_pct: derived from Bekker-Wong physics model + terrain noise
  (represents real regolith heterogeneity the physics model can't fully
  capture — the value the RNN learns to predict better over time)
- position (x, y)

Faults can be injected on demand (for live demo control) or left to
occur "naturally" via small random walk noise.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fault:
    """An active fault being simulated. Decays back to normal over time."""
    fault_type: str
    target: str = "general"
    magnitude: float = 1.0
    ticks_remaining: int = 20


class RoverTelemetry:
    def __init__(self):
        self.battery_pct = 95.0
        self.motor_temp = {
            "front_left":  35.0,
            "front_right": 35.0,
            "rear_left":   35.0,
            "rear_right":  35.0,
        }
        self.tilt_deg = 0.0
        self.comms_signal = 95.0
        self.position = {"x": 0.0, "y": 0.0}
        self.tick = 0
        self.active_faults: list[Fault] = []
        self.mode = "NOMINAL"       # "NOMINAL" | "COOL_DOWN" | "HIBERNATION" | "HAZARD_BYPASS"
        self.mode_override = None
        self.goal_pos = None
        self.replan_requested = False

        # Physics-driven slip (set externally by main.py after dem_processor runs)
        # Represents what wheel_slip_pct the physics model predicts for the current cell.
        self.physics_slip: float = 0.0

    def set_mode(self, mode: str):
        """Sets system mode dynamically (from AI agent or operator override)."""
        valid_modes = ["NOMINAL", "COOL_DOWN", "HIBERNATION", "HAZARD_BYPASS"]
        clean_mode = mode.upper().replace(" ", "_").replace("-", "_")
        if clean_mode in valid_modes:
            self.mode = clean_mode
            self.mode_override = clean_mode if clean_mode != "NOMINAL" else None

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
                self.comms_signal = max(0.0, self.comms_signal)
            elif fault.fault_type == "tilt_spike":
                self.tilt_deg += 6.0 * fault.magnitude

            fault.ticks_remaining -= 1
            if fault.ticks_remaining > 0:
                still_active.append(fault)
        self.active_faults = still_active

    def reset_battery(self):
        """Recharge battery back to nominal 95% and clear all faults/hibernation."""
        self.battery_pct = 95.0
        self.mode = "NOMINAL"
        self.mode_override = None
        self.active_faults = []

    def _normal_drift(self):
        """Small random-walk noise with active solar array power management."""
        has_drain_fault = any(f.fault_type == "battery_drain" for f in self.active_faults)

        if has_drain_fault:
            # Active fault -> rapid drain
            pass
        elif self.battery_pct < 80.0 or self.mode == "HIBERNATION":
            # Solar panel array active charge -> quickly restore to 95%
            self.battery_pct += random.uniform(1.5, 3.0)
            if self.battery_pct >= 90.0:
                self.mode = "NOMINAL"
                self.battery_pct = 95.0
        else:
            # Hover near nominal 95% with small fluctuations
            self.battery_pct += random.uniform(-0.1, 0.1)
            if self.battery_pct > 100.0:
                self.battery_pct = 95.0

        self.battery_pct = round(max(0.0, min(100.0, self.battery_pct)), 2)

        for wheel in self.motor_temp:
            pull_rate = 0.20 if self.mode == "COOL_DOWN" else 0.05
            baseline_pull = (35.0 - self.motor_temp[wheel]) * pull_rate
            self.motor_temp[wheel] += baseline_pull + random.uniform(-0.3, 0.3)

        self.tilt_deg += (0.0 - self.tilt_deg) * 0.1 + random.uniform(-0.5, 0.5)

        self.comms_signal += (95.0 - self.comms_signal) * 0.1 + random.uniform(-1, 1)
        self.comms_signal = max(0.0, min(100.0, self.comms_signal))

        self.position["x"] += random.uniform(0.05, 0.15)
        self.position["y"] += random.uniform(-0.05, 0.05)

    def _compute_wheel_slip_pct(self) -> float:
        """
        Compute the observed wheel slip percentage.

        Formula:
          observed_slip = physics_slip           (Bekker-Wong model baseline)
                        + terrain_noise          (unmodelled regolith variability)
                        + fault_contribution     (motor / tilt faults increase slip)

        This is the key sensor the RNN learns to predict better over time:
        the physics model gives physics_slip, but the unmodelled variability
        (terrain_noise) is what the GRU's residual correction targets.
        """
        # Base from physics model (0.0–1.0 slip ratio → 0–100%)
        base = self.physics_slip * 100.0

        # Unmodelled terrain heterogeneity: ±5% random, spatially correlated
        # via a small mean-reversion term so consecutive ticks aren't pure white noise
        terrain_noise = random.gauss(0, 3.5)

        # Fault contributions: motor faults → overheating → reduced traction
        fault_contribution = 0.0
        for fault in self.active_faults:
            if fault.fault_type == "motor_temp_spike":
                fault_contribution += 8.0 * fault.magnitude
            elif fault.fault_type == "tilt_spike":
                fault_contribution += 5.0 * fault.magnitude

        raw = base + terrain_noise + fault_contribution
        return round(float(max(0.0, min(100.0, raw))), 2)

    def next_reading(self) -> dict:
        self.tick += 1
        self._normal_drift()
        self._apply_faults()

        wheel_slip_pct = self._compute_wheel_slip_pct()

        return {
            "tick":              self.tick,
            "timestamp":         time.time(),
            "battery_pct":       round(self.battery_pct, 2),
            "motor_temp":        {k: round(v, 2) for k, v in self.motor_temp.items()},
            "tilt_deg":          round(self.tilt_deg, 2),
            "comms_signal":      round(self.comms_signal, 2),
            "position":          {k: round(v, 3) for k, v in self.position.items()},
            "active_fault_count": len(self.active_faults),
            "mode":              self.mode,
            "wheel_slip_pct":    wheel_slip_pct,       # NEW — physics + noise sensor
        }