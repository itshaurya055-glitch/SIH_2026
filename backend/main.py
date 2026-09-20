"""
Mission Copilot - Backend

Run with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Endpoints:
    WS   /ws/telemetry              -> live telemetry stream (1 reading/sec)
    POST /api/inject_fault          -> trigger a fault for demo purposes
    GET  /api/status                -> quick health check
    GET  /api/terrain               -> heightmap + physics overlays (soil, slip)
    POST /api/path                  -> compute A* path using physics cost grid
    GET  /api/path                  -> get active path
    GET  /api/terrain/physics       -> per-cell Bekker-Wong detail at (row, col)
    GET  /api/rnn/state             -> RNN learning metrics (before/after MAE)
    GET  /api/mission_log           -> recent anomaly log
    POST /api/agent/chat            -> LLM operator chat (reloaded)
    GET  /api/agent/decisions       -> agent decision history
"""

import asyncio
import json
import math
import os
import numpy as np
from collections import deque
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from telemetry import RoverTelemetry
from dem_processor import DEMProcessor
from pathfinding import astar_path
from anomaly_detector import AnomalyDetector
from agent import AnomalyAgent
from terrain_rnn import TerrainRNN

app = FastAPI(title="Mission Copilot - Telemetry Service")

# ── DEM setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEM_PATH = os.environ.get("DEM_PATH", os.path.join(PROJECT_ROOT, "data", "dem.tif"))
FALLBACK_DEM_PATH = os.path.join(PROJECT_ROOT, "test_data", "synthetic_crater_dem.tif")
GRID_SIZE = int(os.environ.get("GRID_SIZE", "128"))

active_dem_path = DEM_PATH if os.path.exists(DEM_PATH) else FALLBACK_DEM_PATH
dem = DEMProcessor(active_dem_path, grid_size=GRID_SIZE)
print(f"[startup] Loaded DEM from: {active_dem_path} (grid {GRID_SIZE}x{GRID_SIZE})")

# ── ML components ─────────────────────────────────────────────────────────────
anomaly_detector = AnomalyDetector()
terrain_rnn = TerrainRNN(lr=0.01, replay_buffer_size=200)

# ── State ─────────────────────────────────────────────────────────────────────
telemetry_history: list[dict] = []
MAX_TELEMETRY_HISTORY = 500
mission_log: list[dict] = []
MAX_LOG_ENTRIES = 200
is_currently_anomalous = False
connected_clients: list[WebSocket] = []

# ── Agent callbacks ───────────────────────────────────────────────────────────
def get_telemetry_window(start_tick: int, end_tick: int) -> list:
    return [t for t in telemetry_history if start_tick <= t["tick"] <= end_tick]

def get_terrain_info(row: int, col: int) -> dict:
    try:
        slope = dem.slope_at(row, col)
        raw_cost = float(dem.get_cost_grid()[row][col])
        cost = 999.0 if (np.isinf(raw_cost) or np.isnan(raw_cost)) else raw_cost
        slip  = dem.slip_at(row, col)
        soil  = dem.soil_type_at(row, col)
        return {"slope": float(slope), "cost": cost, "slip_ratio": slip, "soil_type": soil}
    except Exception as e:
        return {"error": str(e)}

def trigger_replan(reason: str) -> str:
    print(f"[main] Replan triggered by agent: {reason}")
    if rover.goal_pos is None:
        return "Failed: No active goal position set on rover."

    start = tuple(rover.grid_pos)
    goal  = tuple(rover.goal_pos)

    # Tighten slope limit for a safer emergency path
    dem._compute_cost_grid(max_traversable_slope=15.0)
    path = astar_path(dem.get_cost_grid(), start, goal)
    dem._compute_cost_grid()  # restore normal physics cost

    if path is not None:
        rover.active_path = [list(p) for p in path]
        rover.path_index = 0
        rover.replan_requested = True
        terrain_rnn.reset_hidden()  # new path → reset RNN hidden state
        return f"Success: Replanned path. New waypoints: {len(path)}"
    else:
        return "Failed: No safer path found."

def get_mission_log_callback() -> list:
    return list(mission_log)

def get_terrain_physics_callback(row: int = 64, col: int = 64) -> dict:
    detail = dem.get_cell_physics_detail(row, col)
    rnn_corr = terrain_rnn.predict([detail["traversal_cost"], detail["traversal_cost"], detail["slope_deg"], detail["slip_ratio"]])
    detail["rnn_correction"] = rnn_corr
    return detail

def get_rnn_state_callback() -> dict:
    return terrain_rnn.get_accuracy_metrics()

def set_system_mode_callback(mode: str) -> str:
    print(f"[main] System mode change requested: {mode}")
    rover.set_mode(mode)
    return f"Success: System mode updated to {rover.mode}"

agent = AnomalyAgent({
    "get_telemetry_window": get_telemetry_window,
    "get_terrain_info":     get_terrain_info,
    "get_terrain_physics":  get_terrain_physics_callback,
    "get_rnn_state":        get_rnn_state_callback,
    "trigger_replan":       trigger_replan,
    "set_system_mode":      set_system_mode_callback,
    "get_mission_log":      get_mission_log_callback,
})

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rover state ───────────────────────────────────────────────────────────────
rover = RoverTelemetry()
rover.grid_pos  = [GRID_SIZE // 4, GRID_SIZE // 4]
rover.active_path = []
rover.path_index  = 0

# ── Pydantic models ───────────────────────────────────────────────────────────
class FaultRequest(BaseModel):
    fault_type: str
    target: Optional[str] = "general"
    magnitude: Optional[float] = 1.0
    duration_ticks: Optional[int] = 20

class ModeRequest(BaseModel):
    mode: str

class PathRequest(BaseModel):
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int

class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = []

# ── HTTP Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return {
        "status": "ok",
        "connected_clients": len(connected_clients),
        "dem_source": active_dem_path,
        "grid_size": GRID_SIZE,
        "agent_available": agent.client is not None,
        "rnn_steps": terrain_rnn._step,
    }


@app.get("/api/terrain")
async def get_terrain():
    """Heightmap + soil_map + slip_map for the frontend renderer and overlays."""
    return dem.get_heightmap_payload()


@app.get("/api/terrain/physics")
async def get_terrain_physics(row: int = 64, col: int = 64):
    """
    Full Bekker-Wong breakdown for a single grid cell.
    Useful for the frontend info panel and agent diagnostics.
    """
    detail = dem.get_cell_physics_detail(row, col)
    rnn_correction = terrain_rnn.predict([
        detail["traversal_cost"],
        detail["traversal_cost"],  # no observed cost yet for this query
        detail["slope_deg"],
        detail["slip_ratio"],
    ])
    detail["rnn_correction"] = round(rnn_correction, 4)
    detail["rnn_corrected_cost"] = round(
        max(1.0, detail["traversal_cost"] + rnn_correction)
        if detail["is_traversable"] else 999.0, 4
    )
    return detail


@app.get("/api/rnn/state")
async def get_rnn_state():
    """
    Returns the GRU's learning metrics — the 'before vs after' accuracy
    story: how much lower is the prediction error now vs. the cold start?
    """
    metrics = terrain_rnn.get_accuracy_metrics()
    return {
        "model": "GRU (NumPy, online learning)",
        "input_features": ["physics_cost", "observed_cost_proxy", "slope_deg", "slip_ratio"],
        "hidden_size": terrain_rnn.HIDDEN_SIZE,
        **metrics,
    }


@app.post("/api/path")
async def compute_path(req: PathRequest):
    """
    A* over the physics-based cost grid.
    Optionally blends RNN correction into cost before pathfinding.
    """
    start = (req.start_row, req.start_col)
    goal  = (req.goal_row,  req.goal_col)

    path = astar_path(dem.get_cost_grid(), start, goal)
    if path is None:
        raise HTTPException(
            status_code=422,
            detail="No traversable path found — goal may be unreachable given slope/soil limits.",
        )

    rover.active_path = [list(p) for p in path]
    rover.path_index  = 0
    rover.grid_pos    = list(start)
    rover.goal_pos    = list(goal)
    rover.replan_requested = False
    terrain_rnn.reset_hidden()   # new path → fresh GRU sequence

    return {"waypoint_count": len(path), "path": rover.active_path}


@app.get("/api/path")
async def get_active_path():
    return {"path": rover.active_path, "waypoint_count": len(rover.active_path)}


@app.get("/api/mission_log")
async def get_mission_log(limit: int = 50):
    return {"entries": mission_log[-limit:][::-1]}


@app.post("/api/inject_fault")
async def inject_fault(req: FaultRequest):
    rover.inject_fault(
        fault_type=req.fault_type,
        target=req.target,
        magnitude=req.magnitude,
        duration_ticks=req.duration_ticks,
    )
    return {"ok": True, "message": f"Injected {req.fault_type} on {req.target}"}


@app.post("/api/mode")
async def set_system_mode_api(req: ModeRequest):
    rover.set_mode(req.mode)
    return {"ok": True, "mode": rover.mode}


@app.post("/api/reset_battery")
async def reset_battery():
    rover.reset_battery()
    return {"ok": True, "message": "Battery recharged to 95.0% and system mode reset to NOMINAL"}


@app.post("/api/agent/chat")
async def agent_chat(req: ChatRequest):
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(agent.run_chat_agent, req.message, req.history),
            timeout=4.0
        )
        return result
    except Exception as e:
        print(f"[main] Chat agent timeout or error: {e}, returning instant fallback")
        mission_log_data = agent.callbacks["get_mission_log"]()
        telemetry_data = agent.callbacks["get_telemetry_window"](0, 1000)
        rnn_state = agent.callbacks["get_rnn_state"]() if "get_rnn_state" in agent.callbacks else {}
        resp = agent._run_mock_chat(req.message, mission_log_data, telemetry_data, rnn_state)
        return {
            "response": resp,
            "tools_used": ["get_mission_log", "get_telemetry_window", "get_rnn_state"],
            "tools_called": ["get_mission_log", "get_telemetry_window", "get_rnn_state"]
        }


@app.get("/api/agent/decisions")
async def get_agent_decisions():
    from agent import agent_decision_log
    return {"decisions": agent_decision_log}

# ── Telemetry loop helpers ────────────────────────────────────────────────────

# Running counter for periodic RNN replay
_ticks_since_replay = 0
REPLAY_EVERY_N_TICKS = 50

def _advance_along_path():
    """
    Move rover one step, update slip from physics, feed RNN online update.
    """
    global _ticks_since_replay

    if rover.active_path and rover.path_index < len(rover.active_path) - 1:
        if rover.mode == "COOL_DOWN" and rover.tick % 3 != 0:
            pass   # slow to 1/3 speed
        elif rover.mode == "HIBERNATION":
            pass   # fully halted
        else:
            rover.path_index += 1
            row, col = rover.active_path[rover.path_index]
            rover.grid_pos = [row, col]

    row, col = rover.grid_pos

    # ── Physics inputs for this cell ──────────────────────────────────────
    real_slope   = dem.slope_at(row, col)
    physics_slip = dem.slip_at(row, col)
    physics_cost = float(dem.get_cost_grid()[row][col])

    # Push physics slip into rover so telemetry can compute observed slip
    rover.physics_slip = physics_slip

    # Tilt driven by DEM slope (with tiny chirp for liveness)
    rover.tilt_deg = real_slope + ((-1) ** (row + col)) * 0.5

    # ── RNN: compute correction before the step ────────────────────────────
    rnn_features = [physics_cost, physics_cost, real_slope, physics_slip]
    rnn_correction = terrain_rnn.predict(rnn_features)

    # ── RNN online update (after step — we observe the "actual" cost proxy)─
    # Observed cost proxy: normalised motor temp delta + slip deviation.
    # This is the "what actually happened" that the physics model didn't
    # fully predict — the GRU's residual learning target.
    avg_motor = sum(rover.motor_temp.values()) / 4.0
    # Motor temp above baseline (~35°C) signals more work → higher cost
    motor_overhead = max(0.0, avg_motor - 35.0) / 10.0   # 0 at idle, ~0.5 at 40°C
    observed_cost_proxy = physics_cost + motor_overhead

    residual = observed_cost_proxy - physics_cost   # what the physics missed
    terrain_rnn.update(rnn_features, residual)

    _ticks_since_replay += 1
    if _ticks_since_replay >= REPLAY_EVERY_N_TICKS:
        terrain_rnn.replay_update(n_steps=32)
        _ticks_since_replay = 0
        metrics = terrain_rnn.get_accuracy_metrics()
        print(f"[rnn] Replay done. Steps={metrics['steps']} | "
              f"MAE before={metrics['before_mae']} → after={metrics['after_mae']} "
              f"| improvement={metrics['improvement_pct']}%")

    return rnn_correction, physics_slip


async def run_agent_diagnostics(reading: dict, anomaly_result: dict):
    top_feature = anomaly_result["top_contributing_features"][0] \
        if anomaly_result["top_contributing_features"] else None
    detail = f"{top_feature['feature']} (z={top_feature['z_score']})" \
        if top_feature else "unspecified"
    alert_msg = (f"Tick {reading['tick']}: anomaly detected via "
                 f"{anomaly_result['detected_by']} — {detail}")

    anomaly_details = {
        "message":       alert_msg,
        "sensor":        top_feature["feature"] if top_feature else "general",
        "tick":          reading["tick"],
        "anomaly_score": anomaly_result["anomaly_score"],
        "max_z_score":   anomaly_result["max_z_score"],
    }
    await asyncio.to_thread(agent.run_autonomous_agent, anomaly_details)

# ── WebSocket telemetry ───────────────────────────────────────────────────────

@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            rnn_correction, physics_slip = _advance_along_path()
            reading = rover.next_reading()

            reading["grid_pos"] = rover.grid_pos
            reading["path_progress"] = {
                "index": rover.path_index,
                "total": len(rover.active_path),
            }
            reading["physics_slip"] = round(physics_slip, 4)
            reading["rnn_correction"] = round(rnn_correction, 4)
            reading["soil_type"] = dem.soil_type_at(*rover.grid_pos)

            anomaly_result = anomaly_detector.detect(reading)
            reading["anomaly"] = anomaly_result

            # FDIR (Fault Detection, Isolation, and Recovery)
            has_battery_fault = any(f.fault_type == "battery_drain" for f in rover.active_faults)
            has_motor_fault   = any(f.fault_type == "motor_temp_spike" for f in rover.active_faults)
            max_motor_temp    = max(rover.motor_temp.values()) if rover.motor_temp else 35.0

            if has_battery_fault or rover.battery_pct < 20.0:
                rover.mode = "HIBERNATION"
            elif has_motor_fault or max_motor_temp > 55.0:
                rover.mode = "COOL_DOWN"
            elif abs(rover.tilt_deg) > 25.0:
                rover.mode = "HAZARD_BYPASS"
            elif rover.mode_override:
                rover.mode = rover.mode_override
            else:
                rover.mode = "NOMINAL"

            reading["mode"] = rover.mode
            reading["replan_requested"] = rover.replan_requested

            # History
            telemetry_history.append(reading.copy())
            if len(telemetry_history) > MAX_TELEMETRY_HISTORY:
                telemetry_history.pop(0)

            global is_currently_anomalous
            if anomaly_result["is_anomaly"]:
                log_entry = {
                    "tick":      reading["tick"],
                    "timestamp": reading["timestamp"],
                    "anomaly_score":   anomaly_result["anomaly_score"],
                    "detected_by":     anomaly_result["detected_by"],
                    "top_contributing_features": anomaly_result["top_contributing_features"],
                    "mode":     rover.mode,
                    "snapshot": {
                        "battery_pct":   reading["battery_pct"],
                        "motor_temp":    reading["motor_temp"],
                        "tilt_deg":      reading["tilt_deg"],
                        "comms_signal":  reading["comms_signal"],
                        "wheel_slip_pct": reading["wheel_slip_pct"],
                    },
                }
                mission_log.append(log_entry)
                if len(mission_log) > MAX_LOG_ENTRIES:
                    mission_log.pop(0)

                if not is_currently_anomalous:
                    is_currently_anomalous = True
                    asyncio.create_task(run_agent_diagnostics(reading, anomaly_result))
            else:
                is_currently_anomalous = False

            await websocket.send_text(json.dumps(reading))

            if rover.replan_requested:
                rover.replan_requested = False

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        connected_clients.remove(websocket)