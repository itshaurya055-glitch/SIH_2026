"""
Mission Copilot - Week 1 backend

Run with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Endpoints:
    WS   /ws/telemetry        -> live telemetry stream (1 reading/sec)
    POST /api/inject_fault    -> trigger a fault for demo purposes
    GET  /api/status          -> quick health check
"""

import asyncio
import json
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from telemetry import RoverTelemetry
from dem_processor import DEMProcessor
from pathfinding import astar_path
from anomaly_detector import AnomalyDetector

app = FastAPI(title="Mission Copilot - Telemetry Service")

# Path to the DEM file. In Docker this is mounted from ./data on the host.
# Falls back to the synthetic test DEM if no real file is present, so the
# service still boots cleanly for anyone who hasn't downloaded the real
# USGS/NASA data yet.
DEM_PATH = os.environ.get("DEM_PATH", "data/dem.tif")
FALLBACK_DEM_PATH = "test_data/synthetic_crater_dem.tif"
GRID_SIZE = int(os.environ.get("GRID_SIZE", "128"))

active_dem_path = DEM_PATH if os.path.exists(DEM_PATH) else FALLBACK_DEM_PATH
dem = DEMProcessor(active_dem_path, grid_size=GRID_SIZE)
print(f"[startup] Loaded DEM from: {active_dem_path} (grid {GRID_SIZE}x{GRID_SIZE})")

anomaly_detector = AnomalyDetector()  # trains on first run, loads cached model after

# Allow the frontend (running on a different port/origin during dev) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

rover = RoverTelemetry()
rover.grid_pos = [GRID_SIZE // 4, GRID_SIZE // 4]  # rover's current position on the terrain grid
rover.active_path = []      # list of [row, col] waypoints currently being followed
rover.path_index = 0        # progress along active_path

# Track connected websocket clients so the fault-injection endpoint
# doesn't need its own connection
connected_clients: list[WebSocket] = []

# Simple in-memory log of detected anomalies. This is what the Week 4 AI
# agent will query ("what happened at 14:02?") instead of hallucinating -
# every alert here is a real detection event, not a guess.
mission_log: list[dict] = []
MAX_LOG_ENTRIES = 200


class FaultRequest(BaseModel):
    fault_type: str            # "motor_temp_spike" | "battery_drain" | "comms_dropout" | "tilt_spike"
    target: Optional[str] = "general"   # e.g. "front_left" for motor faults
    magnitude: Optional[float] = 1.0
    duration_ticks: Optional[int] = 20


class PathRequest(BaseModel):
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int


@app.get("/api/status")
async def status():
    return {
        "status": "ok",
        "connected_clients": len(connected_clients),
        "dem_source": active_dem_path,
        "grid_size": GRID_SIZE,
    }


@app.get("/api/terrain")
async def get_terrain():
    """Returns the processed heightmap for the frontend to render as a
    Three.js terrain mesh."""
    return dem.get_heightmap_payload()


@app.post("/api/path")
async def compute_path(req: PathRequest):
    """Runs A* over the DEM-derived cost grid and, if found, sets it as the
    rover's active path so the telemetry loop will animate along it."""
    start = (req.start_row, req.start_col)
    goal = (req.goal_row, req.goal_col)

    path = astar_path(dem.get_cost_grid(), start, goal)
    if path is None:
        raise HTTPException(
            status_code=422,
            detail="No traversable path found - goal may be unreachable given slope limits."
        )

    rover.active_path = [list(p) for p in path]
    rover.path_index = 0
    rover.grid_pos = list(start)

    return {"waypoint_count": len(path), "path": rover.active_path}


@app.get("/api/mission_log")
async def get_mission_log(limit: int = 50):
    """Recent detected anomalies, most recent first. Used by the dashboard's
    alert feed and (Week 4) the AI agent's explanation tool."""
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


def _advance_along_path():
    """Move the rover one step along its active A* path (if any), and set
    its tilt telemetry to the REAL slope at that grid cell from the DEM -
    this is what makes tilt readings meaningful instead of random noise."""
    if rover.active_path and rover.path_index < len(rover.active_path) - 1:
        if rover.mode == "COOL_DOWN" and rover.tick % 3 != 0:
            pass  # slow to 1/3 speed
        elif rover.mode == "HIBERNATION":
            pass  # halt completely
        else:
            rover.path_index += 1
            row, col = rover.active_path[rover.path_index]
            rover.grid_pos = [row, col]

    row, col = rover.grid_pos
    real_slope = dem.slope_at(row, col)
    # blend real terrain slope with the existing small noise so it still
    # feels alive, but is now grounded in actual DEM data
    rover.tilt_deg = real_slope + ((-1) ** (row + col)) * 0.5


@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            _advance_along_path()
            reading = rover.next_reading()
            reading["grid_pos"] = rover.grid_pos
            reading["path_progress"] = {
                "index": rover.path_index,
                "total": len(rover.active_path),
            }

            anomaly_result = anomaly_detector.detect(reading)
            reading["anomaly"] = anomaly_result

            # FDIR: set rover mode based on what the detector found
            if anomaly_result["is_anomaly"]:
                top_features = [f["feature"] for f in anomaly_result["top_contributing_features"]]
                has_battery = any("battery" in f for f in top_features)
                has_motor = any("motor" in f for f in top_features)

                if has_battery:
                    rover.mode = "HIBERNATION"
                elif has_motor:
                    rover.mode = "COOL_DOWN"
            else:
                if len(rover.active_faults) == 0:
                    rover.mode = "NOMINAL"

            reading["mode"] = rover.mode

            if anomaly_result["is_anomaly"]:
                log_entry = {
                    "tick": reading["tick"],
                    "timestamp": reading["timestamp"],
                    "anomaly_score": anomaly_result["anomaly_score"],
                    "detected_by": anomaly_result["detected_by"],
                    "top_contributing_features": anomaly_result["top_contributing_features"],
                    "mode": rover.mode,
                    "snapshot": {
                        "battery_pct": reading["battery_pct"],
                        "motor_temp": reading["motor_temp"],
                        "tilt_deg": reading["tilt_deg"],
                        "comms_signal": reading["comms_signal"],
                    },
                }
                mission_log.append(log_entry)
                if len(mission_log) > MAX_LOG_ENTRIES:
                    mission_log.pop(0)

            await websocket.send_text(json.dumps(reading))
            await asyncio.sleep(1.0)  # 1 reading per second
    except WebSocketDisconnect:
        connected_clients.remove(websocket)