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

from collections import deque
from telemetry import RoverTelemetry
from dem_processor import DEMProcessor
from pathfinding import astar_path
from anomaly_detector import AnomalyDetector

app = FastAPI(title="Mission Copilot - Telemetry Service")

detector = AnomalyDetector()
recent_alerts = deque(maxlen=50)


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


@app.post("/api/inject_fault")
async def inject_fault(req: FaultRequest):
    rover.inject_fault(
        fault_type=req.fault_type,
        target=req.target,
        magnitude=req.magnitude,
        duration_ticks=req.duration_ticks,
    )
    return {"ok": True, "message": f"Injected {req.fault_type} on {req.target}"}


@app.get("/api/alerts")
async def get_alerts():
    return list(recent_alerts)


def _advance_along_path():
    """Move the rover one step along its active A* path (if any), and set
    its tilt telemetry to the REAL slope at that grid cell from the DEM -
    this is what makes tilt readings meaningful instead of random noise."""
    if rover.active_path and rover.path_index < len(rover.active_path) - 1:
        # If cool down is active, only advance every 3 seconds (slows down traversal)
        if rover.mode == "COOL_DOWN" and rover.tick % 3 != 0:
            pass
        # If hibernating, halt the rover completely
        elif rover.mode == "HIBERNATION":
            pass
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
            
            # Check for anomalies using the trained model
            alerts = detector.check(reading)
            reading["alerts"] = alerts
            
            # Determine operating mode based on detected anomalies (FDIR - Fault Detection, Isolation, and Recovery)
            has_motor_anomaly = any("motor" in a["sensor"] for a in alerts)
            has_battery_anomaly = any("battery" in a["sensor"] for a in alerts)
            
            if has_battery_anomaly:
                rover.mode = "HIBERNATION"
            elif has_motor_anomaly:
                rover.mode = "COOL_DOWN"
            else:
                # Recover back to NOMINAL mode when active faults clear
                if len(rover.active_faults) == 0:
                    rover.mode = "NOMINAL"
            
            # Ensure the reading payload contains the updated mode
            reading["mode"] = rover.mode
            
            # Store in backend history
            for alert in alerts:
                recent_alerts.appendleft(alert)
                
            await websocket.send_text(json.dumps(reading))
            await asyncio.sleep(1.0)  # 1 reading per second
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
