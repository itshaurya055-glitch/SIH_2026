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
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from telemetry import RoverTelemetry

app = FastAPI(title="Mission Copilot - Telemetry Service")

# Allow the frontend (running on a different port/origin during dev) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

rover = RoverTelemetry()

# Track connected websocket clients so the fault-injection endpoint
# doesn't need its own connection
connected_clients: list[WebSocket] = []


class FaultRequest(BaseModel):
    fault_type: str            # "motor_temp_spike" | "battery_drain" | "comms_dropout" | "tilt_spike"
    target: Optional[str] = "general"   # e.g. "front_left" for motor faults
    magnitude: Optional[float] = 1.0
    duration_ticks: Optional[int] = 20


@app.get("/api/status")
async def status():
    return {"status": "ok", "connected_clients": len(connected_clients)}


@app.post("/api/inject_fault")
async def inject_fault(req: FaultRequest):
    rover.inject_fault(
        fault_type=req.fault_type,
        target=req.target,
        magnitude=req.magnitude,
        duration_ticks=req.duration_ticks,
    )
    return {"ok": True, "message": f"Injected {req.fault_type} on {req.target}"}


@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            reading = rover.next_reading()
            await websocket.send_text(json.dumps(reading))
            await asyncio.sleep(1.0)  # 1 reading per second
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
