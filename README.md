# Mission Copilot — Week 1

Foundational telemetry pipeline: backend generates synthetic rover telemetry
and streams it over WebSocket; frontend visualizes it live with fault-injection
buttons for demo control.

## What's included
- `backend/telemetry.py` — synthetic telemetry generator + fault injection logic
- `backend/main.py` — FastAPI app: WebSocket stream + REST endpoint to inject faults
- `frontend/index.html` — single-file dashboard (no build step), live charts via Chart.js

## Run it

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Check it's alive: open `http://localhost:8000/api/status` — should show `{"status":"ok", ...}`

### 2. Frontend
Just open `frontend/index.html` directly in your browser (double-click it, or
use VSCode's "Live Server" extension). No npm install needed for Week 1.

You should see 4 live-updating charts (battery, motor temp x4, tilt, comms signal)
within a couple seconds of the backend starting.

### 3. Try fault injection
Click any of the buttons under "Demo Controls" — you'll see the corresponding
chart spike within 1-2 seconds, and an alert appear in the feed below.

## What's next (Week 2)
- Replace the flat frontend with a Three.js 3D scene (terrain + rover model)
- Add real A* pathfinding over a heightmap, tie tilt telemetry to actual terrain slope
- Backend sends waypoints, frontend animates the rover moving through them

## Notes for the team
- `telemetry.py` is intentionally simple/readable — if ML folks want to plug in
  more realistic fault signatures (e.g. gradual drift vs sudden spike vs oscillation),
  extend `_apply_faults()`.
- The WebSocket sends 1 reading/sec. If you want the anomaly detector (Week 3) to
  have more data to work with, lower `asyncio.sleep(1.0)` in `main.py` — just make
  sure the frontend chart buffer (`MAX_POINTS` in `index.html`) still makes sense
  at the new rate.
- CORS is wide open (`allow_origins=["*"]`) for dev convenience — fine for a
  hackathon demo, don't ship this configuration to anything public-facing.
