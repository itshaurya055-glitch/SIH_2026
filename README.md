# Mission Copilot — Week 2

Rover telemetry pipeline (Week 1) + real DEM terrain, A* path planning,
and a Three.js 3D view where the rover walks its planned route with tilt
telemetry driven by actual terrain slope. Now Dockerized.

## What's included
- `backend/telemetry.py` — synthetic sensor generator + fault injection logic
- `backend/dem_processor.py` — loads a real GeoTIFF DEM, produces a heightmap
  grid + slope-based traversal cost grid
- `backend/pathfinding.py` — A* over the DEM cost grid (steep slopes = high
  cost or fully blocked)
- `backend/main.py` — FastAPI app: WebSocket telemetry stream, `/api/terrain`,
  `/api/path`, `/api/inject_fault`
- `frontend/index.html` — Week 1 chart dashboard (still works standalone)
- `frontend/terrain.html` — Week 2: Three.js 3D terrain, live rover marker,
  double-click anywhere on terrain to set a new goal and re-path
- `check_dem.py` — validates a downloaded DEM file before dropping it into Docker
- `docker-compose.yml` + Dockerfiles for both services

## Step 1 — Get a real DEM file

Recommended source: **USGS Astrogeology Moon LRO LOLA DEM catalog**
(https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m) — the
standard lunar elevation dataset derived from NASA's Lunar Reconnaissance
Orbiter laser altimeter.

For a hackathon, a **regional crater DEM** is much easier to work with than
the full global file (which is huge). Search the Astropedia catalog
(https://astrogeology.usgs.gov/search) for terms like "Moon LRO NAC DEM
[crater name]" — e.g. the Aitken Crater NAC DEM
(https://astrogeology.usgs.gov/search/map/Moon/LMMP/AitkenCrater/Moon_LRO_NAC_DEM_Aitken_Crater_17S173E_150cmp)
is a good size for this. Download the `.tif` file.

(Mars alternative: search USGS Astrogeology for "MOLA" DEM data if you'd
rather do a Mars rover instead of lunar.)

Once downloaded, validate it works with our pipeline:
```bash
python3 check_dem.py /path/to/your/downloaded_dem.tif
```
This should print elevation range, slope range, and confirm the file loads
correctly — fix any issues here before touching Docker.

Then place it at:
```
mission-copilot/data/dem.tif
```
(create the `data/` folder if it doesn't exist)

**No real DEM yet?** No problem — the backend automatically falls back to a
bundled synthetic crater DEM (`backend/test_data/synthetic_crater_dem.tif`)
so everything still runs. Swap in the real file whenever it's ready; no code
changes needed.

## Step 2 — Run with Docker

```bash
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend dashboard (Week 1 charts): `http://localhost:8080/index.html`
- Terrain view (Week 2): `http://localhost:8080/terrain.html`

First build takes a bit longer (installing GDAL for rasterio). Subsequent
`docker compose up` runs are fast.

## Step 3 — Try it out

1. Open `http://localhost:8080/terrain.html`
2. You'll see the real terrain rendered as a 3D mesh, with the rover (blue cone) sitting on it
3. Double-click anywhere on the terrain to set a goal — the backend runs A*
   and returns a path (green line) that automatically routes around anything
   too steep to climb
4. The rover animates along that path in real time, and the HUD's tilt
   reading updates based on the actual DEM slope at its current position —
   not random noise
5. Open `http://localhost:8080/index.html` in another tab to see the Week 1
   chart dashboard still working off the same telemetry stream, including
   fault injection buttons

## Running without Docker (for quick local iteration)
```bash
cd backend
pip install -r requirements.txt
DEM_PATH=../data/dem.tif uvicorn main:app --reload --port 8000
```
Then just open `frontend/terrain.html` or `frontend/index.html` directly in
a browser (no build step for either).

## What's next (Week 3)
- Anomaly detection (autoencoder or isolation forest) on the live telemetry stream
- Replace the naive threshold check in `index.html` with a real trained model
- Alerts should fire automatically when injected faults occur, without hardcoded thresholds

## Notes for the team
- `dem_processor.py`'s `max_traversable_slope` (default 25°) controls how
  aggressive the rover's climbing limit is — tune this to make demos more
  or less dramatic (lower = more of the terrain becomes "blocked", more
  interesting-looking reroutes).
- `GRID_SIZE` env var controls heightmap resolution. 128 is a good balance;
  go higher only if your DEM has enough real detail to justify it — a
  low-res DEM at high grid size just interpolates flat, unconvincing terrain.
- The rover currently moves exactly one grid cell per telemetry tick (1/sec).
  If demo pacing feels too slow/fast, adjust in `main.py`'s `_advance_along_path()`
  and the `asyncio.sleep()` interval together.
- CORS is wide open (`allow_origins=["*"]`) for dev/demo convenience — don't
  ship this configuration anywhere public-facing.