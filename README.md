# 🚀 Mission Copilot — SIH 2026

> **AI-Powered Lunar Rover Autonomous Navigation & Telemetry Platform**  
> Built for Smart India Hackathon 2026

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Three.js](https://img.shields.io/badge/Three.js-r185-000000?logo=three.js&logoColor=white)](https://threejs.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [DEM Data (Moon Terrain)](#-dem-data-moon-terrain)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Frontend Controls](#-frontend-controls)
- [Tech Stack](#-tech-stack)
- [Team Notes & Configuration](#-team-notes--configuration)

---

## 🌕 Overview

**Mission Copilot** is a full-stack simulation and monitoring platform for a lunar rover. It combines:

- **Real NASA DEM terrain data** (from the Lunar Reconnaissance Orbiter) rendered as an interactive 3D mesh
- **Bekker-Wong terramechanics physics** to model how a rover wheel behaves on lunar regolith (dust/soil)
- **A\* pathfinding** that routes the rover around slopes too steep to safely climb
- **Online GRU Neural Network** that learns terrain traversal cost corrections in real-time during the mission
- **Random Forest anomaly detection** on live telemetry for fault detection
- **NVIDIA NIM / LLaMA-3 AI Copilot** that reasons autonomously when faults are detected and takes corrective action (replan, mode change, hibernate)

The system is designed for a **hackathon demo** setting — everything runs locally, falls back gracefully without a real DEM, and exposes clean REST + WebSocket APIs.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌍 **Real Lunar DEM** | Loads NASA LRO LOLA GeoTIFF elevation data, resampled to a 128×128 physics grid |
| ⚛️ **Bekker-Wong Physics** | Accurate lunar soil pressure-sinkage, slip ratio, and sinkage depth calculations |
| 🗺️ **A\* Pathfinding** | Avoids steep/impassable slopes; re-routes in real-time when hazards are detected |
| 🧠 **Online GRU RNN** | Zero-dependency NumPy GRU that learns residual traversal cost corrections live |
| 🤖 **AI Agent (NVIDIA NIM)** | LLaMA-3.3-70B agent that autonomously analyzes anomalies and triggers replans |
| 📡 **Live Telemetry** | WebSocket stream: battery, motor temps (4 wheels), tilt, wheel slip, comms signal |
| 🔴 **Fault Injection** | One-click inject motor overheat, battery drain, comms dropout for live demos |
| 🎨 **3D Visualization** | Three.js interactive terrain with rover animation, slope heatmaps, traversability overlays |
| 📊 **Dashboard** | React + Recharts live chart dashboard with anomaly alerts and mission log |
| 🛡️ **FDIR System** | Automated Fault Detection, Isolation & Recovery (mode switching + path replanning) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MISSION COPILOT SYSTEM                          │
├──────────────────────────┬──────────────────────────────────────────────┤
│      BACKEND (Python)    │           FRONTEND (React + Three.js)        │
│                          │                                              │
│  ┌────────────────────┐  │   ┌──────────────────┐  ┌─────────────────┐ │
│  │  DEM Processor     │  │   │  terrain.html    │  │  index.html     │ │
│  │  (GeoTIFF→Grid)    │──┼──▶│  3D Terrain View │  │  Telemetry HUD  │ │
│  └────────────────────┘  │   │  Rover Animation │  │  Chart Dashboard│ │
│           │              │   │  Overlay Modes   │  │  Fault Inject   │ │
│           ▼              │   └──────────────────┘  └─────────────────┘ │
│  ┌────────────────────┐  │                │                │            │
│  │  Bekker-Wong       │  │                └────────────────┘            │
│  │  Physics Engine    │  │                         │                    │
│  └────────────────────┘  │              WebSocket + REST API            │
│           │              │                         │                    │
│           ▼              │   ┌─────────────────────┴──────────────────┐ │
│  ┌────────────────────┐  │   │        FastAPI Backend (main.py)        │ │
│  │  A* Pathfinder     │◀─┼───│  /ws/telemetry  /api/terrain           │ │
│  └────────────────────┘  │   │  /api/path      /api/inject_fault      │ │
│           │              │   │  /api/agent/chat  /api/rnn/state        │ │
│           ▼              │   └────────────────────────────────────────┘ │
│  ┌────────────────────┐  │                         │                    │
│  │  Online GRU RNN    │  │              ┌──────────┴─────────┐          │
│  │  (terrain_rnn.py)  │  │              │                    │          │
│  └────────────────────┘  │   ┌──────────────────┐  ┌─────────────────┐ │
│           │              │   │  Anomaly Detector │  │  AI Agent       │ │
│           ▼              │   │  (Random Forest)  │  │  (NVIDIA NIM /  │ │
│  ┌────────────────────┐  │   └──────────────────┘  │  LLaMA-3.3-70B) │ │
│  │  Telemetry Gen     │  │                          └─────────────────┘ │
│  │  (telemetry.py)    │  │                                              │
│  └────────────────────┘  │                                              │
└──────────────────────────┴──────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SIH_2026/
│
├── 📄 README.md                          ← You are here
├── 📄 SYSTEM_ARCHITECTURE_AND_MODELS.md  ← Deep dive: physics, math, ML models
├── 📄 check_dem.py                       ← Validate a downloaded DEM file
├── 📄 .env                               ← API keys (NVIDIA NIM key here)
├── 📄 .gitignore
│
├── 📁 backend/                           ← Python FastAPI backend
│   ├── main.py                           ← FastAPI app & WebSocket telemetry loop
│   ├── telemetry.py                      ← Rover sensor simulator + fault injection
│   ├── dem_processor.py                  ← GeoTIFF DEM loader + slope/cost grid
│   ├── pathfinding.py                    ← A* over physics cost grid
│   ├── lunar_physics.py                  ← Bekker-Wong terramechanics engine
│   ├── terrain_rnn.py                    ← Online GRU RNN (pure NumPy)
│   ├── anomaly_detector.py               ← Random Forest anomaly classifier
│   ├── agent.py                          ← NVIDIA NIM / LLaMA-3 AI Copilot agent
│   ├── generate_test_dem.py              ← Generates synthetic fallback DEM
│   └── requirements.txt                  ← Python dependencies
│
├── 📁 frontend/                          ← React + Vite frontend
│   ├── index.html                        ← Telemetry chart dashboard
│   ├── terrain.html                      ← 3D Three.js terrain viewer
│   ├── src/
│   │   ├── App.jsx                       ← Main React dashboard app
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── 📁 data/                              ← Place your DEM file here
│   └── dem.tif                           ← (Not committed) Your real lunar DEM
│
└── 📁 ml/                                ← ML documentation and notes
    └── README.md
```

---

## 🌕 DEM Data (Moon Terrain)

This project uses **real NASA lunar elevation data** (Digital Elevation Model / DEM) from the **Lunar Reconnaissance Orbiter (LRO) LOLA** instrument. You must download a DEM file to visualize real lunar terrain.

### 🔗 Download the DEM

**Primary Source (Recommended):**  
👉 **[Moon LRO LOLA DEM 118m/pixel — USGS Astrogeology](https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m#open)**

This is the global lunar DEM from NASA's LRO LOLA laser altimeter — the most accurate publicly available lunar elevation dataset.

> ⚠️ **The global file is very large (~4GB).** For a hackathon demo, it is recommended to use a **regional crater DEM** instead.

### 🌑 Recommended Regional DEMs (Smaller, Faster)

Search the **[USGS Astropedia Catalog](https://astrogeology.usgs.gov/search)** for:
- **"Moon LRO NAC DEM Aitken Crater"** — good demo size  
  Direct link: [Aitken Crater NAC DEM](https://astrogeology.usgs.gov/search/map/Moon/LMMP/AitkenCrater/Moon_LRO_NAC_DEM_Aitken_Crater_17S173E_150cmp)
- **"Moon LRO NAC DEM Shackleton Crater"** — South Pole region
- **"Moon LRO NAC DEM Tycho Crater"** — high-relief, great for demo

Download the **`.tif`** file from any of these pages.

### ✅ Validate the DEM

Before running the backend, validate that your DEM file is compatible:

```bash
python check_dem.py path/to/your/downloaded_dem.tif
```

Expected output:
```
✅ DEM loaded successfully
   Elevation range: -2134.0 m to 1890.5 m
   Slope range:     0.0° to 42.3°
   Grid size:       128 × 128 resampled
```

Fix any issues reported here before continuing.

### 📂 Place the DEM File

Once validated, copy the `.tif` file to:

```
data/dem.tif
```

**No DEM file? No problem!**  
The backend **automatically falls back** to a built-in synthetic crater DEM (`backend/test_data/synthetic_crater_dem.tif`) — everything still runs. Just swap in the real file when ready; no code changes needed.

---

## 🛠️ Installation & Setup

### Prerequisites

| Tool | Version | Download |
|---|---|---|
| Python | 3.10+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| Git | Any | [git-scm.com](https://git-scm.com) |

### 1. Clone the Repository

```bash
git clone https://github.com/itshaurya055-glitch/SIH_2026.git
cd SIH_2026
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

**requirements.txt includes:**
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
websockets==12.0
pydantic==2.7.4
numpy
rasterio          ← reads GeoTIFF DEM files
scikit-learn      ← Random Forest anomaly detector
scipy
joblib
openai            ← NVIDIA NIM / OpenAI API client
python-dotenv
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Configure API Keys (Optional)

The AI Copilot agent uses **NVIDIA NIM** (free tier available). Without a key, it falls back to a built-in rule-based decision engine.

Create a `.env` file in the project root:

```env
# .env — do NOT commit this file
NVIDIA_API_KEY=nvapi-your-key-here
```

Get a free NVIDIA NIM API key at: [https://build.nvidia.com](https://build.nvidia.com)

---

## ▶️ Running the Application

### Step 1: Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**On Windows PowerShell (with custom DEM path):**
```powershell
$env:DEM_PATH="../data/dem.tif"; uvicorn main:app --reload --port 8000
```

**On Windows PowerShell (with custom grid size):**
```powershell
$env:GRID_SIZE="256"; uvicorn main:app --reload --port 8000
```

Backend will be live at:
- **API**: `http://localhost:8000`
- **Health Check**: `http://localhost:8000/api/status`
- **Interactive API Docs**: `http://localhost:8000/docs`

### Step 2: Start the Frontend

```bash
cd frontend
npm run dev
```

Frontend dev server will start at: `http://localhost:5173`

### Step 3: Open the Dashboards

| Dashboard | URL | Description |
|---|---|---|
| **3D Terrain Viewer** | `frontend/terrain.html` | Open directly in browser |
| **Telemetry Dashboard** | `http://localhost:5173` | React + Vite dev server |
| **Legacy Dashboard** | `frontend/index.html` | Open directly in browser |

> **Tip:** Open `terrain.html` directly in Chrome/Firefox — it talks directly to the FastAPI backend on port 8000. No Vite server needed for it.

---

## 📡 API Reference

### WebSocket

| Endpoint | Description |
|---|---|
| `ws://localhost:8000/ws/telemetry` | Live telemetry stream (1 reading/second) |

**Sample telemetry message:**
```json
{
  "tick": 42,
  "timestamp": "2026-09-20T14:30:00Z",
  "battery_pct": 87.3,
  "motor_temp": { "front_left": 38.2, "front_right": 36.1, "rear_left": 37.8, "rear_right": 35.9 },
  "chassis_pitch": 3.4,
  "chassis_roll": 1.2,
  "tilt_deg": 4.1,
  "wheel_slip_pct": 18.5,
  "comms_signal": 72.0,
  "grid_pos": [32, 64],
  "soil_type": "MEDIUM",
  "rnn_correction": 0.012,
  "anomaly": { "is_anomaly": false, "anomaly_score": 0.04 },
  "mode": "NOMINAL"
}
```

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Health check + connected clients + RNN step count |
| `GET` | `/api/terrain` | Heightmap + soil map + slip map (JSON arrays) |
| `GET` | `/api/terrain/physics?row=64&col=64` | Full Bekker-Wong physics breakdown for one grid cell |
| `GET` | `/api/rnn/state` | GRU RNN learning metrics (before/after MAE, improvement %) |
| `POST` | `/api/path` | Compute A\* path from start to goal |
| `GET` | `/api/path` | Get current active rover path |
| `GET` | `/api/mission_log` | Recent anomaly event log |
| `POST` | `/api/inject_fault` | Inject a telemetry fault for demo |
| `POST` | `/api/mode` | Manually set rover system mode |
| `POST` | `/api/reset_battery` | Reset battery to 95% (demo reset) |
| `POST` | `/api/agent/chat` | Chat with the AI Mission Copilot |
| `GET` | `/api/agent/decisions` | View AI agent decision history |

### Inject a Fault (Example)

```bash
curl -X POST http://localhost:8000/api/inject_fault \
  -H "Content-Type: application/json" \
  -d '{"fault_type": "motor_temp_spike", "target": "front_left", "magnitude": 2.0, "duration_ticks": 30}'
```

**Available fault types:**
- `motor_temp_spike` — heats a specific wheel motor
- `battery_drain` — accelerates battery discharge
- `comms_dropout` — degrades communications signal
- `tilt_hazard` — simulates rover entering a steep slope

---

## 🔬 How It Works

### 1. Bekker-Wong Terramechanics (`lunar_physics.py`)

Models how rover wheels interact with lunar regolith under 1/6th Earth gravity. Computes:
- **Sinkage depth** (how deep wheels sink into soil)
- **Slip ratio** (wheel spinning without forward movement)
- **Traversal cost** (combined physics-based navigation difficulty)

Three soil classes based on Apollo mission data:

| Soil | Cohesion | Friction Angle | Location |
|---|---|---|---|
| SOFT | 0.2 kPa | 30° | Crater rims, dust traps |
| MEDIUM | 0.5 kPa | 38° | Standard regolith plains |
| FIRM | 1.0 kPa | 45° | Compacted basin floors |

### 2. Online GRU RNN (`terrain_rnn.py`)

A pure-NumPy Gated Recurrent Unit that runs **entirely on-device** (no PyTorch/TensorFlow needed). It:
- Learns the *difference* between physics-predicted cost and actually-observed cost
- Updates its weights every telemetry tick via online SGD
- Replays past experiences every 50 ticks for stability

This allows the rover to **adapt to unexpected terrain** (sub-surface rocks, unmodeled dust patches) in real time.

### 3. A\* Pathfinding (`pathfinding.py`)

Standard A\* over the 128×128 cost grid where:
- Cost = Bekker-Wong traversal cost (blended with RNN correction)
- Slopes > 25° → `cost = ∞` (completely blocked)
- Emergency replanning tightens slope limit to 15° for a safer route

### 4. Anomaly Detection (`anomaly_detector.py`)

Random Forest Classifier monitors incoming telemetry and flags:
- `MOTOR_OVERHEAT` → wheel temp > 75°C
- `BATTERY_DRAIN_CRITICAL` → >3% voltage drop/min
- `COMMS_DROPOUT` → signal < 15%
- `TILT_HAZARD` → chassis tilt > 30°

### 5. AI Agent (`agent.py`)

When an anomaly fires, the NVIDIA NIM agent (LLaMA-3.3-70B) is invoked with tools:
- `get_telemetry_window()` — inspect sensor history
- `get_terrain_info()` — query slope/soil at current position
- `trigger_replan()` — recompute safer A\* path
- `set_system_mode()` — switch rover mode (COOL_DOWN, HIBERNATION, etc.)
- `get_mission_log()` — review past anomaly events

If the NVIDIA API is unavailable, a deterministic rule-based fallback agent handles decisions locally.

---

## 🎮 Frontend Controls

### 3D Terrain Viewer (`terrain.html`)

| Action | How |
|---|---|
| **Set rover goal** | Double-click anywhere on the terrain mesh |
| **Rotate view** | Click + drag |
| **Zoom** | Scroll wheel |
| **Pan** | Right-click + drag |
| **Elevation mode** | Press `H` key |
| **Slope heatmap** | Press `S` key |
| **Learned traversability** | Press `L` key |

The HUD overlays show:
- **Wheel Slip %** (live)
- **Soil Zone** (SOFT / MEDIUM / FIRM)
- **Sinkage Depth** (cm)
- **RNN Residual Correction** (Δc)
- **RNN Accuracy Improvement** (%)

### Fault Injection (Dashboard)

Use the fault injection buttons on the dashboard to simulate:
- 🔥 Motor temperature spike
- 🔋 Battery drain
- 📡 Comms dropout

Each fault triggers the anomaly detector → FDIR system → AI agent autonomously responds.

---

## 🛠️ Tech Stack

### Backend
| Technology | Use |
|---|---|
| **Python 3.10+** | Core language |
| **FastAPI** | REST API + WebSocket server |
| **Uvicorn** | ASGI server |
| **Rasterio** | GeoTIFF DEM file loading |
| **NumPy** | Physics engine + GRU RNN |
| **Scikit-learn** | Random Forest anomaly detector |
| **SciPy** | Spatial interpolation |
| **OpenAI SDK** | NVIDIA NIM API client |

### Frontend
| Technology | Use |
|---|---|
| **React 19** | UI framework |
| **Vite 8** | Build tool + dev server |
| **Three.js** | 3D terrain visualization |
| **Recharts** | Live telemetry charts |
| **Lucide React** | Icons |
| **Vanilla CSS** | Styling |

---

## ⚙️ Team Notes & Configuration

### Tuning the Rover

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `max_traversable_slope` | `dem_processor.py` | `25°` | Lower → more terrain blocked → more dramatic rerouteing |
| `GRID_SIZE` | `main.py` env var | `128` | Higher → more detail but slower (only useful with high-res DEM) |
| Telemetry rate | `main.py` → `asyncio.sleep()` | `1.0 sec` | Adjust rover speed |
| Replay interval | `main.py` | `50 ticks` | RNN experience replay frequency |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEM_PATH` | `../data/dem.tif` | Path to DEM GeoTIFF file |
| `GRID_SIZE` | `128` | Heightmap resolution |
| `NVIDIA_API_KEY` | (none) | NVIDIA NIM API key for AI agent |

### Important Notes

- **CORS** is set to `allow_origins=["*"]` for dev convenience — **do not deploy this to production** without restricting origins.
- The backend automatically falls back to a synthetic crater DEM if `data/dem.tif` is missing.
- The AI agent falls back to a deterministic rule engine if the NVIDIA API is unavailable or times out (3s timeout).
- `data/dem.tif` is excluded from git (in `.gitignore`) because it can be hundreds of MB. Team members must each download the DEM manually.

---

## 📖 Further Reading

- [SYSTEM_ARCHITECTURE_AND_MODELS.md](./SYSTEM_ARCHITECTURE_AND_MODELS.md) — Full mathematical specification: Bekker-Wong equations, GRU math, cost grid formulas
- [USGS Astrogeology Moon LRO LOLA DEM](https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m) — Primary DEM data source
- [USGS Astropedia Search](https://astrogeology.usgs.gov/search) — Browse all available lunar DEMs
- [NVIDIA NIM API](https://build.nvidia.com) — Free LLM inference API used by the AI agent
- [LRO LOLA Mission Overview](https://lunar.gsfc.nasa.gov/lola/) — NASA LOLA instrument background

---

## 👥 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

<div align="center">

**Built with ❤️ for Smart India Hackathon 2026**  
🚀 *Advancing autonomous lunar rover technology*

</div>