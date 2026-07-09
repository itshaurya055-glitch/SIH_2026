# ML / Data Notes (Week 1)

## Anomaly Detection Dataset Collection
The primary telemetry generation is located in `backend/telemetry.py`.
This script simulates the realistic conditions of a rover's sensor stream (battery, motor temps, tilt, comms).
It also includes a fault injection mechanism (`_apply_faults`) which can manually trigger specific anomalies like `motor_temp_spike` or `comms_dropout`.

**For the ML team:**
`telemetry.py` is the file you will want to look at closely to decide "what does a realistic fault look like". Better fault modeling here directly improves how well the Week 3 anomaly detector performs. You can extend this to add oscillation, slow drift, etc. 
In the short term, you can use the WebSocket stream or modify the backend to save the telemetry to CSV/JSON lines to start building the "normal vs anomalous" dataset for training.

## Terrain Generation Decision
We are deciding between using real DEM (e.g., lunar/Mars elevation data from NASA PDS or ISRO Bhuvan) vs procedurally generated terrain.
For velocity in the immediate term, **procedurally generated terrain** will be used as a placeholder. We will upgrade to real DEM data later if time permits and a higher degree of realism is desired.
