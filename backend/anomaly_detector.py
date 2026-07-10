import numpy as np
import time
from sklearn.ensemble import IsolationForest
from telemetry import RoverTelemetry

class AnomalyDetector:
    def __init__(self, training_ticks: int = 600):
        print("[detector] Initializing and training Isolation Forest anomaly detector...")
        # A low contamination means we assume training data is almost entirely clean normal telemetry
        self.model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        self.features_keys = [
            "battery_pct",
            "motor_temp_fl",
            "motor_temp_fr",
            "motor_temp_rl",
            "motor_temp_rr",
            "tilt_deg",
            "comms_signal"
        ]
        self.means = {}
        self.stds = {}
        self._train(training_ticks)

    def _extract_features(self, reading: dict) -> np.ndarray:
        return np.array([
            reading["battery_pct"],
            reading["motor_temp"]["front_left"],
            reading["motor_temp"]["front_right"],
            reading["motor_temp"]["rear_left"],
            reading["motor_temp"]["rear_right"],
            reading["tilt_deg"],
            reading["comms_signal"]
        ])

    def _train(self, ticks: int):
        sim = RoverTelemetry()
        data = []
        for _ in range(ticks):
            # Advance telemetry simulation under normal drift
            reading = sim.next_reading()
            feats = self._extract_features(reading)
            data.append(feats)
        
        X = np.array(data)
        self.model.fit(X)
        
        # Calculate baseline statistics for interpreting anomalies
        for idx, key in enumerate(self.features_keys):
            self.means[key] = np.mean(X[:, idx])
            self.stds[key] = np.std(X[:, idx])
            
        print(f"[detector] Training complete. Baseline stats computed for {ticks} ticks.")

    def check(self, reading: dict) -> list[dict]:
        feats = self._extract_features(reading).reshape(1, -1)
        pred = self.model.predict(feats)[0] # -1 for anomaly, 1 for normal
        score = self.model.decision_function(feats)[0] # negative = anomalous, positive = normal
        
        alerts = []
        if pred == -1:
            anomalous_sensors = []
            # Find which feature(s) are contributing most to the anomaly
            for idx, key in enumerate(self.features_keys):
                val = feats[0, idx]
                mean = self.means[key]
                std = max(self.stds[key], 1e-3)
                z = (val - mean) / std
                
                # Flag if it's more than 3 standard deviations away from the normal baseline
                if abs(z) > 3.0:
                    sensor_name = key.replace("_", " ")
                    anomalous_sensors.append({
                        "sensor": key,
                        "sensor_label": sensor_name,
                        "value": round(float(val), 2),
                        "z_score": round(float(z), 2)
                    })
            
            # Fallback if no individual feature crosses the 3-std threshold (multivariate drift)
            if not anomalous_sensors:
                z_scores = []
                for idx, key in enumerate(self.features_keys):
                    val = feats[0, idx]
                    mean = self.means[key]
                    std = max(self.stds[key], 1e-3)
                    z = (val - mean) / std
                    z_scores.append((abs(z), key, val, z))
                z_scores.sort(reverse=True)
                top_z = z_scores[0]
                anomalous_sensors.append({
                    "sensor": top_z[1],
                    "sensor_label": top_z[1].replace("_", " "),
                    "value": round(float(top_z[2]), 2),
                    "z_score": round(float(top_z[3]), 2)
                })

            for s in anomalous_sensors:
                alerts.append({
                    "sensor": s["sensor"],
                    "message": f"Anomaly detected on {s['sensor_label']}: {s['value']} (z-score: {s['z_score']})",
                    "score": round(float(score), 4),
                    "timestamp": time.time(),
                })
        return alerts
