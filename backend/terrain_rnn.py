"""
Terrain-adaptive GRU — online self-supervised traversability learning.

Architecture:
  input  : [physics_cost, observed_cost_proxy, slope_deg, slip_ratio]  (4 features)
  hidden : GRU cell, hidden_size=16
  output : residual_correction scalar  (signed; positive = harder than physics predicts)

Training signal:
  After each rover step we observe a cost PROXY from telemetry
  (motor_temp delta + tilt change relative to expectation). The
  residual (observed - physics_predicted) is the learning target.

  The GRU learns the *sequence pattern* of residuals — e.g. "every time
  the rover enters this kind of slope after crossing loose soil it gets
  more resistance than the physics model predicted" — and pre-corrects.

Why GRU not LSTM:
  GRU has the same expressive power for short sequences but fewer
  parameters (no separate cell state), making it faster and more stable
  for online learning with small updates per step.

Why NumPy not PyTorch:
  Zero extra dependencies, instant cold-start, runs on any CPU.
  The trade-off is we implement backprop manually — but for a 4→16→1
  network this is < 100 lines of math and fully auditable.
"""

import numpy as np
from collections import deque


# ── Sigmoid / tanh helpers ────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clean = np.nan_to_num(x, nan=0.0, posinf=15.0, neginf=-15.0)
    return 1.0 / (1.0 + np.exp(-np.clip(x_clean, -15, 15)))


def _tanh(x: np.ndarray) -> np.ndarray:
    x_clean = np.nan_to_num(x, nan=0.0, posinf=15.0, neginf=-15.0)
    return np.tanh(np.clip(x_clean, -15, 15))


# ── GRU cell (forward pass) ───────────────────────────────────────────────────

def _gru_forward(x: np.ndarray, h: np.ndarray, W_z, U_z, b_z,
                  W_r, U_r, b_r, W_h, U_h, b_h) -> np.ndarray:
    """
    One GRU step.

    x : input vector  (input_size,)
    h : hidden state  (hidden_size,)
    Returns h_new     (hidden_size,)
    """
    z = _sigmoid(W_z @ x + U_z @ h + b_z)   # update gate
    r = _sigmoid(W_r @ x + U_r @ h + b_r)   # reset gate
    h_hat = _tanh(W_h @ x + U_h @ (r * h) + b_h)  # candidate hidden
    return (1 - z) * h + z * h_hat


class TerrainRNN:
    """
    Online-learning GRU that tracks the residual between physics-predicted
    and actually-observed traversal cost, and outputs a per-step correction.

    Usage
    -----
    rnn = TerrainRNN()

    # After each rover step:
    correction = rnn.predict(features)
    rnn.update(features, residual)          # online single-step gradient update

    # Every N steps (optional batch replay to reduce variance):
    rnn.replay_update()
    """

    INPUT_SIZE  = 4    # [physics_cost, observed_cost_proxy, slope_deg, slip_ratio]
    HIDDEN_SIZE = 16
    OUTPUT_SIZE = 1    # scalar residual correction

    def __init__(self, lr: float = 0.01, replay_buffer_size: int = 200, seed: int = 42):
        self.lr = lr
        self.replay_buffer: deque[tuple] = deque(maxlen=replay_buffer_size)
        self._h = np.zeros(self.HIDDEN_SIZE)   # hidden state
        self._step = 0
        self._accuracy_log: list[float] = []   # |residual| before correction, for metrics

        rng = np.random.default_rng(seed)
        scale = 0.1  # small init → stable online learning

        # ── GRU parameters ─────────────────────────────────────────────────
        I, H = self.INPUT_SIZE, self.HIDDEN_SIZE

        # Update gate
        self.W_z = rng.normal(0, scale, (H, I))
        self.U_z = rng.normal(0, scale, (H, H))
        self.b_z = np.zeros(H)

        # Reset gate
        self.W_r = rng.normal(0, scale, (H, I))
        self.U_r = rng.normal(0, scale, (H, H))
        self.b_r = np.zeros(H)

        # Candidate hidden
        self.W_h = rng.normal(0, scale, (H, I))
        self.U_h = rng.normal(0, scale, (H, H))
        self.b_h = np.zeros(H)

        # ── Output layer (linear) ──────────────────────────────────────────
        self.W_out = rng.normal(0, scale, (self.OUTPUT_SIZE, H))
        self.b_out = np.zeros(self.OUTPUT_SIZE)

        # ── Adam optimiser state ───────────────────────────────────────────
        # We only do Adam on the output layer for efficiency.
        self._m_out = np.zeros_like(self.W_out)
        self._v_out = np.zeros_like(self.W_out)
        self._m_b   = np.zeros_like(self.b_out)
        self._v_b   = np.zeros_like(self.b_out)
        self._adam_t = 0
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps   = 1e-8

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _normalize_input(self, features: list[float]) -> np.ndarray:
        """
        Rough per-feature normalisation so all inputs are in a similar range.
        features = [physics_cost, observed_cost_proxy, slope_deg, slip_ratio]
        """
        x = np.array(features, dtype=float)
        x = np.nan_to_num(x, nan=1.0, posinf=10.0, neginf=0.0)
        # Approximate normalisation: cost ~O(1-10), slope ~O(0-30), slip ~O(0-1)
        scale = np.array([5.0, 5.0, 30.0, 1.0])
        return np.clip(x / scale, -5.0, 5.0)

    def predict(self, features: list[float]) -> float:
        """
        Run one GRU step and return the residual correction scalar.
        Does NOT update weights or hidden state — call update() separately.
        """
        if np.isnan(self._h).any():
            self._h = np.zeros(self.HIDDEN_SIZE)
        x = self._normalize_input(features)
        h_new = _gru_forward(
            x, self._h,
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        )
        if np.isnan(h_new).any():
            h_new = np.zeros(self.HIDDEN_SIZE)
        out = float((self.W_out @ h_new + self.b_out)[0])
        if np.isnan(out) or np.isinf(out):
            out = 0.0
        return round(float(np.clip(out, -2.0, 2.0)), 4)

    # ── Online update ─────────────────────────────────────────────────────────

    def update(self, features: list[float], residual: float) -> float:
        """
        One online gradient update:
          1. Forward pass (updates hidden state).
          2. Compute MSE loss vs target residual.
          3. Adam update on output layer only (cheap, stable).
          4. Store in replay buffer.

        Returns the prediction error |predicted - residual| before update.
        """
        if np.isnan(self._h).any():
            self._h = np.zeros(self.HIDDEN_SIZE)
        residual = float(np.nan_to_num(residual, nan=0.0, posinf=2.0, neginf=-2.0))
        x = self._normalize_input(features)
        h_new = _gru_forward(
            x, self._h,
            self.W_z, self.U_z, self.b_z,
            self.W_r, self.U_r, self.b_r,
            self.W_h, self.U_h, self.b_h,
        )
        if np.isnan(h_new).any():
            h_new = np.zeros(self.HIDDEN_SIZE)
        self._h = h_new  # commit hidden state

        pred = float((self.W_out @ h_new + self.b_out)[0])
        if np.isnan(pred) or np.isinf(pred):
            pred = 0.0
        error = float(np.clip(pred - residual, -5.0, 5.0))
        self._accuracy_log.append(abs(error))

        # ── Output-layer Adam gradient ──────────────────────────────────────
        self._adam_t += 1
        grad_W = np.array([[error]]) * h_new.reshape(1, -1)  # (1, H)
        grad_b = np.array([error])

        self._m_out = self._beta1 * self._m_out + (1 - self._beta1) * grad_W
        self._v_out = self._beta2 * self._v_out + (1 - self._beta2) * grad_W ** 2
        m_hat = self._m_out / (1 - self._beta1 ** self._adam_t)
        v_hat = self._v_out / (1 - self._beta2 ** self._adam_t)
        self.W_out -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

        self._m_b = self._beta1 * self._m_b + (1 - self._beta1) * grad_b
        self._v_b = self._beta2 * self._v_b + (1 - self._beta2) * grad_b ** 2
        m_hat_b = self._m_b / (1 - self._beta1 ** self._adam_t)
        v_hat_b = self._v_b / (1 - self._beta2 ** self._adam_t)
        self.b_out -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self._eps)

        # Store in replay buffer
        self.replay_buffer.append((features, residual))
        self._step += 1

        return abs(error)

    def replay_update(self, n_steps: int = 32) -> float:
        """
        Mini-batch replay from history to reduce variance. Call every ~50 ticks.
        Returns mean loss over the replay batch.
        """
        if len(self.replay_buffer) < 8:
            return 0.0

        indices = np.random.choice(len(self.replay_buffer), size=min(n_steps, len(self.replay_buffer)), replace=False)
        buffer_list = list(self.replay_buffer)
        total_loss = 0.0

        for i in indices:
            feat, target = buffer_list[i]
            total_loss += self.update(feat, target)

        return total_loss / len(indices)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_accuracy_metrics(self) -> dict:
        """
        Returns before/after accuracy comparison for the frontend/API.
        'before' = mean |residual| in first 20 steps (cold model)
        'after'  = mean |residual| in last 20 steps (warm model)
        This is the measurable learning improvement story.
        """
        log = self._accuracy_log
        if len(log) < 5:
            return {"steps": self._step, "before_mae": None, "after_mae": None,
                    "improvement_pct": None, "replay_buffer_size": len(self.replay_buffer)}

        window = min(20, len(log) // 4)
        before_mae = float(np.mean(log[:window]))
        after_mae  = float(np.mean(log[-window:]))

        improvement = (before_mae - after_mae) / max(before_mae, 1e-6) * 100.0

        return {
            "steps": self._step,
            "before_mae": round(before_mae, 4),
            "after_mae": round(after_mae, 4),
            "improvement_pct": round(improvement, 1),
            "replay_buffer_size": len(self.replay_buffer),
        }

    def reset_hidden(self):
        """Reset hidden state — call when rover is teleported to a new location."""
        self._h = np.zeros(self.HIDDEN_SIZE)
