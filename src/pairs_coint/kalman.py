"""Online state-space Kalman Filter tracking dynamic beta and intercept without memory allocation."""

import math
import numpy as np
from numpy.typing import NDArray
from pairs_coint.models import KalmanStepResult


class DynamicHedgeKalmanFilter:
    """
    Two-state recursive Kalman Filter estimating y_t = beta_t * x_t + alpha_t.
    Operates in strictly deterministic O(1) time per tick.
    """

    __slots__ = (
        "delta",
        "obs_variance_r",
        "step",
        "_state",
        "_p",
        "_q",
    )

    def __init__(
        self,
        delta: float = 1e-4,
        observation_variance_r: float = 1e-3,
        initial_beta: float = 1.0,
        initial_alpha: float = 0.0,
    ) -> None:
        if delta <= 0 or observation_variance_r <= 0:
            raise ValueError("Variance parameters delta and R must be positive.")

        self.delta = delta
        self.obs_variance_r = observation_variance_r
        self.step: int = 0

        self._state: NDArray[np.float64] = np.array([initial_beta, initial_alpha], dtype=np.float64)
        self._p: NDArray[np.float64] = np.eye(2, dtype=np.float64) * 1.0

        q_scalar = delta / (1.0 - delta) if delta < 1.0 else delta
        self._q: NDArray[np.float64] = np.eye(2, dtype=np.float64) * q_scalar

    def update(self, x: float, y: float) -> KalmanStepResult:
        """
        Ingest simultaneous tick prices (x, y) and compute updated state in O(1).
        """
        self.step += 1

        # 1. State Prediction
        p_pred_00 = self._p[0, 0] + self._q[0, 0]
        p_pred_01 = self._p[0, 1]
        p_pred_10 = self._p[1, 0]
        p_pred_11 = self._p[1, 1] + self._q[1, 1]

        # 2. Measurement Update
        y_hat = self._state[0] * x + self._state[1]
        innovation = y - y_hat

        hp0 = x * p_pred_00 + p_pred_10
        hp1 = x * p_pred_01 + p_pred_11
        s = hp0 * x + hp1 + self.obs_variance_r

        k0 = (p_pred_00 * x + p_pred_01) / s
        k1 = (p_pred_10 * x + p_pred_11) / s

        self._state[0] += k0 * innovation
        self._state[1] += k1 * innovation

        self._p[0, 0] = p_pred_00 - k0 * hp0
        self._p[0, 1] = p_pred_01 - k0 * hp1
        self._p[1, 0] = p_pred_10 - k1 * hp0
        self._p[1, 1] = p_pred_11 - k1 * hp1

        z_score = innovation / math.sqrt(max(1e-12, s))

        return KalmanStepResult(
            step=self.step,
            beta=float(self._state[0]),
            alpha=float(self._state[1]),
            innovation=float(innovation),
            innovation_variance=float(s),
            spread_z_score=float(z_score),
        )

    @property
    def current_beta(self) -> float:
        return float(self._state[0])

    @property
    def current_alpha(self) -> float:
        return float(self._state[1])
