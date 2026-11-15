"""Domain models for cointegration tests, Kalman states, and VECM dynamics."""

from dataclasses import dataclass


@dataclass(slots=True)
class CointegrationSummary:
    """Summary metrics of Engle-Granger cointegration and OU mean-reversion analysis."""
    is_cointegrated: bool
    static_beta: float
    static_alpha: float
    adf_statistic: float
    adf_p_value: float
    half_life_ticks: float
    spread_mean: float
    spread_std: float


@dataclass(slots=True)
class KalmanStepResult:
    """State-space estimate output at an individual tick observation."""
    step: int
    beta: float
    alpha: float
    innovation: float
    innovation_variance: float
    spread_z_score: float


@dataclass(slots=True)
class VECMParameters:
    """Estimated Vector Error Correction Model parameters for a bivariate system."""
    alpha_y: float
    alpha_x: float
    cointegrating_beta: float
    intercept: float
    r_squared_y: float
    r_squared_x: float
