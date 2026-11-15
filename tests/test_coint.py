import numpy as np
import pytest
from pairs_coint.coint import CointegrationEngine
from pairs_coint.kalman import DynamicHedgeKalmanFilter


def test_kalman_dynamic_beta_adaptation():
    rng = np.random.default_rng(42)
    n = 2_000

    x = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=n))
    beta_true = np.empty(n, dtype=np.float64)
    beta_true[:1000] = 1.5
    beta_true[1000:] = 2.2

    noise = rng.normal(0.0, 0.1, size=n)
    y = beta_true * x + 10.0 + noise

    kf = DynamicHedgeKalmanFilter(delta=1e-3, observation_variance_r=0.05, initial_beta=1.0)

    for i in range(n):
        _ = kf.update(x[i], y[i])

    assert pytest.approx(kf.current_beta, abs=0.08) == 2.2
    assert pytest.approx(kf.current_alpha, abs=1.5) == 10.0


def test_engle_granger_synthetic_cointegration():
    rng = np.random.default_rng(123)
    n = 1_500

    x = 50.0 + np.cumsum(rng.normal(0.0, 1.0, size=n))
    z = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        z[t] = 0.85 * z[t - 1] + rng.normal(0.0, 0.2)

    y = 2.0 * x + 5.0 + z

    res = CointegrationEngine.test_engle_granger(y, x)

    assert res.is_cointegrated is True
    assert pytest.approx(res.static_beta, abs=0.05) == 2.0
    assert pytest.approx(res.static_alpha, abs=0.5) == 5.0
    assert res.adf_statistic < -3.37
    assert 2.0 < res.half_life_ticks < 8.0


def test_vecm_error_correction_speeds():
    rng = np.random.default_rng(999)
    n = 1_000

    x = np.cumsum(rng.normal(0.0, 0.5, size=n)) + 100.0
    spread = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        spread[t] = 0.80 * spread[t - 1] + rng.normal(0.0, 0.1)

    y = x + spread

    vecm = CointegrationEngine.fit_vecm(y_series=y, x_series=x, beta=1.0, alpha=0.0)

    assert vecm.alpha_y < -0.05
    assert abs(vecm.alpha_x) < abs(vecm.alpha_y)
