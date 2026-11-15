"""Econometric estimators for Engle-Granger cointegration, OU half-life, and VECM systems."""

import math
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from pairs_coint.models import CointegrationSummary, VECMParameters


class CointegrationEngine:
    """Statistical verification of cointegration and vector error correction dynamics."""

    @staticmethod
    def test_engle_granger(
        y_series: NDArray[np.float64],
        x_series: NDArray[np.float64],
        p_val_threshold: float = 0.05,
    ) -> CointegrationSummary:
        """
        Conduct Engle-Granger two-step cointegration test with OLS and Dickey-Fuller residual test.
        """
        n = len(y_series)
        if n != len(x_series):
            raise ValueError("Series lengths must match.")
        if n < 30:
            raise ValueError("Insufficient data points for cointegration test (min 30).")

        # Step 1: OLS regression y = beta * x + alpha
        x_design = np.column_stack([x_series, np.ones(n, dtype=np.float64)])
        coeffs, _, _, _ = np.linalg.lstsq(x_design, y_series, rcond=None)
        static_beta = float(coeffs[0])
        static_alpha = float(coeffs[1])

        # Step 2: Cointegrating residuals (spread)
        spread = y_series - (static_beta * x_series + static_alpha)
        spread_mean = float(np.mean(spread))
        spread_std = float(np.std(spread, ddof=1))

        # Step 3: Fit OU process: delta_z_t = gamma * z_{t-1} + error
        delta_spread = np.diff(spread)
        lagged_spread = spread[:-1]

        gamma_stat, _, _, _ = np.linalg.lstsq(
            lagged_spread[:, np.newaxis], delta_spread, rcond=None
        )
        gamma = float(gamma_stat[0])

        residuals = delta_spread - gamma * lagged_spread
        s_sq = np.sum(residuals ** 2) / max(1, (len(delta_spread) - 1))
        denom = np.sum(lagged_spread ** 2)
        se_gamma = math.sqrt(s_sq / max(1e-12, denom))
        t_stat = gamma / max(1e-12, se_gamma)

        is_coint = t_stat < -3.37
        approx_p_val = float(stats.t.cdf(t_stat, df=len(delta_spread) - 1))

        # Step 4: Half-life derivation
        if -1.0 < gamma < 0.0:
            half_life = math.log(2.0) / (-math.log(1.0 + gamma))
        else:
            half_life = float("inf")

        return CointegrationSummary(
            is_cointegrated=is_coint,
            static_beta=static_beta,
            static_alpha=static_alpha,
            adf_statistic=t_stat,
            adf_p_value=approx_p_val,
            half_life_ticks=half_life,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

    @staticmethod
    def fit_vecm(
        y_series: NDArray[np.float64],
        x_series: NDArray[np.float64],
        beta: float,
        alpha: float,
    ) -> VECMParameters:
        """
        Fit bivariate Vector Error Correction Model:
        Delta Y_t = mu_y + alpha_y * z_{t-1} + e_{y, t}
        Delta X_t = mu_x + alpha_x * z_{t-1} + e_{x, t}
        """
        z = y_series - (beta * x_series + alpha)
        delta_y = np.diff(y_series)
        delta_x = np.diff(x_series)
        z_lag = z[:-1]

        m = len(delta_y)
        design = np.column_stack([z_lag, np.ones(m, dtype=np.float64)])

        coeffs_y, _, _, _ = np.linalg.lstsq(design, delta_y, rcond=None)
        alpha_y = float(coeffs_y[0])

        coeffs_x, _, _, _ = np.linalg.lstsq(design, delta_x, rcond=None)
        alpha_x = float(coeffs_x[0])

        y_pred = design @ coeffs_y
        ss_res_y = np.sum((delta_y - y_pred) ** 2)
        ss_tot_y = np.sum((delta_y - np.mean(delta_y)) ** 2)
        r2_y = 1.0 - (ss_res_y / ss_tot_y) if ss_tot_y > 1e-12 else 0.0

        x_pred = design @ coeffs_x
        ss_res_x = np.sum((delta_x - x_pred) ** 2)
        ss_tot_x = np.sum((delta_x - np.mean(delta_x)) ** 2)
        r2_x = 1.0 - (ss_res_x / ss_tot_x) if ss_tot_x > 1e-12 else 0.0

        return VECMParameters(
            alpha_y=alpha_y,
            alpha_x=alpha_x,
            cointegrating_beta=beta,
            intercept=alpha,
            r_squared_y=float(max(0.0, r2_y)),
            r_squared_x=float(max(0.0, r2_x)),
        )
