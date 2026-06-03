"""
Physics-Based Exponential Degradation Model
=============================================
Fits an exponential decay curve:  SOH(n) = a · exp(-b · n) + c

Where:
    n : cycle number
    a : initial amplitude (≈ initial SOH above asymptote)
    b : degradation rate  (larger b → faster fade)
    c : asymptotic SOH    (lower bound)

The model is fit per battery via scipy.optimize.curve_fit.
RUL is estimated analytically by solving for the cycle at which
SOH crosses the End-of-Life (EOL) threshold (default 80%).
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error


# ── Decay functions ────────────────────────────────────────────────────────────

def exponential_decay(n, a, b, c):
    """y = a · exp(−b · n) + c"""
    return a * np.exp(-b * n) + c


def double_exponential(n, a1, b1, a2, b2, c):
    """y = a1·exp(−b1·n) + a2·exp(−b2·n) + c  (two-phase degradation)"""
    return a1 * np.exp(-b1 * n) + a2 * np.exp(-b2 * n) + c


def power_law_decay(n, a, b, c):
    """y = a · n^(−b) + c"""
    return a * np.power(np.maximum(n, 1e-8), -b) + c


# ── Main class ─────────────────────────────────────────────────────────────────

class ExponentialDegradationModel:
    """
    Per-battery physics-based degradation model using curve fitting.

    Attributes:
        battery_params : dict  {battery_id: {a, b, c, R2, RMSE}}
        EOL_THRESHOLD  : float  SOH threshold for End-of-Life (default 80%)
    """

    EOL_THRESHOLD = 80.0

    def __init__(self, func: str = 'exponential'):
        """
        Args:
            func : 'exponential', 'double_exponential', or 'power_law'
        """
        self.func_name     = func
        self.battery_params = {}
        self.is_fitted      = False

        _map = {
            'exponential'        : (exponential_decay,
                                    [100.0, 0.002, 40.0],
                                    ([0, 1e-10, 0], [200, 1, 150])),
            'double_exponential' : (double_exponential,
                                    [50.0, 0.01, 50.0, 0.001, 40.0],
                                    ([0]*5, [200, 10, 200, 10, 150])),
            'power_law'          : (power_law_decay,
                                    [100.0, 0.1, 40.0],
                                    ([0, 1e-8, 0], [500, 5, 150]))
        }
        if func not in _map:
            raise ValueError(f"Unknown func '{func}'. "
                             f"Choose: exponential, double_exponential, power_law")
        self._func, self._p0, self._bounds = _map[func]

    # ── Fitting ────────────────────────────────────────────────────
    def _fit_single(self, cycles: np.ndarray, soh: np.ndarray, bid: str) -> dict:
        """Fit curve to a single battery's data."""
        try:
            popt, _ = curve_fit(
                self._func, cycles, soh,
                p0=self._p0, bounds=self._bounds,
                maxfev=50_000
            )
            soh_pred = self._func(cycles, *popt)
            r2   = r2_score(soh, soh_pred)
            rmse = np.sqrt(mean_squared_error(soh, soh_pred))

            params = {'params': popt, 'R2': round(r2, 5), 'RMSE': round(rmse, 5)}
            self.battery_params[bid] = params

            if self.func_name == 'exponential':
                a, b, c = popt
                print(f"    ✔ {bid:4s} | a={a:.3f}  b={b:.6f}  c={c:.3f} "
                      f"| R²={r2:.4f}  RMSE={rmse:.4f}")
            else:
                print(f"    ✔ {bid:4s} | params={np.round(popt, 4)} "
                      f"| R²={r2:.4f}  RMSE={rmse:.4f}")
            return params

        except Exception as exc:
            warnings.warn(f"⚠️  Curve fit failed for {bid}: {exc}")
            return {}

    def fit(self, df: pd.DataFrame) -> 'ExponentialDegradationModel':
        """Fit the degradation model for every battery in df."""
        print(f"\n  Fitting {self.func_name} decay per battery:")
        for bid in sorted(df['battery_id'].unique()):
            bdf    = df[df['battery_id'] == bid].sort_values('cycle')
            cycles = bdf['cycle'].values.astype(float)
            soh    = bdf['SOH'].values.astype(float)
            self._fit_single(cycles, soh, bid)

        self.is_fitted = True
        return self

    # ── Prediction ─────────────────────────────────────────────────
    def predict_soh(self, battery_id: str, cycles: np.ndarray) -> np.ndarray:
        """Predict SOH for given cycle numbers."""
        if battery_id not in self.battery_params:
            raise KeyError(f"Battery '{battery_id}' not fitted. "
                           f"Available: {list(self.battery_params.keys())}")
        popt = self.battery_params[battery_id]['params']
        return self._func(cycles.astype(float), *popt)

    def predict_rul(self, battery_id: str, current_cycle: int,
                     search_horizon: int = 10_000) -> int:
        """
        Estimate RUL: number of cycles until SOH ≤ EOL_THRESHOLD.

        Uses binary-search over future cycles to find the first crossing.

        Returns:
            Estimated RUL (int). Returns search_horizon if EOL not reached.
        """
        if battery_id not in self.battery_params:
            raise KeyError(f"Battery '{battery_id}' not fitted.")

        future = np.arange(current_cycle, current_cycle + search_horizon, dtype=float)
        soh_future = self.predict_soh(battery_id, future)

        below_eol = np.where(soh_future <= self.EOL_THRESHOLD)[0]
        if len(below_eol) == 0:
            return search_horizon

        eol_cycle = int(future[below_eol[0]])
        return max(0, eol_cycle - current_cycle)

    # ── Summary ────────────────────────────────────────────────────
    def get_summary(self) -> pd.DataFrame:
        """Return fitted parameters as a DataFrame."""
        rows = []
        for bid, info in self.battery_params.items():
            row = {'battery_id': bid, 'R2': info['R2'], 'RMSE': info['RMSE']}
            popt = info['params']
            if self.func_name == 'exponential':
                row.update({'a': popt[0], 'b': popt[1], 'c': popt[2]})
            else:
                row.update({f'p{i}': v for i, v in enumerate(popt)})
            rows.append(row)
        return pd.DataFrame(rows)

    def estimate_eol_cycle(self, battery_id: str) -> float:
        """Return the predicted cycle at which SOH crosses EOL threshold."""
        if battery_id not in self.battery_params:
            return np.nan
        return float(self.predict_rul(battery_id, current_cycle=1))
