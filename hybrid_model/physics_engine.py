"""
Physics Engine  —  Thermal + Stress Models
===========================================
Implements the Physics Engine block from the Hybrid System Architecture (Fig 12.1).

Outputs per sample:
    SOH_phys      : Physics-based SOH from exponential decay model
                    SOH(n) = a · exp(−b · n) + c
    RUL_phys_norm : Physics-based RUL normalised to [0, 1]
    PDI           : Physics Degradation Index
                    PDI = 1 − BCt / BCt_initial  ∈ [0, 1]
    thermal_stress: Arrhenius-inspired thermal stress indicator
                    thermal_stress = (chT − T_ref) / T_range ∈ [0, 1]
    current_stress: C-rate stress indicator
                    current_stress = disI / I_max ∈ [0, 1]

The Physics Engine is fit per battery using exponential curve fitting (scipy).
SOH_phys and RUL_phys are then used as physics regularisation signals
fed directly into the loss function of the Physics-Informed ANN (dashed
orange arrow in Fig 12.1).
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# ── Physical constants / reference values ─────────────────────────────────────
T_REF   = 25.0   # Reference temperature (°C)  — room temperature
T_RANGE = 20.0   # Expected temperature range above T_REF (°C)
I_MAX   = 2.5    # Expected maximum discharge current (A)
EOL_SOH = 80.0   # End-of-Life SOH threshold (%)


# ── Degradation model ──────────────────────────────────────────────────────────
def _exp_decay(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay:  SOH(n) = a · exp(−b · n) + c"""
    return a * np.exp(-b * n) + c


class PhysicsEngine:
    """
    Implements the Physics Engine block (Fig 12.1):
        - Exponential degradation model fitted per battery (Thermal model proxy)
        - Thermal stress indicator (Stress model)
        - Current stress indicator (Stress model)
        - Physics Degradation Index (PDI)

    Usage:
        engine = PhysicsEngine()
        engine.fit(df_train)                   # fit per-battery degradation curves
        df_with_physics = engine.transform(df) # add physics columns to any split
    """

    EOL_SOH = EOL_SOH

    def __init__(self):
        self.battery_params: dict = {}  # {battery_id: (a, b, c)}
        self.bct_initial:    dict = {}  # {battery_id: BCt at cycle 1}
        self.max_rul:        dict = {}  # {battery_id: max RUL observed}
        self.is_fitted:      bool = False

    # ── Fitting ────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> 'PhysicsEngine':
        """
        Fit exponential degradation curve per battery.
        Records initial capacity and maximum RUL for normalisation.
        """
        print("\n[PhysicsEngine] Fitting thermal + stress models per battery:")
        for bid in sorted(df['battery_id'].unique()):
            bdf    = df[df['battery_id'] == bid].sort_values('cycle')
            cycles = bdf['cycle'].values.astype(float)
            soh    = bdf['SOH'].values.astype(float)

            # Initial capacity = maximum BCt seen for this battery
            self.bct_initial[bid] = bdf['BCt'].max()
            self.max_rul[bid]     = bdf['RUL'].max()

            # Fit exponential decay
            try:
                p0     = [100.0, 0.003, 0.0]
                bounds = ([0, 1e-10, 0], [200, 1, 150])
                popt, _ = curve_fit(_exp_decay, cycles, soh,
                                    p0=p0, bounds=bounds, maxfev=50_000)
                r2 = r2_score(soh, _exp_decay(cycles, *popt))
                self.battery_params[bid] = popt
                print(f"  {bid:4s} | a={popt[0]:.3f}  b={popt[1]:.6f}  "
                      f"c={popt[2]:.3f} | R²={r2:.4f}")
            except Exception as e:
                warnings.warn(f"  {bid}: curve fit failed — {e}. Using linear fallback.")
                slope = np.polyfit(cycles, soh, 1)
                self.battery_params[bid] = None
                self._linear_params = {bid: slope}

        self.is_fitted = True
        return self

    # ── SOH / RUL physics predictions ─────────────────────────────────────────
    def _predict_soh_phys(self, bid: str, cycles: np.ndarray) -> np.ndarray:
        """Return physics SOH for given cycles using fitted exponential model."""
        if self.battery_params.get(bid) is None:
            # linear fallback
            slope = getattr(self, '_linear_params', {}).get(bid, [0, 100])
            return np.polyval(slope, cycles)
        a, b, c = self.battery_params[bid]
        return _exp_decay(cycles.astype(float), a, b, c)

    def _predict_rul_phys(self, bid: str, current_cycle: int,
                           horizon: int = 10_000) -> float:
        """Estimate RUL from physics model: cycles until SOH_phys <= EOL_SOH."""
        if bid not in self.battery_params or self.battery_params[bid] is None:
            return 0.0
        future = np.arange(current_cycle, current_cycle + horizon, dtype=float)
        soh_f  = _exp_decay(future, *self.battery_params[bid])
        below  = np.where(soh_f <= self.EOL_SOH)[0]
        if len(below) == 0:
            return float(horizon)
        return float(future[below[0]] - current_cycle)

    # ── Main transform ─────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and append physics features to df:
            SOH_phys, RUL_phys_norm, PDI, thermal_stress, current_stress
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy().sort_values(['battery_id', 'cycle']).reset_index(drop=True)

        soh_phys_list   = []
        rul_phys_list   = []
        pdi_list        = []
        therm_list      = []
        curr_list       = []

        for bid in df['battery_id'].unique():
            mask   = df['battery_id'] == bid
            bdf    = df.loc[mask].sort_values('cycle')
            cycles = bdf['cycle'].values.astype(float)

            # ── SOH_phys ──────────────────────────────────────────
            soh_p = self._predict_soh_phys(bid, cycles)
            soh_p = np.clip(soh_p, 0, 100)

            # ── RUL_phys (normalised to [0,1] by max_rul) ─────────
            max_rul = self.max_rul.get(bid, 1.0) or 1.0
            rul_p   = np.array([self._predict_rul_phys(bid, int(c)) for c in cycles])
            rul_p_n = np.clip(rul_p / max_rul, 0, 1)

            # ── PDI = 1 − BCt / BCt_initial ───────────────────────
            bct_init = self.bct_initial.get(bid, bdf['BCt'].max())
            if bct_init == 0:
                bct_init = 1.0
            pdi = np.clip(1.0 - bdf['BCt'].values / bct_init, 0, 1)

            # ── Thermal stress: Arrhenius proxy ───────────────────
            # thermal_stress = (chT − T_ref) / T_range, clipped [0, 1]
            therm = np.clip((bdf['chT'].values - T_REF) / T_RANGE, 0, 1)

            # ── Current (C-rate) stress ────────────────────────────
            i_max_actual = max(df['disI'].max(), I_MAX)
            curr = np.clip(bdf['disI'].values / i_max_actual, 0, 1)

            soh_phys_list.extend(soh_p)
            rul_phys_list.extend(rul_p_n)
            pdi_list.extend(pdi)
            therm_list.extend(therm)
            curr_list.extend(curr)

        df['SOH_phys']      = soh_phys_list
        df['RUL_phys_norm'] = rul_phys_list
        df['PDI']           = pdi_list
        df['thermal_stress'] = therm_list
        df['current_stress'] = curr_list

        print(f"[PhysicsEngine] Physics features added: "
              f"SOH_phys, RUL_phys_norm, PDI, thermal_stress, current_stress")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_params_summary(self) -> pd.DataFrame:
        rows = []
        for bid, params in self.battery_params.items():
            if params is not None:
                rows.append({'battery_id': bid,
                             'a': params[0], 'b': params[1], 'c': params[2],
                             'BCt_initial': self.bct_initial.get(bid),
                             'max_RUL': self.max_rul.get(bid)})
        return pd.DataFrame(rows)
