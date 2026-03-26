"""
StintEngine — Tyre Degradation Model
Fit degradation curves from real FastF1 data per compound.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from src.config import COMPOUNDS


def fit_tyre_degradation(laps: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """
    Fit a quadratic degradation model per tyre compound:
        laptime = a + b * tyre_age + c * tyre_age^2

    Args:
        laps: Cleaned laps DataFrame (all drivers) with columns:
              Compound, TyreLife, LapTimeSeconds

    Returns:
        Dict mapping compound name → (a, b, c) coefficients.
        Example: {"SOFT": (91.5, 0.05, 0.002), ...}
    """
    coefficients = {}

    for compound in COMPOUNDS:
        compound_laps = laps[laps["Compound"] == compound].copy()

        if len(compound_laps) < 5:
            # Fallback: not enough data — use generic values
            print(f"  ⚠ {compound}: only {len(compound_laps)} laps — using fallback coefficients")
            if compound == "SOFT":
                coefficients[compound] = (91.0, 0.08, 0.003)
            elif compound == "MEDIUM":
                coefficients[compound] = (91.5, 0.05, 0.002)
            else:
                coefficients[compound] = (92.0, 0.03, 0.001)
            continue

        tyre_age = compound_laps["TyreLife"].values.astype(float)
        lap_times = compound_laps["LapTimeSeconds"].values.astype(float)

        # Remove outliers: drop laps > 2 std from mean (pit in/out laps, SC laps)
        mean_lt = np.mean(lap_times)
        std_lt = np.std(lap_times)
        mask = np.abs(lap_times - mean_lt) < 2 * std_lt
        tyre_age = tyre_age[mask]
        lap_times = lap_times[mask]

        if len(tyre_age) < 3:
            print(f"  ⚠ {compound}: insufficient clean data — using fallback")
            coefficients[compound] = (mean_lt, 0.05, 0.001)
            continue

        # Fit quadratic: laptime = a + b*age + c*age^2
        coeffs = np.polyfit(tyre_age, lap_times, deg=2)
        # polyfit returns [c, b, a] (highest degree first), we store as (a, b, c)
        a, b, c = coeffs[2], coeffs[1], coeffs[0]
        coefficients[compound] = (a, b, c)

        print(f"  ✓ {compound}: a={a:.2f}s  b={b:.4f}  c={c:.6f}  (n={len(tyre_age)} laps)")

    return coefficients


def predict_laptime(compound: str, tyre_age: int, coefficients: Dict) -> float:
    """
    Predict lap time for a given compound and tyre age.

    Args:
        compound: "SOFT", "MEDIUM", or "HARD"
        tyre_age: Number of laps on current set of tyres
        coefficients: Dict from fit_tyre_degradation()

    Returns:
        Predicted lap time in seconds.
    """
    a, b, c = coefficients[compound]
    return a + b * tyre_age + c * (tyre_age ** 2)


def get_base_laptime(coefficients: Dict) -> float:
    """Get the fastest base lap time across all compounds (at tyre_age=1)."""
    return min(predict_laptime(comp, 1, coefficients) for comp in coefficients)


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import get_all_drivers_laps

    print("Loading race data for tyre model calibration...")
    all_laps = get_all_drivers_laps()

    print(f"\nFitting degradation model ({len(all_laps)} total laps):\n")
    coeffs = fit_tyre_degradation(all_laps)

    print("\n─── Predicted lap times by tyre age ───")
    print(f"{'Age':>4}  {'SOFT':>8}  {'MEDIUM':>8}  {'HARD':>8}")
    print("─" * 36)
    for age in [1, 5, 10, 15, 20, 25, 30]:
        times = [predict_laptime(c, age, coeffs) for c in COMPOUNDS]
        print(f"{age:>4}  {times[0]:>8.2f}  {times[1]:>8.2f}  {times[2]:>8.2f}")

    print(f"\n✓ Base lap time: {get_base_laptime(coeffs):.2f}s")
    print("✓ Sanity check: SOFT should degrade fastest →",
          "PASS ✓" if coeffs["SOFT"][2] > coeffs["HARD"][2] else "FAIL ✗ (check data)")
