"""
StintEngine — Data Loader
Pull and clean F1 race data from FastF1.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import CACHE_DIR, RACE_YEAR, RACE_GP, RACE_SESSION, REFERENCE_DRIVER


def init_cache():
    """Enable FastF1 cache to avoid re-downloading."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_session(year: int = RACE_YEAR, gp: str = RACE_GP, session_type: str = RACE_SESSION):
    """
    Load an F1 session via FastF1.
    
    Args:
        year: Season year (e.g. 2023)
        gp: Grand Prix name (e.g. 'Bahrain')
        session_type: Session type — 'R' for race, 'Q' for qualifying, etc.
    
    Returns:
        fastf1.core.Session object with data loaded.
    """
    init_cache()
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the laps DataFrame:
    - Convert LapTime from timedelta to seconds (float)
    - Forward-fill missing tyre data
    - Drop laps with no timing data
    """
    df = laps.copy()

    # Convert LapTime to seconds
    if "LapTime" in df.columns:
        df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    # Forward-fill missing tyre info
    tyre_cols = ["Compound", "TyreLife"]
    for col in tyre_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Drop rows where we have no lap time at all
    df = df.dropna(subset=["LapTimeSeconds"])

    return df


def get_driver_laps(laps: pd.DataFrame, driver: str = REFERENCE_DRIVER) -> pd.DataFrame:
    """Filter laps to a single driver."""
    return laps[laps["Driver"] == driver].copy().reset_index(drop=True)


def get_race_data(year: int = RACE_YEAR, gp: str = RACE_GP, driver: str = REFERENCE_DRIVER) -> pd.DataFrame:
    """
    One-call convenience: load session → clean laps → filter to driver.
    
    Returns:
        Cleaned DataFrame of race laps for the given driver.
    """
    session = load_session(year, gp)
    laps = clean_laps(session.laps)
    driver_laps = get_driver_laps(laps, driver)
    return driver_laps


def get_all_drivers_laps(year: int = RACE_YEAR, gp: str = RACE_GP, drivers: list = None) -> pd.DataFrame:
    """
    Load cleaned laps for ALL drivers (or a selected subset) — needed for tyre model calibration.
    
    Returns:
        Cleaned DataFrame of race laps.
    """
    session = load_session(year, gp)
    laps = clean_laps(session.laps)
    if drivers:
        laps = laps[laps["Driver"].isin(drivers)].copy().reset_index(drop=True)
    return laps


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading {RACE_YEAR} {RACE_GP} GP data for {REFERENCE_DRIVER}...")
    driver_laps = get_race_data()
    print(f"\n✓ Loaded {len(driver_laps)} laps")
    print(f"\nColumns: {list(driver_laps.columns)}")
    print(f"\nCompounds used: {driver_laps['Compound'].unique()}")
    print(f"\nSample (first 5 laps):")
    print(driver_laps[["LapNumber", "Compound", "TyreLife", "LapTimeSeconds", "Position"]].head())
    print(f"\nLap time range: {driver_laps['LapTimeSeconds'].min():.1f}s — {driver_laps['LapTimeSeconds'].max():.1f}s")
