"""
StintEngine — Visualization (Advanced)
Generate comparison plots: Agent vs Real F1 Strategy with fuel & SC data.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

from stable_baselines3 import PPO

from src.env import F1StrategyEnv
from src.data_loader import get_race_data, get_all_drivers_laps
from src.tyre_model import fit_tyre_degradation
from src.config import (
    TOTAL_LAPS, REFERENCE_DRIVER, RACE_YEAR, RACE_GP,
    MODEL_FINAL, MODEL_INITIAL, PLOTS_DIR, DEVICE,
    COMPOUND_TO_IDX, COMPOUNDS,
)

COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0096FF",
}


def run_agent_race(model_path: Path, tyre_coefficients: dict) -> pd.DataFrame:
    """Run a full race with the trained agent and record every lap."""
    model = PPO.load(str(model_path), device=DEVICE)
    env = F1StrategyEnv(tyre_coefficients=tyre_coefficients)

    obs, info = env.reset()
    records = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        records.append({
            "LapNumber": info["lap"],
            "Compound": info["compound"],
            "TyreAge": info["tyre_age"],
            "Position": info["position"],
            "LapTime": info["laptime"],
            "FuelKg": info["fuel_kg"],
            "GapToLeader": info["gap_to_leader"],
            "SafetyCar": info["safety_car"],
            "Action": int(action),
            "PitStop": int(action) > 0,
        })

    return pd.DataFrame(records)


def get_real_strategy(driver_laps: pd.DataFrame) -> pd.DataFrame:
    df = driver_laps[["LapNumber", "Compound", "TyreLife", "Position", "LapTimeSeconds"]].copy()
    df = df.rename(columns={"TyreLife": "TyreAge", "LapTimeSeconds": "LapTime"})
    df["PitStop"] = df["Compound"] != df["Compound"].shift(1)
    df.loc[df.index[0], "PitStop"] = False
    return df


def extract_stints(df: pd.DataFrame) -> list:
    stints = []
    current_compound = df.iloc[0]["Compound"]
    start_lap = int(df.iloc[0]["LapNumber"])
    for _, row in df.iterrows():
        if row["Compound"] != current_compound:
            stints.append((start_lap, int(row["LapNumber"]) - 1, current_compound))
            current_compound = row["Compound"]
            start_lap = int(row["LapNumber"])
    stints.append((start_lap, int(df.iloc[-1]["LapNumber"]), current_compound))
    return stints


def plot_strategy_comparison(agent_df, real_df, save_path):
    fig, ax = plt.subplots(figsize=(14, 3.5), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    for label, stints, y in [
        ("RL Agent", extract_stints(agent_df), 1.5),
        (f"Real ({REFERENCE_DRIVER})", extract_stints(real_df), 0.5),
    ]:
        for start, end, compound in stints:
            color = COMPOUND_COLORS.get(compound, "#666")
            width = end - start + 1
            ax.barh(y, width, left=start - 0.5, height=0.6,
                    color=color, edgecolor="#333", linewidth=0.5, alpha=0.9)
            mid = start + width / 2
            text_color = "#000" if compound in ["MEDIUM", "HARD"] else "#fff"
            ax.text(mid, y, f"{compound[0]}\n{width}L",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=text_color, fontfamily="monospace")

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([f"Real ({REFERENCE_DRIVER})", "RL Agent"], color="#bbb", fontsize=11)
    ax.set_xlabel("Lap Number", color="#888", fontsize=10)
    ax.set_xlim(0, TOTAL_LAPS + 1)
    ax.set_ylim(0, 2.2)
    ax.tick_params(colors="#555")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]: ax.spines[spine].set_color("#333")
    ax.set_title("STRATEGY TIMELINE — Agent vs Real", color="#E8002D",
                 fontsize=14, fontweight="bold", pad=12, fontfamily="monospace")
    patches = [mpatches.Patch(facecolor=COMPOUND_COLORS[c], edgecolor="#333", label=c) for c in ["SOFT", "MEDIUM", "HARD"]]
    ax.legend(handles=patches, loc="upper right", fontsize=8, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#bbb")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_position_over_laps(agent_df, real_df, save_path):
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    ax.plot(agent_df["LapNumber"], agent_df["Position"], color="#00e676", linewidth=1.8, label="RL Agent", alpha=0.9)
    ax.plot(real_df["LapNumber"], real_df["Position"], color="#2979FF", linewidth=1.8, label=f"Real ({REFERENCE_DRIVER})", alpha=0.9)

    # SC highlight zones on agent trace
    if "SafetyCar" in agent_df.columns:
        sc_mask = agent_df["SafetyCar"] == True
        if sc_mask.any():
            for _, row in agent_df[sc_mask].iterrows():
                ax.axvspan(row["LapNumber"] - 0.5, row["LapNumber"] + 0.5, alpha=0.08, color="#FFD600")

    for df, color, label in [(agent_df, "#00e676", "Agent"), (real_df, "#2979FF", REFERENCE_DRIVER)]:
        pits = df[df["PitStop"]]
        if len(pits):
            for _, pit in pits.iterrows():
                ax.axvline(pit["LapNumber"], color=color, linestyle="--", alpha=0.3, linewidth=0.8)
            ax.scatter(pits["LapNumber"], pits["Position"], color=color, marker="v", s=60, zorder=5, label=f"{label} Pit")

    ax.set_xlabel("Lap Number", color="#888", fontsize=10)
    ax.set_ylabel("Position", color="#888", fontsize=10)
    ax.set_xlim(1, TOTAL_LAPS)
    ax.set_ylim(20, 0)
    ax.tick_params(colors="#555")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]: ax.spines[spine].set_color("#333")
    ax.grid(axis="y", color="#1a1a1a", linewidth=0.5)
    ax.set_title("POSITION OVER LAPS", color="#E8002D", fontsize=14, fontweight="bold", pad=12, fontfamily="monospace")
    ax.legend(loc="lower right", fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#bbb")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_fuel_and_gap(agent_df, save_path):
    """Plot fuel load and gap-to-leader over laps."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), facecolor="#0a0a0a", sharex=True)

    for ax in [ax1, ax2]:
        ax.set_facecolor("#0a0a0a")
        for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]: ax.spines[spine].set_color("#333")
        ax.tick_params(colors="#555")

    ax1.plot(agent_df["LapNumber"], agent_df["FuelKg"], color="#ff7043", linewidth=1.5)
    ax1.fill_between(agent_df["LapNumber"], 0, agent_df["FuelKg"], alpha=0.1, color="#ff7043")
    ax1.set_ylabel("Fuel (kg)", color="#888", fontsize=10)
    ax1.set_title("FUEL LOAD & GAP TO LEADER", color="#E8002D", fontsize=14, fontweight="bold", pad=12, fontfamily="monospace")

    ax2.plot(agent_df["LapNumber"], agent_df["GapToLeader"], color="#2979FF", linewidth=1.5)
    ax2.fill_between(agent_df["LapNumber"], 0, agent_df["GapToLeader"], alpha=0.1, color="#2979FF")
    ax2.set_xlabel("Lap Number", color="#888", fontsize=10)
    ax2.set_ylabel("Gap to Leader (s)", color="#888", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_tyre_degradation_curves(tyre_coefficients, save_path):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    ages = np.arange(1, 36)
    for compound in COMPOUNDS:
        a, b, c = tyre_coefficients[compound]
        times = a + b * ages + c * ages ** 2
        color = COMPOUND_COLORS.get(compound, "#888")
        ax.plot(ages, times, color=color, linewidth=2, label=compound, alpha=0.9)
    ax.set_xlabel("Tyre Age (laps)", color="#888", fontsize=10)
    ax.set_ylabel("Predicted Lap Time (s)", color="#888", fontsize=10)
    ax.tick_params(colors="#555")
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]: ax.spines[spine].set_color("#333")
    ax.grid(axis="both", color="#1a1a1a", linewidth=0.5)
    ax.set_title("TYRE DEGRADATION MODEL", color="#E8002D", fontsize=14, fontweight="bold", pad=12, fontfamily="monospace")
    ax.legend(loc="upper left", fontsize=10, facecolor="#1a1a1a", edgecolor="#333", labelcolor="#bbb")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def generate_all_plots():
    print("═" * 60)
    print("  STINT ENGINE — Generating Visualizations")
    print("═" * 60)

    all_laps = get_all_drivers_laps()
    coeffs = fit_tyre_degradation(all_laps)

    model_path = MODEL_FINAL if MODEL_FINAL.with_suffix(".zip").exists() else MODEL_INITIAL
    print(f"\n  Loading model: {model_path}")
    agent_df = run_agent_race(model_path, coeffs)

    driver_laps = get_race_data()
    real_df = get_real_strategy(driver_laps)

    print(f"\n  Generating plots → {PLOTS_DIR}/\n")
    plot_strategy_comparison(agent_df, real_df, PLOTS_DIR / "strategy_comparison.png")
    plot_position_over_laps(agent_df, real_df, PLOTS_DIR / "position_over_laps.png")
    plot_fuel_and_gap(agent_df, PLOTS_DIR / "fuel_and_gap.png")
    plot_tyre_degradation_curves(coeffs, PLOTS_DIR / "tyre_degradation.png")

    print(f"\n  ✓ All plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
