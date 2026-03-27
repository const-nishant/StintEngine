"""
StintEngine — Central Configuration
All race parameters, paths, and hyperparameters in one place.
"""

import torch
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

# Create dirs on import
for d in [CACHE_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Device (GPU if available) ───────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ─── Race Parameters ─────────────────────────────────────────────────────────
RACE_YEAR = 2023
RACE_GP = "Bahrain"
RACE_SESSION = "R"
TOTAL_LAPS = 57
NUM_DRIVERS = 20
REFERENCE_DRIVER = "VER"  # Verstappen — baseline for comparison

# ─── Tyre Compounds (dry + wet) ──────────────────────────────────────────────
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTER", "WET"]
COMPOUND_TO_IDX = {c: i for i, c in enumerate(COMPOUNDS)}
IDX_TO_COMPOUND = {i: c for i, c in enumerate(COMPOUNDS)}
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTER", "WET"]

# ─── Environment Parameters ──────────────────────────────────────────────────
STARTING_COMPOUND = "SOFT"
STARTING_POSITION = 10
PIT_STOP_TIME_LOSS = 22.0       # seconds lost per pit stop (stationary + in/out laps)
MAX_TYRE_AGE = 40               # normalizer for tyre age in observation space
TYRE_CLIFF_AGE = 25             # after this age, heavy penalty kicks in

# ─── Advanced Sim Parameters ─────────────────────────────────────────────────
FUEL_LOAD_KG = 110.0            # starting fuel in kg
FUEL_BURN_PER_LAP = 1.85        # kg burned per lap
FUEL_TIME_PENALTY = 0.035       # seconds per kg of fuel onboard
COLD_TYRE_PENALTY = 2.0         # seconds lost on out-lap after pit
COLD_TYRE_WARMUP_LAPS = 2       # laps to fully warm up new tyres
SAFETY_CAR_PROBABILITY = 0.04   # ~4% chance per lap of SC deployment
SAFETY_CAR_MIN_LAPS = 3         # minimum SC duration
SAFETY_CAR_MAX_LAPS = 7         # maximum SC duration
SAFETY_CAR_LAPTIME = 120.0      # lap time under safety car (seconds)
GAP_NORMALIZER = 60.0           # max gap in seconds for normalization

# ─── Weather System ──────────────────────────────────────────────────────────
RAIN_PROBABILITY = 0.06         # ~6% chance per lap of rain starting
RAIN_MIN_LAPS = 4               # minimum rain duration
RAIN_MAX_LAPS = 12              # maximum rain duration
RAIN_INTENSITY_LEVELS = 3       # 0=dry, 1=light, 2=heavy
DRY_ON_WET_PENALTY = 8.0        # seconds/lap penalty for dry tyres in rain
WET_ON_DRY_PENALTY = 5.0        # seconds/lap penalty for wet tyres on dry
INTER_RAIN_BONUS = -1.5         # inters are optimal in light rain (negative=faster)
WET_HEAVY_RAIN_BONUS = -2.0     # full wets are optimal in heavy rain

# ─── Reward Shaping ──────────────────────────────────────────────────────────
REWARD_POSITION_GAIN = 1.0      # reward per position gained
REWARD_PIT_COST = -2.0          # penalty for making a pit stop
REWARD_TYRE_CLIFF_PENALTY = -3.0  # penalty per lap beyond tyre cliff age
REWARD_FINISH_BONUS_SCALE = 50.0  # end-of-race bonus: scale * (20 - final_pos)
REWARD_SC_PIT_BONUS = 1.5       # bonus for pitting during safety car (free stop)
REWARD_WRONG_TYRE_PENALTY = -4.0  # penalty for wrong compound in weather

# ─── PPO Hyperparameters ─────────────────────────────────────────────────────
import os
PPO_LEARNING_RATE = 3e-4
PPO_POLICY = "MlpPolicy"
PPO_INITIAL_TIMESTEPS = 50_000
PPO_FINAL_TIMESTEPS = 200_000

# Hardware configurations
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable aggressive CUDA JIT optimizations
    PPO_N_ENVS = 16                        # High parallelism for heavy GPU throughput
    PPO_BATCH_SIZE = 256
    PPO_N_STEPS = 2048
elif DEVICE == "mps":
    PPO_N_ENVS = 8                         # Balanced Apple Silicon scaling to avoid dispatch overhead
    PPO_BATCH_SIZE = 128
    PPO_N_STEPS = 1024
else:
    PPO_N_ENVS = max(1, os.cpu_count() or 4) # CPU scaling uses available cores
    PPO_BATCH_SIZE = 64                      # Smaller batches reduce memory latency on CPU
    PPO_N_STEPS = 512

# ─── Model Paths ─────────────────────────────────────────────────────────────
MODEL_INITIAL = MODELS_DIR / "f1_strategy_agent"
MODEL_FINAL = MODELS_DIR / "f1_strategy_agent_final"

# ─── Web Dashboard ───────────────────────────────────────────────────────────
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
