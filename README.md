# StintEngine

> **A Reinforcement Learning agent that learns F1 pit stop strategy from real telemetry data.**

StintEngine trains a PPO agent against a physics-grounded Formula 1 race simulation — fuel burn, tyre degradation curves fitted from real FastF1 data, stochastic safety car events — and exposes the full training loop through a live web dashboard with SSE-streamed telemetry.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Environment Design](#environment-design)
- [Tyre Degradation Model](#tyre-degradation-model)
- [Training Pipeline](#training-pipeline)
- [Dashboard](#dashboard)
- [Getting Started](#getting-started)
- [Docker](#docker)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Pit wall strategy is one of the few remaining domains in Formula 1 where human intuition still competes with data. A suboptimal tyre call — staying out one lap too long, pitting under a virtual safety car unnecessarily — can cost podiums.

StintEngine models this decision as a Markov Decision Process and trains a Proximal Policy Optimization agent to make pit/stay decisions lap-by-lap, using an 8-dimensional observation vector derived from actual race telemetry loaded via FastF1. The agent learns compound-specific degradation curves, fuel-adjusted lap time penalties, and the expected value of pitting under different track conditions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        StintEngine                          │
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│  │  FastF1 Data │────▶│  Tyre Model  │────▶│  F1StrategyEnv │
│  │  Loader      │     │  (Poly Fit)  │     │  (Gymnasium)│  │
│  └──────────────┘     └──────────────┘     └─────┬──────┘  │
│                                                   │         │
│                                            ┌──────▼──────┐  │
│                                            │  PPO Agent  │  │
│                                            │  (SB3)      │  │
│                                            └──────┬──────┘  │
│                                                   │         │
│  ┌────────────────────────────────────────────────▼──────┐  │
│  │         Flask Backend  (REST API + SSE Stream)        │  │
│  └────────────────────────────────────────────────┬──────┘  │
│                                                   │         │
│  ┌────────────────────────────────────────────────▼──────┐  │
│  │              Web Dashboard  (Vanilla JS)              │  │
│  │   Command Center · Learning Curve · Strategy Timeline │  │
│  │   Tyre Degradation · Telemetry Feed                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Environment Design

The `F1StrategyEnv` is a custom [Gymnasium](https://gymnasium.farama.org/) environment that simulates a full race distance.

### Observation Space (8-dimensional, normalized)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `lap_number` | Current lap / total laps |
| 1 | `tyre_life` | Laps on current set (normalized) |
| 2 | `compound` | One-hot encoded: SOFT=0, MEDIUM=1, HARD=2 |
| 3 | `fuel_load` | Remaining fuel fraction (affects lap time) |
| 4 | `cold_tyre_penalty` | Active in first N laps after a pit stop |
| 5 | `safety_car` | Boolean — SC/VSC on track this lap |
| 6 | `gap_to_leader` | Relative position proxy |
| 7 | `position` | Current running order (normalized) |

### Action Space

Binary: `0 = Stay Out`, `1 = Pit`

### Reward Signal

The reward function is shaped around lap time minimisation with strategic bonuses:

- **Negative reward**: proportional to simulated lap time (fuel-corrected, tyre-degraded)
- **Pit bonus**: reward for pitting under safety car conditions
- **Stint length penalty**: disincentivises unnecessarily long stints on worn rubber
- **Terminal reward**: based on finishing position relative to no-strategy baseline

---

## Tyre Degradation Model

Tyre degradation is not assumed — it is **fitted from real F1 telemetry**.

`data_loader.py` pulls multi-driver lap data for a reference race via FastF1. `tyre_model.py` fits compound-specific polynomial degradation curves mapping tyre age to lap time delta. These coefficients are serialised and reloaded at environment init, ensuring the simulation reflects actual observed degradation rates rather than hand-tuned constants.

```
LapTimeDelta(age) = a·age² + b·age + c     (per compound)
```

Curves are plotted during the `viz` pipeline and visible in the dashboard's Tyre Degradation panel.

---

## Training Pipeline

StintEngine uses a two-phase training schedule:

| Phase | Timesteps | Model |
|-------|-----------|-------|
| Initial | 50,000 | `models/ppo_f1_initial` |
| Final | 200,000 | `models/ppo_f1_final` (resumes from initial) |

**Hardware**: Training auto-detects CUDA → MPS → CPU. The `DEVICE` flag in `config.py` is passed through to PyTorch and Stable-Baselines3. GPU pass-through is supported in Docker via the NVIDIA container runtime.

**Metrics callback**: A custom SB3 callback streams episode reward, mean lap time, and strategy decisions to the Flask backend over an internal queue, which the dashboard consumes via SSE in real time.

---

## Dashboard

Launch with `python main.py dashboard` and open `http://localhost:5000`.

| Panel | Description |
|-------|-------------|
| **Command Center** | Trigger train / inference runs; live status feed |
| **Learning Curve** | Episode reward over training steps (Chart.js, SSE-updated) |
| **Strategy Timeline** | Compound stint bars — visualise the agent's pit decisions per race |
| **Tyre Degradation** | Fitted polynomial curves per compound overlaid on raw telemetry scatter |
| **Telemetry Feed** | Lap-by-lap table: compound, tyre age, lap time, position, fuel |

All panels update in real time via **Server-Sent Events** — no polling, no WebSocket overhead.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/const-nishant/StintEngine.git
cd StintEngine

python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### Run the Dashboard

```bash
python main.py dashboard
# → http://127.0.0.1:5000
```

### Full Pipeline (data → train → visualise)

```bash
# Step 1: Load race data and fit tyre model
python main.py data

# Step 2: Initial training run (50k steps)
python main.py train

# Step 3: Final training run (200k steps, resumes from initial)
python main.py train-final

# Step 4: Run trained agent through a full race
python main.py infer

# Step 5: Generate comparison plots
python main.py viz

# Or run everything at once
python main.py all
```

---

## Docker

The project ships a `docker-compose.yml` with optional NVIDIA GPU pass-through. Volumes are mounted for `cache/`, `models/`, `logs/`, and `plots/` so trained weights and FastF1 data persist across container rebuilds.

```bash
# CPU / MPS
docker compose up --build

# With GPU (requires nvidia-container-toolkit on host)
docker compose up --build   # GPU reservation declared in compose; auto-skipped if unavailable
```

Dashboard available at `http://localhost:5000`.

---

## CLI Reference

```
python main.py <command> [args]

Commands:
  data           Load race telemetry and fit tyre degradation model
  train          Train PPO agent — 50k steps (override: python main.py train 100000)
  train-final    Train final model — 200k steps, resumes from initial checkpoint
  infer          Run trained agent through a full simulated race
  viz            Generate matplotlib/plotly comparison plots → plots/
  dashboard      Launch Flask dashboard with SSE streaming
  all            Full pipeline: data → train → train-final → viz
```

---

## Project Structure

```
StintEngine/
├── src/
│   ├── config.py          # Centralised hyperparameters and paths
│   ├── env.py             # F1StrategyEnv (Gymnasium, 8D obs, binary action)
│   ├── train.py           # PPO training loop + metrics streaming callback
│   ├── data_loader.py     # FastF1 race data ingestion
│   ├── tyre_model.py      # Polynomial degradation curve fitting
│   └── visualize.py       # Matplotlib / Plotly plot generation
├── app.py                 # Flask backend — REST API + SSE endpoint
├── templates/             # Jinja2 HTML (Flask-served dashboard)
├── static/                # Dashboard CSS + JS
├── frontend/              # Standalone static build (GitHub Pages)
├── .agents/workflows/     # Agent workflow definitions
├── .github/workflows/     # CI/CD workflows
├── main.py                # CLI entry point
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Reinforcement Learning | Stable-Baselines3 (PPO), Gymnasium |
| Race Data | FastF1 |
| Backend | Flask, Server-Sent Events |
| ML Runtime | PyTorch (CUDA / MPS / CPU auto-detect) |
| Visualisation | Matplotlib, Plotly, Chart.js |
| Containerisation | Docker, Docker Compose (NVIDIA GPU pass-through) |
| Frontend | Vanilla JS, Chart.js |

---

## License

[MIT](./LICENSE) © Nishant Patil
