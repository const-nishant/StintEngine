# 🏎️ StintEngine

**F1 Pit Stop Strategy Reinforcement Learning Agent** — Train a PPO agent to make optimal tyre strategy decisions using real F1 race data.

## Features

- **8D Observation Space** — Fuel load, safety car, gap-to-leader, cold tyre penalty
- **Advanced Simulation** — Fuel burn model, safety car events, tyre degradation curves from FastF1
- **Real-Time Web Dashboard** — Live training metrics, strategy timeline, lap-by-lap telemetry
- **SSE Streaming** — Zero-latency updates via Server-Sent Events

## Quick Start

```bash
# Create venv & install
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# Launch dashboard
python main.py dashboard
# → http://127.0.0.1:5000

# Or use CLI
python main.py train         # 50K training
python main.py train-final   # 200K final
python main.py infer         # Run agent race
python main.py viz           # Generate plots
```

## Architecture

```
StintEngine/
├── src/
│   ├── config.py        # All parameters in one place
│   ├── env.py           # Gymnasium F1StrategyEnv (8D obs)
│   ├── train.py         # PPO training + metrics callback
│   ├── data_loader.py   # FastF1 race data loading
│   ├── tyre_model.py    # Polynomial degradation fitting
│   └── visualize.py     # Matplotlib/Plotly charts
├── app.py               # Flask backend (REST API + SSE)
├── templates/            # Dashboard HTML (Flask-served)
├── static/              # Dashboard CSS/JS (Flask-served)
├── frontend/            # Standalone static site (GitHub Pages)
├── main.py              # CLI entry point
└── requirements.txt
```

## Dashboard Panels

| Panel | Description |
|---|---|
| **01 Command Center** | Train/Run Race buttons + live metrics |
| **02 Learning Curve** | Real-time reward chart |
| **03 Strategy Timeline** | Compound stint bars |
| **04 Tyre Degradation** | Fitted degradation curves |
| **05 Telemetry Feed** | Lap-by-lap race data |

## Tech Stack

- **RL**: Stable-Baselines3 PPO
- **Data**: FastF1 (real F1 telemetry)
- **Backend**: Flask + SSE
- **Frontend**: Vanilla JS + Chart.js

## License

This project is licensed under the [MIT License](LICENSE).
