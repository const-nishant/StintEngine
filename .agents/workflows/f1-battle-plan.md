---
description: How to build the F1 Pit Stop Strategy RL Agent following the battle plan
---

# F1 Pit Stop Strategy Agent — Build Workflow

> **Reference:** [battle-plan.html](file:///p:/vs-code/projects/StintEngine/docs/battle-plan.html)
> **Stack:** Python · FastF1 · Gymnasium · stable-baselines3 (PPO) · PyTorch (CPU)
> **Estimated Time:** 8 hours (single-day build)

---

## PHASE 01 — Setup & Data Pull

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
// turbo
```bash
pip install fastf1 gymnasium stable-baselines3[extra] pandas numpy matplotlib plotly
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> [!IMPORTANT]
> Pin `gymnasium==0.29.1` — sb3 and gym version conflicts are common.

### 3. Pull race data with FastF1
- Enable cache: `fastf1.Cache.enable_cache('./cache')`
- Load session: `fastf1.get_session(2023, 'Bahrain', 'R')`
- Call `session.load()` — takes ~2 min first time
- Pull `session.laps` into a DataFrame

### 4. Explore & clean the data
- Key columns: `LapNumber, Compound, TyreLife, LapTime, Position, Driver`
- Filter to one driver first (Verstappen = `VER`)
- Convert `LapTime` from timedelta → seconds float
- Fill forward any null tyre data
- Plot lap time vs tyre age per compound to see degradation curves

---

## PHASE 02 — Build the Gym Environment

### 5. Define state & action spaces
- **State (observation):** `[current_lap, tyre_age, compound_encoded, lap_time_delta, position]`
- **Actions:** `0 = Stay Out, 1 = Pit Soft, 2 = Pit Medium, 3 = Pit Hard`
- Use `gymnasium.spaces.Box` for observations, `Discrete(4)` for actions
- Normalize all observation values to `[0, 1]`

### 6. Write `F1StrategyEnv(gymnasium.Env)`
- `reset()` — start lap 1, fresh soft tyres, position 10
- `step(action)` — advance one lap, apply tyre degradation, update position
- If action = pit: add 20–25s pit stop time, reset tyre age to 0, change compound
- Tyre deg model: `base_deg * (1 + 0.03 * tyre_age)` added to lap time each lap
- Reward = position gained this lap (negative = lost positions)
- Episode ends at lap 57 (Bahrain race length)

### 7. Calibrate tyre degradation from real data
- Group laps by Compound → compute mean lap time per TyreAge bucket
- Fit regression: `laptime = a + b*tyre_age + c*tyre_age²`
- Store coefficients per compound (Soft / Medium / Hard)
- Use these inside `step()` — soft should degrade faster than hard

> [!TIP]
> Start with just 1 compound (Medium only) to validate the env, then add others.

---

## PHASE 03 — Train the RL Agent

### 8. Validate the environment
```python
from gymnasium.utils.env_checker import check_env
check_env(env)  # fixes 90% of bugs
```
- Run a random policy loop: `env.step(env.action_space.sample())`
- Print state & reward every lap — check they make sense
- Verify episode terminates at lap 57

### 9. Set up PPO training
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env(F1StrategyEnv, n_envs=4)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=50_000)
model.save("f1_strategy_agent")
```

### 10. Tune the reward function
- Agent never pits? → Add tyre cliff penalty (if `tyre_age > 25`, big negative reward)
- Agent pits every lap? → Add pit cost (−25s equivalent in reward units)
- End-of-race bonus: `+50 * (20 - final_position)`
- Retrain after each reward change (~3 min per run)
- **Target:** 2–3 pit stops total across 57 laps

### 11. Train final model — 200k timesteps
```python
model.learn(total_timesteps=200_000)
model.save("f1_strategy_agent_final")
```
- Use TensorBoard callback to log reward curves
- ~10 min on CPU

> [!WARNING]
> If reward doesn't improve after 50k steps, your reward scale is off — normalize it.

---

## PHASE 04 — Visualize & Compare

### 12. Run agent & record decisions
- Load: `model = PPO.load("f1_strategy_agent_final")`
- Run one full episode, log: lap number, action, compound, tyre age, position
- Build a DataFrame of the agent's race
- Store real Verstappen strategy from FastF1 data for comparison

### 13. Build 3 key plots
1. **Strategy Timeline** — Horizontal bar per compound stint, Agent vs. Real (red=soft, yellow=medium, white=hard)
2. **Position Over Laps** — Line chart, Agent vs. Real, pit lap markers as vertical dashed lines
3. **Reward Curve** — Training reward over timesteps (shows when agent "figured it out")

### 14. Bonus: test on a different race
- Pull a second race session (Monaco GP or Monza GP 2023)
- Recalibrate tyre deg coefficients for new circuit
- Run agent on new env without retraining — does it generalize?

---

## End of Day Deliverables

| Deliverable | Description |
|---|---|
| 🤖 Trained RL Agent | PPO model that makes pit decisions across a 57-lap race |
| 🏎️ F1 Simulation Env | Gym env calibrated to actual 2023 Bahrain tyre data |
| 📊 Strategy Comparison Plots | Agent vs. real team strategy visualized side-by-side |
| 📈 Learning Curve | Reward progression showing when agent discovered the undercut |

---

## Survival Tips

- If FastF1 download stalls → `fastf1.Cache.enable_cache()` before any session load
- Pin `gymnasium==0.29.1`
- Use `n_envs=4` in `make_vec_env` → 4× faster training on CPU
- Undercut discovered = agent pits 1–2 laps before opponent and gains position
