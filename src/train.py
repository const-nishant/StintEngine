"""
StintEngine — RL Training Script (GPU-Accelerated)
Train a PPO agent on the F1StrategyEnv using CUDA if available.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from src.env import F1StrategyEnv
from src.data_loader import get_all_drivers_laps
from src.tyre_model import fit_tyre_degradation
from src.config import (
    PPO_LEARNING_RATE, PPO_N_ENVS, PPO_POLICY,
    PPO_INITIAL_TIMESTEPS, PPO_FINAL_TIMESTEPS,
    MODEL_INITIAL, MODEL_FINAL, LOGS_DIR, DEVICE,
)


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback that logs training metrics to a JSON file
    for the web dashboard to poll.
    """

    def __init__(self, log_path: Path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.metrics = {
            "status": "training",
            "start_time": None,
            "total_timesteps": 0,
            "current_timestep": 0,
            "episodes": 0,
            "mean_reward": 0.0,
            "reward_history": [],
            "episode_lengths": [],
            "device": DEVICE,
        }
        self.episode_rewards = []

    def _on_training_start(self):
        self.metrics["start_time"] = time.time()
        self.metrics["total_timesteps"] = self.model._total_timesteps
        self._save()

    def _on_step(self) -> bool:
        self.metrics["current_timestep"] = self.num_timesteps

        # Check for episode completion
        if len(self.model.ep_info_buffer) > 0:
            recent = list(self.model.ep_info_buffer)
            if recent:
                last = recent[-1]
                self.metrics["episodes"] = len(self.episode_rewards) + 1
                ep_reward = last.get("r", 0)
                self.episode_rewards.append(ep_reward)
                self.metrics["mean_reward"] = round(
                    sum(self.episode_rewards[-50:]) / max(len(self.episode_rewards[-50:]), 1), 2
                )
                # Store sampled reward history (every 10th episode to keep JSON small)
                if len(self.episode_rewards) % 10 == 0:
                    self.metrics["reward_history"].append({
                        "episode": len(self.episode_rewards),
                        "reward": round(ep_reward, 2),
                        "mean_reward": self.metrics["mean_reward"],
                        "timestep": self.num_timesteps,
                    })

        # Save metrics every 2048 steps
        if self.num_timesteps % 2048 == 0:
            self._save()

        return True

    def _on_training_end(self):
        self.metrics["status"] = "complete"
        self.metrics["end_time"] = time.time()
        elapsed = self.metrics["end_time"] - self.metrics["start_time"]
        self.metrics["elapsed_seconds"] = round(elapsed, 1)
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.metrics, f, indent=2)


def load_tyre_coefficients():
    """Load race data and fit tyre degradation model."""
    print("═" * 60)
    print("  STINT ENGINE — Loading race data & fitting tyre model")
    print(f"  Device: {DEVICE.upper()}")
    print("═" * 60)

    all_laps = get_all_drivers_laps()
    print(f"\n  Loaded {len(all_laps)} laps from all drivers\n")

    coeffs = fit_tyre_degradation(all_laps)
    return coeffs


def make_env_factory(tyre_coefficients):
    """Return a factory function that creates F1StrategyEnv with fitted coefficients."""
    def _make():
        return F1StrategyEnv(tyre_coefficients=tyre_coefficients)
    return _make


def train(timesteps: int, model_path: Path, tyre_coefficients: dict,
          resume_from: Path = None, callback_cls=None):
    """Train (or continue training) a PPO agent with GPU acceleration."""
    print(f"\n{'═' * 60}")
    print(f"  TRAINING — {timesteps:,} timesteps on {DEVICE.upper()}")
    print(f"  Envs: {PPO_N_ENVS} parallel | Saving to: {model_path}")
    print(f"{'═' * 60}\n")

    env = make_vec_env(
        make_env_factory(tyre_coefficients),
        n_envs=PPO_N_ENVS,
    )

    metrics_path = LOGS_DIR / "training_metrics.json"
    CallbackClass = callback_cls or TrainingMetricsCallback
    callback = CallbackClass(log_path=metrics_path)

    if resume_from and resume_from.with_suffix(".zip").exists():
        print(f"  Resuming from: {resume_from}")
        model = PPO.load(str(resume_from), env=env, device=DEVICE)
    else:
        model = PPO(
            PPO_POLICY,
            env,
            verbose=1,
            learning_rate=PPO_LEARNING_RATE,
            tensorboard_log=str(LOGS_DIR),
            device=DEVICE,
            n_steps=2048,
            batch_size=128,  # larger batch for GPU
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    model.learn(
        total_timesteps=timesteps,
        tb_log_name="f1_strategy",
        callback=callback,
        progress_bar=True,
    )

    model.save(str(model_path))
    print(f"\n  ✓ Model saved to: {model_path}")
    print(f"  ✓ Metrics saved to: {metrics_path}")

    env.close()
    return model


def run_inference(model_path: Path, tyre_coefficients: dict) -> dict:
    """Run one full race episode and return the race log."""
    print(f"\n{'═' * 60}")
    print(f"  INFERENCE — Running agent race")
    print(f"{'═' * 60}\n")

    model = PPO.load(str(model_path), device=DEVICE)
    env = F1StrategyEnv(tyre_coefficients=tyre_coefficients)

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    action_names = ["STAY OUT", "PIT SOFT", "PIT MED ", "PIT HARD"]

    print(f"  {'Lap':>4}  {'Action':<10}  {'Comp':<6}  {'TAge':>4}  {'Pos':>3}  "
          f"{'Fuel':>6}  {'Gap':>6}  {'LapT':>7}  {'SC':<3}  {'Rew':>6}")
    print(f"  {'─' * 72}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

        sc = "🟡" if info["safety_car"] else "  "
        pit = " ◄" if action > 0 else "  "
        print(f"  {info['lap']:>4}  {action_names[action]:<10}  {info['compound']:<6}  "
              f"{info['tyre_age']:>4}  {info['position']:>3}  "
              f"{info['fuel_kg']:>5.1f}  {info['gap_to_leader']:>5.1f}s  "
              f"{info['laptime']:>7.2f}  {sc:<3}  {reward:>+5.1f}{pit}")

    print(f"\n  ═══ RACE RESULT ═══")
    print(f"  Final Position: P{info['position']}")
    print(f"  Pit Stops: {info['pit_stops']} (laps: {info['pit_laps']})")
    print(f"  Total Reward: {total_reward:.2f}")

    # Save race log for UI
    race_log_path = LOGS_DIR / "last_race.json"
    with open(race_log_path, "w") as f:
        json.dump({"laps": env.race_log, "total_reward": total_reward}, f, indent=2)
    print(f"  ✓ Race log saved to: {race_log_path}")

    return {"race_log": env.race_log, "total_reward": total_reward, "info": info}


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StintEngine — Train F1 Strategy Agent")
    parser.add_argument("--timesteps", type=int, default=PPO_INITIAL_TIMESTEPS,
                        help=f"Training timesteps (default: {PPO_INITIAL_TIMESTEPS:,})")
    parser.add_argument("--model", type=str, default=str(MODEL_INITIAL),
                        help="Model save path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from saved model")
    parser.add_argument("--inference", action="store_true",
                        help="Run inference instead of training")
    parser.add_argument("--final", action="store_true",
                        help="Run final 200k training")

    args = parser.parse_args()
    coeffs = load_tyre_coefficients()

    if args.inference:
        run_inference(Path(args.model), coeffs)
    elif args.final:
        train(PPO_FINAL_TIMESTEPS, MODEL_FINAL, coeffs, resume_from=MODEL_INITIAL)
    else:
        train(args.timesteps, Path(args.model), coeffs,
              resume_from=Path(args.resume) if args.resume else None)
