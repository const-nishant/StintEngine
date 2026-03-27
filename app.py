"""
StintEngine — Web Dashboard (Flask Backend) v2
REST API + SSE streaming + race history + config tuning.
"""

import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_from_directory, Response

from src.config import (
    FLASK_HOST, FLASK_PORT, MODELS_DIR, LOGS_DIR, PLOTS_DIR,
    MODEL_INITIAL, MODEL_FINAL, DEVICE,
    PPO_INITIAL_TIMESTEPS, PPO_FINAL_TIMESTEPS,
    TEMPLATES_DIR, STATIC_DIR, COMPOUNDS,
    STARTING_POSITION,
)

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

# ─── CORS (allows GitHub Pages frontend to connect) ──────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ─── Thread-safe state ───────────────────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "is_training": False,
    "is_inferring": False,
}

# SSE event queues — one per connected client
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()

# Cached tyre coefficients (loaded once, reused)
_tyre_cache = {"key": None, "coeffs": None, "curves": None}

def _get_cached_coefficients(year=None, track=None, drivers=None):
    """Fetch or calculate custom tyre coefficients and cache them."""
    # Convert types
    y = int(year) if year else None
    t = str(track) if track else None
    
    # Process comma-separated drivers string into a list
    d_list = None
    if drivers and isinstance(drivers, str):
        d_list = [d.strip() for d in drivers.split(',') if d.strip()]
    elif isinstance(drivers, list):
        d_list = drivers
        
    cache_key = f"{y}_{t}_{','.join(d_list) if d_list else 'ALL'}"
    
    if _tyre_cache.get("key") == cache_key and _tyre_cache.get("coeffs") is not None:
        return _tyre_cache["coeffs"]

    from src.data_loader import get_all_drivers_laps
    from src.tyre_model import fit_tyre_degradation

    kwargs = {}
    if y is not None: kwargs["year"] = y
    if t is not None: kwargs["gp"] = t
    if d_list: kwargs["drivers"] = d_list

    df = get_all_drivers_laps(**kwargs)
    
    # If df is empty, fallback to default no-filter
    if df.empty and d_list:
        print(f"Warning: No data found for drivers {d_list}. Falling back to all drivers.")
        kwargs.pop("drivers")
        df = get_all_drivers_laps(**kwargs)
        
    coeffs = fit_tyre_degradation(df)

    _tyre_cache["key"] = cache_key
    _tyre_cache["coeffs"] = coeffs
    return coeffs


def _broadcast_event(event_type: str, data: dict):
    """Push an SSE event to all connected clients."""
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    dead = []
    with _sse_lock:
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


def _load_race_history():
    """Load race history from JSON file."""
    path = LOGS_DIR / "race_history.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_race_result(result: dict):
    """Append a race result to persistent history."""
    history = _load_race_history()
    result["id"] = len(history) + 1
    result["timestamp"] = datetime.now().isoformat()
    history.append(result)
    path = LOGS_DIR / "race_history.json"
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return result


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events endpoint for real-time updates."""
    def event_stream():
        q = queue.Queue(maxsize=200)
        with _sse_lock:
            _sse_clients.append(q)
        try:
            yield f"event: connected\ndata: {json.dumps({'device': DEVICE})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/status")
def api_status():
    """Get current training status + metrics (fallback for SSE)."""
    metrics_path = LOGS_DIR / "training_metrics.json"
    metrics = {"status": "idle", "device": DEVICE}

    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    with _lock:
        metrics["is_training"] = _state["is_training"]
        metrics["is_inferring"] = _state["is_inferring"]

    metrics["device"] = DEVICE
    metrics["models"] = {
        "initial": MODEL_INITIAL.with_suffix(".zip").exists(),
        "final": MODEL_FINAL.with_suffix(".zip").exists(),
    }

    return jsonify(metrics)


@app.route("/api/train", methods=["POST"])
def api_train():
    """Start training in a background thread."""
    with _lock:
        if _state["is_training"]:
            return jsonify({"error": "Training already in progress"}), 409
        _state["is_training"] = True

    data = request.json or {}
    timesteps = data.get("timesteps", PPO_INITIAL_TIMESTEPS)
    is_final = data.get("final", False)
    
    year = data.get("year", None)
    track = data.get("track", None)
    drivers = data.get("drivers", None)
    
    env_kwargs = {}
    if "weather_mode" in data: env_kwargs["weather_mode"] = data["weather_mode"]
    if "sc_probability" in data: env_kwargs["sc_probability"] = float(data["sc_probability"])
    if "starting_position" in data: env_kwargs["starting_position"] = int(data["starting_position"])

    def _train_worker():
        try:
            _broadcast_event("training_start", {"timesteps": timesteps, "final": is_final})

            coeffs = _get_cached_coefficients(year=year, track=track, drivers=drivers)

            from src.train import train as do_train, TrainingMetricsCallback
            from src.config import MODEL_INITIAL as MI, MODEL_FINAL as MF, PPO_FINAL_TIMESTEPS as FTS

            class SSEMetricsCallback(TrainingMetricsCallback):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._last_sse = 0

                def _on_step(self):
                    result = super()._on_step()
                    if self.num_timesteps - self._last_sse >= 2048:
                        _broadcast_event("training_progress", {
                            "timestep": self.num_timesteps,
                            "total": self.metrics["total_timesteps"],
                            "episodes": self.metrics["episodes"],
                            "mean_reward": self.metrics["mean_reward"],
                            "reward_history": self.metrics["reward_history"][-5:],
                        })
                        self._last_sse = self.num_timesteps
                    return result

            if is_final:
                do_train(FTS, MF, coeffs, resume_from=MI, callback_cls=SSEMetricsCallback, env_kwargs=env_kwargs)
            else:
                do_train(timesteps, MI, coeffs, callback_cls=SSEMetricsCallback, env_kwargs=env_kwargs)

        except Exception as e:
            traceback.print_exc()
            _broadcast_event("training_error", {"error": str(e)})
            metrics_path = LOGS_DIR / "training_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"status": "error", "error": str(e)}, f)
        finally:
            with _lock:
                _state["is_training"] = False
            # Broadcast completion AFTER state is cleared so any
            # fetchStatus() poll triggered by this event sees is_training=False
            metrics_path = LOGS_DIR / "training_metrics.json"
            final_metrics = {}
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        final_metrics = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
            _broadcast_event("training_complete", {
                "elapsed": final_metrics.get("elapsed_seconds", 0),
                "episodes": final_metrics.get("episodes", 0),
                "mean_reward": final_metrics.get("mean_reward", 0),
            })

    threading.Thread(target=_train_worker, daemon=True).start()
    return jsonify({"message": f"Training started ({timesteps:,} steps on {DEVICE})"})


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """Run inference — streams lap-by-lap results via SSE."""
    with _lock:
        if _state["is_training"]:
            return jsonify({"error": "Cannot run inference while training"}), 409
        if _state["is_inferring"]:
            return jsonify({"error": "Inference already running"}), 409

    # Check model exists BEFORE spawning thread
    if MODEL_FINAL.with_suffix(".zip").exists():
        model_path = MODEL_FINAL
    elif MODEL_INITIAL.with_suffix(".zip").exists():
        model_path = MODEL_INITIAL
    else:
        return jsonify({"error": "No trained model found. Train first."}), 404

    data = request.json or {}
    year = data.get("year", None)
    track = data.get("track", None)
    drivers = data.get("drivers", None)
    
    env_kwargs = {}
    if "weather_mode" in data: env_kwargs["weather_mode"] = data["weather_mode"]
    if "sc_probability" in data: env_kwargs["sc_probability"] = float(data["sc_probability"])
    if "starting_position" in data: env_kwargs["starting_position"] = int(data["starting_position"])

    with _lock:
        _state["is_inferring"] = True

    def _infer_worker():
        try:
            _broadcast_event("infer_start", {})

            from stable_baselines3 import PPO
            from src.env import F1StrategyEnv

            model = PPO.load(str(model_path), device=DEVICE)

            d_list = ["Agent"]
            if drivers:
                if isinstance(drivers, str):
                    d_list = [d.strip() for d in drivers.split(',') if d.strip()]
                elif isinstance(drivers, list):
                    d_list = drivers
            if not d_list:
                d_list = ["Agent"]

            envs = {}
            for d in d_list:
                c = _get_cached_coefficients(year=year, track=track, drivers=d if d != "Agent" else None)
                env = F1StrategyEnv(tyre_coefficients=c, **env_kwargs)
                obs, info = env.reset(seed=42)
                envs[d] = {
                    "env": env,
                    "obs": obs,
                    "done": False,
                    "total_reward": 0.0,
                    "laps": [],
                    "weather_events": [],
                    "best_lap": {"time": 999, "lap": 0},
                    "worst_lap": {"time": 0, "lap": 0},
                    "info": info
                }

            all_done = False
            while not all_done:
                all_done = True
                
                for d, data in envs.items():
                    if data["done"]:
                        continue
                    
                    all_done = False
                    
                    action, _ = model.predict(data["obs"], deterministic=True)
                    obs, reward, terminated, truncated, info = data["env"].step(int(action))
                    data["obs"] = obs
                    data["total_reward"] += reward
                    data["done"] = terminated or truncated
                    data["info"] = info
                    
                    lap_data = {
                        "driver": d,
                        "lap": int(info["lap"]),
                        "tyre_age": int(info["tyre_age"]),
                        "compound": str(info["compound"]),
                        "position": int(info["position"]),
                        "pit_stops": int(info["pit_stops"]),
                        "pit_laps": [int(x) for x in info["pit_laps"]],
                        "laptime": float(info["laptime"]),
                        "fuel_kg": float(info["fuel_kg"]),
                        "gap_to_leader": float(info["gap_to_leader"]),
                        "safety_car": bool(info["safety_car"]),
                        "cumulative_time": float(info["cumulative_time"]),
                        "is_raining": bool(info.get("is_raining", False)),
                        "rain_intensity": int(info.get("rain_intensity", 0)),
                        "action": int(action),
                        "reward": round(float(reward), 3),
                    }
                    data["laps"].append(lap_data)
                    
                    if not lap_data["safety_car"]:
                        if lap_data["laptime"] < data["best_lap"]["time"]:
                            data["best_lap"] = {"time": lap_data["laptime"], "lap": lap_data["lap"]}
                        if lap_data["laptime"] > data["worst_lap"]["time"]:
                            data["worst_lap"] = {"time": lap_data["laptime"], "lap": lap_data["lap"]}
                            
                    if lap_data["is_raining"] and (not data["laps"][-2]["is_raining"] if len(data["laps"]) > 1 else True):
                        data["weather_events"].append({"lap": lap_data["lap"], "type": "rain_start", "intensity": lap_data["rain_intensity"]})
                    elif not lap_data["is_raining"] and len(data["laps"]) > 1 and data["laps"][-2]["is_raining"]:
                        data["weather_events"].append({"lap": lap_data["lap"], "type": "rain_stop"})

                    _broadcast_event("infer_lap", lap_data)
                    
                time.sleep(0.05)

            result = {"drivers": {}}
            for d, data in envs.items():
                laps = data["laps"]
                env_info = data["info"]
                rain_laps = sum(1 for l in laps if l["is_raining"])
                sc_laps = sum(1 for l in laps if l["safety_car"])
                positions_gained = STARTING_POSITION - int(env_info["position"])
                
                result["drivers"][d] = {
                    "laps": laps,
                    "total_reward": round(float(data["total_reward"]), 2),
                    "final_position": int(env_info["position"]),
                    "start_position": STARTING_POSITION,
                    "positions_gained": positions_gained,
                    "pit_stops": int(env_info["pit_stops"]),
                    "pit_laps": [int(x) for x in env_info["pit_laps"]],
                    "best_lap": data["best_lap"],
                    "worst_lap": data["worst_lap"],
                    "rain_laps": rain_laps,
                    "sc_laps": sc_laps,
                    "weather_events": data["weather_events"],
                    "total_laps": len(laps),
                }

            d_first = d_list[0]
            main_res = result["drivers"][d_first]
            
            race_path = LOGS_DIR / "last_race.json"
            with open(race_path, "w") as f:
                json.dump(result, f, indent=2)

            history_entry = {
                "drivers": d_list,
                "final_position": main_res["final_position"],
                "positions_gained": main_res["positions_gained"],
                "pit_stops": main_res["pit_stops"],
                "total_reward": main_res["total_reward"],
                "best_lap_time": main_res["best_lap"]["time"],
                "rain_laps": main_res["rain_laps"],
                "sc_laps": main_res["sc_laps"],
                "total_laps": main_res["total_laps"],
            }
            _save_race_result(history_entry)

            _broadcast_event("infer_complete", result)

        except Exception as e:
            traceback.print_exc()
            _broadcast_event("infer_error", {"error": str(e)})
        finally:
            with _lock:
                _state["is_inferring"] = False

    threading.Thread(target=_infer_worker, daemon=True).start()
    return jsonify({"message": "Race started — watch the telemetry feed"})


@app.route("/api/race-data")
def api_race_data():
    race_path = LOGS_DIR / "last_race.json"
    if race_path.exists():
        with open(race_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No race data. Run inference first."}), 404


@app.route("/api/race-history")
def api_race_history():
    """Get all past race results."""
    history = _load_race_history()
    return jsonify({"races": history, "total": len(history)})


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Get current reward config weights."""
    from src import config as cfg
    return jsonify({
        "reward_position_gain": cfg.REWARD_POSITION_GAIN,
        "reward_pit_cost": cfg.REWARD_PIT_COST,
        "reward_tyre_cliff_penalty": cfg.REWARD_TYRE_CLIFF_PENALTY,
        "reward_finish_bonus_scale": cfg.REWARD_FINISH_BONUS_SCALE,
        "reward_sc_pit_bonus": cfg.REWARD_SC_PIT_BONUS,
        "reward_wrong_tyre_penalty": cfg.REWARD_WRONG_TYRE_PENALTY,
        "rain_probability": cfg.RAIN_PROBABILITY,
        "safety_car_probability": cfg.SAFETY_CAR_PROBABILITY,
        "total_laps": cfg.TOTAL_LAPS,
        "starting_position": cfg.STARTING_POSITION,
    })


@app.route("/api/config", methods=["POST"])
def api_set_config():
    """Update reward config weights at runtime (does NOT persist to file)."""
    from src import config as cfg
    data = request.json or {}
    updated = []
    for key, value in data.items():
        attr = key.upper()
        if hasattr(cfg, attr):
            setattr(cfg, attr, float(value))
            updated.append(attr)
    if updated:
        return jsonify({"message": f"Updated: {', '.join(updated)}", "updated": updated})
    return jsonify({"error": "No valid config keys provided"}), 400


@app.route("/api/tyre-model")
def api_tyre_model():
    """Get tyre degradation curves (cached)."""
    try:
        from src.tyre_model import predict_laptime
        coeffs = _get_cached_coefficients()

        if _tyre_cache["curves"] is None:
            curves = {}
            for compound in ["SOFT", "MEDIUM", "HARD"]:  # Only dry compounds have fitted curves
                if compound in coeffs:
                    curves[compound] = {
                        "coefficients": list(coeffs[compound]),
                        "predicted": [
                            {"age": age, "laptime": round(predict_laptime(compound, age, coeffs), 3)}
                            for age in range(1, 36)
                        ],
                    }
            _tyre_cache["curves"] = curves

        return jsonify(_tyre_cache["curves"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/plots/<filename>")
def api_plot(filename):
    return send_from_directory(str(PLOTS_DIR), filename)


if __name__ == "__main__":
    host = os.environ.get("HOST", FLASK_HOST)
    port = int(os.environ.get("PORT", FLASK_PORT))
    print(f"\n{'═' * 60}")
    print(f"  STINT ENGINE — Dashboard v2")
    print(f"  http://{host}:{port}")
    print(f"  Device: {DEVICE.upper()}")
    print(f"{'═' * 60}\n")
    app.run(host=host, port=port, debug=True, threaded=True)
