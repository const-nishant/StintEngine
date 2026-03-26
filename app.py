"""
StintEngine — Web Dashboard (Flask Backend)
REST API with threaded execution + SSE for real-time updates.
"""

import json
import queue
import threading
import time
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_from_directory, Response

from src.config import (
    FLASK_HOST, FLASK_PORT, MODELS_DIR, LOGS_DIR, PLOTS_DIR,
    MODEL_INITIAL, MODEL_FINAL, DEVICE,
    PPO_INITIAL_TIMESTEPS, PPO_FINAL_TIMESTEPS,
    TEMPLATES_DIR, STATIC_DIR, COMPOUNDS,
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
_tyre_cache = {"coeffs": None, "curves": None}


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


def _get_cached_coefficients():
    """Load tyre coefficients once and cache them."""
    if _tyre_cache["coeffs"] is None:
        from src.data_loader import get_all_drivers_laps
        from src.tyre_model import fit_tyre_degradation
        all_laps = get_all_drivers_laps()
        _tyre_cache["coeffs"] = fit_tyre_degradation(all_laps)
    return _tyre_cache["coeffs"]


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
            # Send initial state
            yield f"event: connected\ndata: {json.dumps({'device': DEVICE})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    # Keep-alive ping
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

    def _train_worker():
        try:
            _broadcast_event("training_start", {"timesteps": timesteps, "final": is_final})

            coeffs = _get_cached_coefficients()

            from src.train import train as do_train, TrainingMetricsCallback
            from src.config import MODEL_INITIAL as MI, MODEL_FINAL as MF, PPO_FINAL_TIMESTEPS as FTS

            # Custom callback that also broadcasts SSE
            class SSEMetricsCallback(TrainingMetricsCallback):
                def _on_step(self):
                    result = super()._on_step()
                    # Broadcast every 2048 steps
                    if self.num_timesteps % 2048 == 0:
                        _broadcast_event("training_progress", {
                            "timestep": self.num_timesteps,
                            "total": self.metrics["total_timesteps"],
                            "episodes": self.metrics["episodes"],
                            "mean_reward": self.metrics["mean_reward"],
                            "reward_history": self.metrics["reward_history"][-5:],
                        })
                    return result

                def _on_training_end(self):
                    super()._on_training_end()
                    _broadcast_event("training_complete", {
                        "elapsed": self.metrics.get("elapsed_seconds", 0),
                        "episodes": self.metrics["episodes"],
                        "mean_reward": self.metrics["mean_reward"],
                    })

            if is_final:
                do_train(FTS, MF, coeffs, resume_from=MI, callback_cls=SSEMetricsCallback)
            else:
                do_train(timesteps, MI, coeffs, callback_cls=SSEMetricsCallback)

        except Exception as e:
            print(f"Training error: {e}")
            _broadcast_event("training_error", {"error": str(e)})
            metrics_path = LOGS_DIR / "training_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"status": "error", "error": str(e)}, f)
        finally:
            with _lock:
                _state["is_training"] = False

    threading.Thread(target=_train_worker, daemon=True).start()
    return jsonify({"message": f"Training started ({timesteps:,} steps on {DEVICE})"})


@app.route("/api/infer", methods=["POST"])
def api_infer():
    """Run inference in a background thread — returns immediately, streams results via SSE."""
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

    with _lock:
        _state["is_inferring"] = True

    def _infer_worker():
        try:
            _broadcast_event("infer_start", {})

            coeffs = _get_cached_coefficients()

            from stable_baselines3 import PPO
            from src.env import F1StrategyEnv

            model = PPO.load(str(model_path), device=DEVICE)
            env = F1StrategyEnv(tyre_coefficients=coeffs)

            obs, info = env.reset()
            done = False
            total_reward = 0.0
            laps = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward
                done = terminated or truncated

                # Explicit serialization (avoid numpy types in JSON)
                lap_data = {
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
                    "action": int(action),
                    "reward": round(float(reward), 3),
                }
                laps.append(lap_data)

                # Stream each lap in real-time
                _broadcast_event("infer_lap", lap_data)
                time.sleep(0.05)

            result = {
                "laps": laps,
                "total_reward": round(float(total_reward), 2),
                "final_position": int(info["position"]),
                "pit_stops": int(info["pit_stops"]),
                "pit_laps": [int(x) for x in info["pit_laps"]],
            }

            # Save race log
            race_path = LOGS_DIR / "last_race.json"
            with open(race_path, "w") as f:
                json.dump(result, f, indent=2)

            _broadcast_event("infer_complete", result)

        except Exception as e:
            import traceback
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


@app.route("/api/tyre-model")
def api_tyre_model():
    """Get tyre degradation curves (cached)."""
    try:
        from src.tyre_model import predict_laptime
        coeffs = _get_cached_coefficients()

        if _tyre_cache["curves"] is None:
            curves = {}
            for compound in COMPOUNDS:
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
    print(f"\n{'═' * 60}")
    print(f"  STINT ENGINE — Dashboard")
    print(f"  http://{FLASK_HOST}:{FLASK_PORT}")
    print(f"  Device: {DEVICE.upper()}")
    print(f"{'═' * 60}\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True, threaded=True)
