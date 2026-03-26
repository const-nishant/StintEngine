"""
StintEngine — CLI Entry Point
Usage:
    python main.py data        Load & inspect race data + tyre model
    python main.py train       Train PPO agent (50k initial, GPU)
    python main.py train-final Train final model (200k steps, GPU)
    python main.py infer       Run trained agent race
    python main.py viz         Generate comparison plots
    python main.py dashboard   Launch web dashboard
    python main.py all         Full pipeline: data → train → viz
"""

import sys


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "data":
        from src.data_loader import get_race_data, get_all_drivers_laps
        from src.tyre_model import fit_tyre_degradation
        from src.config import REFERENCE_DRIVER, DEVICE

        print(f"Device: {DEVICE.upper()}")
        print("Loading race data...")
        driver_laps = get_race_data()
        print(f"\n{REFERENCE_DRIVER} laps: {len(driver_laps)}")
        print(driver_laps[["LapNumber", "Compound", "TyreLife", "LapTimeSeconds", "Position"]].head(10))

        print("\nFitting tyre degradation model...")
        all_laps = get_all_drivers_laps()
        coeffs = fit_tyre_degradation(all_laps)

    elif command == "train":
        from src.train import load_tyre_coefficients, train
        from src.config import PPO_INITIAL_TIMESTEPS, MODEL_INITIAL

        timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else PPO_INITIAL_TIMESTEPS
        coeffs = load_tyre_coefficients()
        train(timesteps, MODEL_INITIAL, coeffs)

    elif command == "train-final":
        from src.train import load_tyre_coefficients, train
        from src.config import PPO_FINAL_TIMESTEPS, MODEL_FINAL, MODEL_INITIAL

        coeffs = load_tyre_coefficients()
        train(PPO_FINAL_TIMESTEPS, MODEL_FINAL, coeffs, resume_from=MODEL_INITIAL)

    elif command == "infer":
        from src.train import load_tyre_coefficients, run_inference
        from src.config import MODEL_FINAL, MODEL_INITIAL

        coeffs = load_tyre_coefficients()
        model_path = MODEL_FINAL if MODEL_FINAL.with_suffix(".zip").exists() else MODEL_INITIAL
        run_inference(model_path, coeffs)

    elif command == "viz":
        from src.visualize import generate_all_plots
        generate_all_plots()

    elif command == "dashboard":
        from app import app
        from src.config import FLASK_HOST, FLASK_PORT, DEVICE
        print(f"\n  StintEngine Dashboard — http://{FLASK_HOST}:{FLASK_PORT}")
        print(f"  Device: {DEVICE.upper()}\n")
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)

    elif command == "all":
        print("═" * 60)
        print("  STINT ENGINE — Full Pipeline")
        print("═" * 60)

        from src.train import load_tyre_coefficients, train
        from src.config import PPO_INITIAL_TIMESTEPS, PPO_FINAL_TIMESTEPS
        from src.config import MODEL_INITIAL, MODEL_FINAL

        coeffs = load_tyre_coefficients()
        train(PPO_INITIAL_TIMESTEPS, MODEL_INITIAL, coeffs)
        train(PPO_FINAL_TIMESTEPS, MODEL_FINAL, coeffs, resume_from=MODEL_INITIAL)

        from src.visualize import generate_all_plots
        generate_all_plots()

        print("\n✓ Full pipeline complete!")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
