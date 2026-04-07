"""
train.py
--------
Main training entry point for the CityLearn HVAC RL project.

Run from the project root directory:
    python src/train.py

What this script does
---------------------
1. Print the environment summary (observation/action space details).
2. Run the Rule-Based Controller (baseline) for one episode.
   - Saves KPIs to results/baseline_kpis.csv.
   - Saves baseline temperature trace data to results/baseline_trace.npz.
3. Train the SAC agent for config.TRAIN_EPISODES episodes.
   - Prints progress after each episode.
   - Saves trained model to results/sac_hvac_model.zip.
4. Save baseline reward data for later comparison in evaluate.py.

After running this script, run `python src/evaluate.py` to generate plots.
"""

import os
import sys
import time

import numpy as np

# Ensure project root is on sys.path so 'src.*' imports work.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.env_setup import make_baseline_env, print_env_info
from src.baseline_agent import run_baseline
from src.rl_agent import train_sac


def main():
    print("\n" + "=" * 60)
    print("  CityLearn HVAC RL Project — Training Pipeline")
    print("=" * 60)
    print(f"  Building        : {config.BUILDING_TO_USE}")
    print(f"  Dataset         : {config.SCHEMA_NAME}")
    print(f"  Train episodes  : {config.TRAIN_EPISODES}")
    print(f"  Comfort band    : ±{config.COMFORT_BAND} °C")
    print(f"  Reward weights  : energy={config.REWARD_ENERGY_WEIGHT}, "
          f"comfort={config.REWARD_COMFORT_WEIGHT}")
    print(f"  Results dir     : {config.RESULTS_DIR}")
    print("=" * 60 + "\n")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Print environment info
    # ------------------------------------------------------------------
    print("--- Step 1: Environment summary ---")
    try:
        raw_env = make_baseline_env()
        print_env_info(raw_env)
    except Exception as exc:
        print(f"  Warning: could not print env info ({exc}). Continuing.")

    # ------------------------------------------------------------------
    # Step 2: Run baseline
    # ------------------------------------------------------------------
    print("--- Step 2: Running Rule-Based Controller baseline ---")
    t0 = time.time()
    baseline_results = run_baseline()
    elapsed = time.time() - t0
    print(f"  Baseline rollout completed in {elapsed:.1f}s")

    # Save baseline KPIs to CSV.
    try:
        baseline_results["kpis"].to_csv(config.BASELINE_KPI_PATH)
        print(f"  Baseline KPIs saved to: {config.BASELINE_KPI_PATH}")
    except Exception as exc:
        print(f"  Warning: could not save baseline KPIs ({exc}).")

    # Save baseline trace data for temperature plots.
    trace_path = os.path.join(config.RESULTS_DIR, "baseline_trace.npz")
    try:
        np.savez(
            trace_path,
            rewards=np.array(baseline_results["rewards_per_step"]),
            indoor_temps=np.array(baseline_results["indoor_temps"]),
            setpoints=np.array(baseline_results["setpoints"]),
        )
        print(f"  Baseline trace data saved to: {trace_path}")
    except Exception as exc:
        print(f"  Warning: could not save baseline trace ({exc}).")

    # ------------------------------------------------------------------
    # Step 3: Train SAC
    # ------------------------------------------------------------------
    print("\n--- Step 3: Training SAC agent ---")
    t0 = time.time()
    try:
        model = train_sac()
        elapsed = time.time() - t0
        print(f"\n  SAC training completed in {elapsed / 60:.1f} minutes.")
    except Exception as exc:
        print(f"\n  ERROR during SAC training: {exc}")
        raise

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Model saved to : {config.MODEL_SAVE_PATH}.zip")
    print(f"  Baseline KPIs  : {config.BASELINE_KPI_PATH}")
    print()
    print("  Next step: run  python src/evaluate.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
