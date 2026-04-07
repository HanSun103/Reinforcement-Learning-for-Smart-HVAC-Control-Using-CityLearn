"""
evaluate.py
-----------
Evaluation and comparison entry point for the CityLearn HVAC RL project.

Run from the project root directory (after running train.py):
    python src/evaluate.py

What this script does
---------------------
1. Load the trained SAC model from disk.
2. Run one deterministic evaluation episode and collect KPIs + traces.
3. Load the baseline KPIs and trace data saved by train.py.
4. Generate and save the following outputs to results/:
     - rl_kpis.csv                     : RL agent KPI table
     - kpi_comparison.png              : grouped bar chart (baseline vs SAC)
     - training_rewards.png            : SAC episode reward curve
     - temperature_trace_baseline.png  : indoor temp vs setpoint (baseline)
     - temperature_trace_sac.png       : indoor temp vs setpoint (SAC)
     - reward_comparison.png           : per-step reward side-by-side
5. Print a formatted summary table to the console.

Prerequisites
-------------
- Run `python src/train.py` first to produce the model and baseline files.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so 'src.*' imports resolve correctly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.rl_agent import evaluate_sac
from src.utils import (
    plot_training_rewards,
    plot_kpi_comparison,
    plot_temperature_trace,
    plot_reward_comparison,
    save_metrics_csv,
    print_summary_table,
)


def _load_baseline_kpis() -> pd.DataFrame:
    """
    Load the baseline KPI CSV saved by train.py.

    Returns
    -------
    pd.DataFrame
        KPI table with cost_function as index.

    Raises
    ------
    FileNotFoundError
        If train.py has not been run yet.
    """
    if not os.path.exists(config.BASELINE_KPI_PATH):
        raise FileNotFoundError(
            f"Baseline KPI file not found at '{config.BASELINE_KPI_PATH}'. "
            "Please run `python src/train.py` first."
        )
    kpis = pd.read_csv(config.BASELINE_KPI_PATH, index_col=0)
    print(f"  Loaded baseline KPIs from: {config.BASELINE_KPI_PATH}")
    return kpis


def _load_baseline_trace() -> dict:
    """
    Load the baseline temperature trace saved by train.py.

    Returns
    -------
    dict with keys 'indoor_temps', 'setpoints', 'rewards'.
    Returns empty lists if the file does not exist.
    """
    trace_path = os.path.join(config.RESULTS_DIR, "baseline_trace.npz")
    if not os.path.exists(trace_path):
        print(f"  Warning: baseline trace not found at '{trace_path}'. "
              "Temperature comparison plot will be skipped.")
        return {"indoor_temps": [], "setpoints": [], "rewards": []}

    data = np.load(trace_path, allow_pickle=True)
    print(f"  Loaded baseline trace from: {trace_path}")
    return {
        "indoor_temps": data["indoor_temps"].tolist(),
        "setpoints": data["setpoints"].tolist(),
        "rewards": data["rewards"].tolist(),
    }


def main():
    print("\n" + "=" * 60)
    print("  CityLearn HVAC RL Project — Evaluation Pipeline")
    print("=" * 60)
    print(f"  Building   : {config.BUILDING_TO_USE}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print("=" * 60 + "\n")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load baseline data
    # ------------------------------------------------------------------
    print("--- Step 1: Loading baseline results ---")
    baseline_kpis = _load_baseline_kpis()
    baseline_trace = _load_baseline_trace()

    # ------------------------------------------------------------------
    # Step 2: Evaluate SAC agent
    # ------------------------------------------------------------------
    print("\n--- Step 2: Evaluating trained SAC agent ---")
    t0 = time.time()
    rl_results = evaluate_sac()   # loads model from MODEL_SAVE_PATH
    elapsed = time.time() - t0
    print(f"  RL evaluation completed in {elapsed:.1f}s")

    rl_kpis = rl_results["kpis"]

    # Save RL KPIs to CSV.
    save_metrics_csv(rl_kpis, config.RL_KPI_PATH)

    # ------------------------------------------------------------------
    # Step 3: Print summary comparison
    # ------------------------------------------------------------------
    print("\n--- Step 3: KPI comparison ---")
    print_summary_table(baseline_kpis, rl_kpis)

    # ------------------------------------------------------------------
    # Step 4: Generate plots
    # ------------------------------------------------------------------
    print("--- Step 4: Generating plots ---")

    # 4a. Training reward curve (reads SB3 Monitor CSV).
    print("  Plotting training reward curve...")
    try:
        plot_training_rewards(log_dir=config.MONITOR_LOG_DIR)
    except Exception as exc:
        print(f"  Warning: could not plot training rewards ({exc}).")

    # 4b. KPI comparison bar chart.
    print("  Plotting KPI comparison...")
    try:
        plot_kpi_comparison(baseline_kpis, rl_kpis)
    except Exception as exc:
        print(f"  Warning: could not plot KPI comparison ({exc}).")

    # 4c. Temperature trace — baseline.
    print("  Plotting baseline temperature trace...")
    try:
        if baseline_trace["indoor_temps"]:
            plot_temperature_trace(
                indoor_temps=baseline_trace["indoor_temps"],
                setpoints=baseline_trace["setpoints"],
                label="Baseline (RBC)",
                filename="temperature_trace_baseline.png",
            )
        else:
            print("  Skipping baseline temperature trace (no data).")
    except Exception as exc:
        print(f"  Warning: could not plot baseline temperature trace ({exc}).")

    # 4d. Temperature trace — RL agent.
    print("  Plotting SAC temperature trace...")
    try:
        if rl_results["indoor_temps"]:
            plot_temperature_trace(
                indoor_temps=rl_results["indoor_temps"],
                setpoints=rl_results["setpoints"],
                label="SAC (RL agent)",
                filename="temperature_trace_sac.png",
            )
        else:
            print("  Skipping SAC temperature trace (no data).")
    except Exception as exc:
        print(f"  Warning: could not plot SAC temperature trace ({exc}).")

    # 4e. Per-step reward comparison.
    print("  Plotting per-step reward comparison...")
    try:
        if baseline_trace["rewards"] and rl_results["rewards_per_step"]:
            plot_reward_comparison(
                baseline_rewards=baseline_trace["rewards"],
                rl_rewards=rl_results["rewards_per_step"],
            )
        else:
            print("  Skipping reward comparison (missing data for one or both agents).")
    except Exception as exc:
        print(f"  Warning: could not plot reward comparison ({exc}).")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print(f"  All outputs saved to: {config.RESULTS_DIR}/")
    print()
    print("  Files generated:")
    output_files = [
        "rl_kpis.csv",
        "kpi_comparison.png",
        "training_rewards.png",
        "temperature_trace_baseline.png",
        "temperature_trace_sac.png",
        "reward_comparison.png",
    ]
    for f in output_files:
        path = os.path.join(config.RESULTS_DIR, f)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"    [{status}] {f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
