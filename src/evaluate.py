"""
evaluate.py
-----------
Evaluation and comparison entry point for the CityLearn HVAC RL project.

Run from the project root directory (after running train.py):
    python src/evaluate.py

What this script does
---------------------
1. Load the baseline KPIs and trace data saved by train.py.
2. Evaluate every RL model that exists on disk (SAC, PPO, and/or TD3).
3. Generate outputs in results/:
     - sac_kpis.csv / ppo_kpis.csv / td3_kpis.csv : per-algorithm KPI tables
     - kpi_comparison.png              : grouped bar chart (all agents)
     - training_rewards.png            : training reward curves (all algorithms)
     - temperature_trace_*.png         : indoor temp vs setpoint, one per agent
     - reward_comparison.png           : per-step reward (all agents)
4. Print a formatted side-by-side KPI summary table.

Prerequisites
-------------
    python src/train.py          # trains SAC + PPO + TD3 (default)
    python src/train.py --algo sac   # or just one
"""

import os
import sys
import time

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.rl_agent import evaluate_agent
from src.utils import (
    plot_training_rewards,
    plot_kpi_comparison,
    plot_temperature_trace,
    plot_reward_comparison,
    save_metrics_csv,
    print_summary_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_baseline_kpis() -> pd.DataFrame:
    if not os.path.exists(config.BASELINE_KPI_PATH):
        raise FileNotFoundError(
            f"Baseline KPI file not found at '{config.BASELINE_KPI_PATH}'. "
            "Please run `python src/train.py` first."
        )
    kpis = pd.read_csv(config.BASELINE_KPI_PATH, index_col=0)
    print(f"  Loaded baseline KPIs from: {config.BASELINE_KPI_PATH}")
    return kpis


def _load_baseline_trace() -> dict:
    trace_path = os.path.join(config.RESULTS_DIR, "baseline_trace.npz")
    if not os.path.exists(trace_path):
        print("  Warning: baseline trace not found — temperature plots will be skipped.")
        return {"indoor_temps": [], "setpoints": [], "rewards": []}
    data = np.load(trace_path, allow_pickle=True)
    print(f"  Loaded baseline trace from: {trace_path}")
    return {
        "indoor_temps": data["indoor_temps"].tolist(),
        "setpoints":    data["setpoints"].tolist(),
        "rewards":      data["rewards"].tolist(),
    }


def _available_algos() -> list:
    """Return the list of algorithms whose model files exist on disk."""
    candidates = [
        ("sac", config.SAC_MODEL_SAVE_PATH + ".zip"),
        ("ppo", config.PPO_MODEL_SAVE_PATH + ".zip"),
        ("td3", config.TD3_MODEL_SAVE_PATH + ".zip"),
    ]
    return [algo for algo, path in candidates if os.path.exists(path)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  CityLearn HVAC RL Project — Evaluation Pipeline")
    print("=" * 60)
    print(f"  Building   : {config.BUILDING_TO_USE}")
    print(f"  Results dir: {config.RESULTS_DIR}")
    print("=" * 60 + "\n")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load baseline
    # ------------------------------------------------------------------
    print("--- Step 1: Loading baseline results ---")
    baseline_kpis = _load_baseline_kpis()
    baseline_trace = _load_baseline_trace()

    # ------------------------------------------------------------------
    # Step 2: Evaluate available RL models
    # ------------------------------------------------------------------
    algos = _available_algos()
    if not algos:
        print("\n  No trained RL models found. Run `python src/train.py` first.\n")
        return

    print(f"\n--- Step 2: Evaluating {', '.join(a.upper() for a in algos)} ---")

    # Collect results for all agents.
    # agents_kpis  : {label: kpi_DataFrame}   for bar chart + summary table
    # agents_rewards: {label: rewards_list}   for reward comparison plot
    agents_kpis = {"RBC": baseline_kpis}
    agents_rewards = {"RBC": baseline_trace["rewards"]}
    rl_results = {}

    kpi_path_map = {
        "sac": config.SAC_KPI_PATH,
        "ppo": config.PPO_KPI_PATH,
        "td3": config.TD3_KPI_PATH,
    }

    for algo in algos:
        t0 = time.time()
        result = evaluate_agent(algo)
        print(f"  {algo.upper()} evaluation completed in {time.time() - t0:.1f}s")

        save_metrics_csv(result["kpis"], kpi_path_map[algo])

        label = algo.upper()
        agents_kpis[label] = result["kpis"]
        agents_rewards[label] = result["rewards_per_step"]
        rl_results[algo] = result

    # ------------------------------------------------------------------
    # Step 3: KPI summary table
    # ------------------------------------------------------------------
    print("\n--- Step 3: KPI comparison ---")
    print_summary_table(agents_kpis)

    # ------------------------------------------------------------------
    # Step 4: Plots
    # ------------------------------------------------------------------
    print("--- Step 4: Generating plots ---")

    # 4a. Training reward curves (all algorithms on one figure).
    print("  Plotting training reward curves…")
    try:
        plot_training_rewards()
    except Exception as exc:
        print(f"  Warning: could not plot training rewards ({exc}).")

    # 4b. KPI comparison bar chart.
    print("  Plotting KPI comparison…")
    try:
        plot_kpi_comparison(agents_kpis)
    except Exception as exc:
        print(f"  Warning: could not plot KPI comparison ({exc}).")

    # 4c. Temperature trace — baseline.
    print("  Plotting baseline temperature trace…")
    try:
        if baseline_trace["indoor_temps"]:
            plot_temperature_trace(
                indoor_temps=baseline_trace["indoor_temps"],
                setpoints=baseline_trace["setpoints"],
                label="Baseline (RBC)",
                filename="temperature_trace_baseline.png",
            )
    except Exception as exc:
        print(f"  Warning: {exc}")

    # 4d. Temperature trace — each RL agent.
    for algo, result in rl_results.items():
        label = algo.upper()
        print(f"  Plotting {label} temperature trace…")
        try:
            if result["indoor_temps"]:
                plot_temperature_trace(
                    indoor_temps=result["indoor_temps"],
                    setpoints=result["setpoints"],
                    label=f"{label} (RL agent)",
                    filename=f"temperature_trace_{algo}.png",
                )
        except Exception as exc:
            print(f"  Warning: {exc}")

    # 4e. Per-step reward comparison (all agents).
    print("  Plotting per-step reward comparison…")
    try:
        if any(len(v) > 0 for v in agents_rewards.values()):
            plot_reward_comparison(agents=agents_rewards)
        else:
            print("  Skipping (no reward data).")
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
    expected = (
        [f"{algo}_kpis.csv" for algo in algos]
        + ["kpi_comparison.png", "training_rewards.png",
           "temperature_trace_baseline.png"]
        + [f"temperature_trace_{algo}.png" for algo in algos]
        + ["reward_comparison.png"]
    )
    for f in expected:
        path = os.path.join(config.RESULTS_DIR, f)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"    [{status}] {f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
