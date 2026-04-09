"""
train.py
--------
Main training entry point for the CityLearn HVAC RL project.

Usage
-----
    # Train all three algorithms:
    python src/train.py

    # Train a specific algorithm:
    python src/train.py --algo sac
    python src/train.py --algo ppo
    python src/train.py --algo td3

    # Enable TensorBoard logging (view with: tensorboard --logdir results/tensorboard):
    python src/train.py --tensorboard

    # Skip EvalCallback (faster, but no best-checkpoint saving):
    python src/train.py --no-eval-callback

Steps
-----
1. Print environment summary (obs/action space).
2. Run the Rule-Based Controller (baseline) and save KPIs + trace.
3. Train each requested RL algorithm and save models.
"""

import argparse
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.env_setup import make_baseline_env, print_env_info
from src.baseline_agent import run_baseline
from src.rl_agent import train_agent


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL agents for CityLearn HVAC control."
    )
    parser.add_argument(
        "--algo",
        choices=["sac", "ppo", "td3", "all"],
        default="all",
        help=(
            "Which algorithm(s) to train. "
            "'all' trains SAC + PPO + TD3 sequentially (default)."
        ),
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help=(
            "Enable TensorBoard logging. "
            "View with: tensorboard --logdir results/tensorboard"
        ),
    )
    parser.add_argument(
        "--no-eval-callback",
        action="store_true",
        default=False,
        help="Disable EvalCallback (faster training, no best-checkpoint saving).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    algos = ["sac", "ppo", "td3"] if args.algo == "all" else [args.algo]
    use_eval = not args.no_eval_callback

    print("\n" + "=" * 60)
    print("  CityLearn HVAC RL Project — Training Pipeline")
    print("=" * 60)
    print(f"  Building        : {config.BUILDING_TO_USE}"
          f"{' (multi-building)' if config.MULTI_BUILDING else ''}")
    print(f"  Dataset         : {config.SCHEMA_NAME}")
    print(f"  Algorithm(s)    : {', '.join(a.upper() for a in algos)}")
    print(f"  Train episodes  : {config.TRAIN_EPISODES}")
    print(f"  Observations    : {len(config.ACTIVE_OBSERVATIONS)} configured"
          f"{' (incl. forecasts)' if config.USE_PREDICTION_OBS else ''}")
    print(f"  Reward weights  : energy={config.REWARD_ENERGY_WEIGHT}, "
          f"comfort={config.REWARD_COMFORT_WEIGHT}, "
          f"carbon={config.REWARD_CARBON_WEIGHT}")
    print(f"  TensorBoard     : {'enabled' if args.tensorboard else 'disabled'}")
    print(f"  EvalCallback    : {'enabled' if use_eval else 'disabled'}")
    print(f"  Results dir     : {config.RESULTS_DIR}")
    print("=" * 60 + "\n")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Environment summary
    # ------------------------------------------------------------------
    print("--- Step 1: Environment summary ---")
    try:
        raw_env = make_baseline_env()
        print_env_info(raw_env)
    except Exception as exc:
        print(f"  Warning: could not print env info ({exc}). Continuing.")

    # ------------------------------------------------------------------
    # Step 2: Baseline rollout
    # ------------------------------------------------------------------
    print("--- Step 2: Running Rule-Based Controller baseline ---")
    t0 = time.time()
    baseline_results = run_baseline()
    print(f"  Baseline completed in {time.time() - t0:.1f}s")

    try:
        baseline_results["kpis"].to_csv(config.BASELINE_KPI_PATH)
        print(f"  KPIs saved to: {config.BASELINE_KPI_PATH}")
    except Exception as exc:
        print(f"  Warning: {exc}")

    trace_path = os.path.join(config.RESULTS_DIR, "baseline_trace.npz")
    try:
        np.savez(
            trace_path,
            rewards=np.array(baseline_results["rewards_per_step"]),
            indoor_temps=np.array(baseline_results["indoor_temps"]),
            setpoints=np.array(baseline_results["setpoints"]),
        )
        print(f"  Trace saved to: {trace_path}")
    except Exception as exc:
        print(f"  Warning: {exc}")

    # ------------------------------------------------------------------
    # Step 3: RL training
    # ------------------------------------------------------------------
    for algo in algos:
        print(f"\n--- Step 3: Training {algo.upper()} ---")
        t0 = time.time()
        try:
            train_agent(
                algo,
                use_tensorboard=args.tensorboard,
                use_eval_callback=use_eval,
            )
            elapsed = time.time() - t0
            print(f"\n  {algo.upper()} completed in {elapsed / 60:.1f} minutes.")
        except Exception as exc:
            print(f"\n  ERROR during {algo.upper()} training: {exc}")
            raise

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Training complete!")
    path_map = {
        "sac": config.SAC_MODEL_SAVE_PATH,
        "ppo": config.PPO_MODEL_SAVE_PATH,
        "td3": config.TD3_MODEL_SAVE_PATH,
    }
    for algo in algos:
        print(f"  {algo.upper()} model  : {path_map[algo]}.zip")
    print(f"  Baseline KPIs : {config.BASELINE_KPI_PATH}")
    if args.tensorboard:
        print(f"\n  TensorBoard: tensorboard --logdir {config.TENSORBOARD_LOG_DIR}")
    print()
    print("  Next step: python src/evaluate.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
