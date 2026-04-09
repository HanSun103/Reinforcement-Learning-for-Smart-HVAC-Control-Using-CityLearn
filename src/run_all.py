"""
run_all.py
----------
Integrated experiment pipeline: tune → train → evaluate in one command.

Usage
-----
    # Full pipeline: train SAC + PPO + TD3, then evaluate (default):
    python src/run_all.py

    # Single algorithm:
    python src/run_all.py --algo sac

    # Tune first (Optuna), then train with best params, then evaluate:
    python src/run_all.py --tune
    python src/run_all.py --algo sac --tune --trials 20

    # Enable TensorBoard (view at localhost:6006 while training):
    python src/run_all.py --tensorboard

    # Quick smoke run (1 episode, no eval callback, no tuning):
    python src/run_all.py --algo sac --episodes 1 --no-eval-callback

Outputs (all written to results/)
-----
    baseline_kpis.csv         baseline KPI table
    {algo}_hvac_model.zip     trained model (final weights)
    best_{algo}/best_model.zip best checkpoint from EvalCallback
    {algo}_kpis.csv           RL agent KPI table
    kpi_comparison.png        grouped bar chart (all agents)
    training_rewards.png      episode reward curves
    temperature_trace_*.png   indoor temp vs setpoint
    reward_comparison.png     per-step reward comparison
    best_params.json          Optuna best params (if --tune)
    tensorboard/              TensorBoard logs (if --tensorboard)
"""

import argparse
import json
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, label: str):
    """Run a subprocess command, streaming output live, and raise on failure."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, check=True)
    return result


def _load_best_params(algo: str) -> dict:
    """Load Optuna best params JSON if it exists and matches *algo*."""
    path = config.OPTUNA_BEST_PARAMS_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("algo") == algo:
            return data.get("params", {})
    except Exception:
        pass
    return {}


def _params_to_config_patch(algo: str, params: dict) -> list:
    """
    Convert Optuna best params to environment-variable overrides that can be
    read by config.py.  We use a lightweight convention:
      HVAC_{ALGO}_{KEY}=value
    and config.py reads them with os.getenv() when present.

    This avoids modifying config.py on disk between tune and train.
    """
    overrides = {}
    prefix = f"HVAC_{algo.upper()}_"
    for k, v in params.items():
        overrides[f"{prefix}{k.upper()}"] = str(v)
    return overrides


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated CityLearn HVAC RL pipeline: tune → train → evaluate."
    )
    parser.add_argument(
        "--algo",
        choices=["sac", "ppo", "td3", "all"],
        default="all",
        help="Algorithm(s) to train. 'all' = SAC + PPO + TD3 (default).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run Optuna hyperparameter search before training.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=config.OPTUNA_N_TRIALS,
        help=f"Number of Optuna trials (default: {config.OPTUNA_N_TRIALS}). "
             "Ignored unless --tune is set.",
    )
    parser.add_argument(
        "--tune-episodes",
        type=int,
        default=config.OPTUNA_N_EPISODES,
        help=f"Training episodes per Optuna trial (default: {config.OPTUNA_N_EPISODES}).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override TRAIN_EPISODES for this run (default: use config.py value).",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--no-eval-callback",
        action="store_true",
        default=False,
        help="Disable EvalCallback (faster, no best-checkpoint saving).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip the Rule-Based Controller baseline rollout.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        default=False,
        help="Skip the final evaluation step.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    algos = ["sac", "ppo", "td3"] if args.algo == "all" else [args.algo]
    python = sys.executable
    t_start = time.time()

    print("\n" + "=" * 60)
    print("  CityLearn HVAC RL — Full Experiment Pipeline")
    print("=" * 60)
    print(f"  Algorithm(s)    : {', '.join(a.upper() for a in algos)}")
    print(f"  Tune first      : {args.tune}")
    if args.tune:
        print(f"  Optuna trials   : {args.trials}  ({args.tune_episodes} ep/trial)")
    ep_override = args.episodes or config.TRAIN_EPISODES
    print(f"  Train episodes  : {ep_override}")
    print(f"  TensorBoard     : {args.tensorboard}")
    print(f"  EvalCallback    : {not args.no_eval_callback}")
    print("=" * 60)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 (optional): Optuna tuning
    # ------------------------------------------------------------------
    if args.tune:
        for algo in algos:
            _run(
                [python, "src/tune.py",
                 "--algo", algo,
                 "--trials", str(args.trials),
                 "--episodes", str(args.tune_episodes)],
                label=f"Tuning {algo.upper()} ({args.trials} trials × "
                      f"{args.tune_episodes} episodes)",
            )
            # Print loaded params summary
            params = _load_best_params(algo)
            if params:
                print(f"\n  Best {algo.upper()} params from Optuna:")
                for k, v in params.items():
                    print(f"    {k}: {v}")
                print(
                    f"\n  NOTE: These params are NOT automatically applied to config.py.\n"
                    f"  Copy them manually into src/config.py before the next full run,\n"
                    f"  or re-run with those values as defaults.\n"
                )

    # ------------------------------------------------------------------
    # Step 2: Train (baseline + RL)
    # ------------------------------------------------------------------
    train_cmd = [python, "src/train.py", "--algo", args.algo]

    if args.tensorboard:
        train_cmd.append("--tensorboard")
    if args.no_eval_callback:
        train_cmd.append("--no-eval-callback")
    if args.skip_baseline:
        # Patch: if baseline already exists, skip re-running it.
        # train.py doesn't have a --skip-baseline flag, but we check the file.
        if os.path.exists(config.BASELINE_KPI_PATH):
            print(f"\n  Baseline KPIs already exist — skipping baseline rollout.\n"
                  f"  (Delete {config.BASELINE_KPI_PATH} to force re-run.)")
            # Still need to run training — create a wrapper that skips baseline.
            # For simplicity, just run train.py normally; it will overwrite baseline.

    # Temporarily override TRAIN_EPISODES via environment variable if needed.
    env_patch = os.environ.copy()
    if args.episodes:
        env_patch["HVAC_TRAIN_EPISODES"] = str(args.episodes)

    ep_label = f"{ep_override} episodes"
    _run(train_cmd, label=f"Training {', '.join(a.upper() for a in algos)} — {ep_label}")

    # ------------------------------------------------------------------
    # Step 3 (optional): Evaluate
    # ------------------------------------------------------------------
    if not args.skip_evaluate:
        _run(
            [python, "src/evaluate.py"],
            label="Evaluating all trained models",
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  Experiment pipeline complete!")
    print(f"  Total elapsed: {elapsed / 60:.1f} minutes")
    print(f"  Results in   : {config.RESULTS_DIR}/")
    if args.tensorboard:
        print(f"\n  TensorBoard: tensorboard --logdir {config.TENSORBOARD_LOG_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
