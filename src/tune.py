"""
tune.py
-------
Hyperparameter optimisation using Optuna.

Optuna runs N independent trials.  Each trial:
  1. Samples a hyperparameter configuration from the search space.
  2. Trains the chosen RL algorithm for OPTUNA_N_EPISODES episodes.
  3. Reports the mean episode reward as the objective (maximise).

The best configuration is saved to results/best_params.json and printed
at the end.  You can then copy the values into config.py for the full run.

Usage
-----
    # Tune SAC (default, 20 trials):
    python src/tune.py

    # Tune PPO with 30 trials:
    python src/tune.py --algo ppo --trials 30

    # Tune TD3 quickly (5 trials, 1 episode each):
    python src/tune.py --algo td3 --trials 5 --episodes 1

Prerequisites
-------------
    pip install optuna
"""

import argparse
import json
import os
import sys
import warnings

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print(
        "\n  Optuna is not installed. Run:\n"
        "      pip install optuna\n"
        "  then re-run this script.\n"
    )
    sys.exit(1)

from src import config
from src.env_setup import make_env


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def _suggest_sac(trial) -> dict:
    """Optuna search space for SAC."""
    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_units    = trial.suggest_categorical("net_units", [128, 256, 512])
    n_layers   = trial.suggest_int("net_layers", 1, 3)
    buf_size   = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])
    return dict(
        learning_rate=lr,
        batch_size=batch_size,
        policy_kwargs={"net_arch": [n_units] * n_layers},
        buffer_size=buf_size,
        verbose=0,
    )


def _suggest_ppo(trial, steps_per_episode: int) -> dict:
    """Optuna search space for PPO."""
    lr         = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    n_epochs   = trial.suggest_int("n_epochs", 5, 20)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    n_units    = trial.suggest_categorical("net_units", [128, 256, 512])
    n_layers   = trial.suggest_int("net_layers", 1, 3)

    # batch_size must divide n_steps; pick the largest valid value <= suggested.
    bs_choice  = trial.suggest_categorical("batch_size", [60, 72, 90, 120, 180])
    bs = max(d for d in range(1, bs_choice + 1) if steps_per_episode % d == 0)

    return dict(
        learning_rate=lr,
        n_steps=steps_per_episode,
        batch_size=bs,
        n_epochs=n_epochs,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        policy_kwargs={"net_arch": [n_units] * n_layers},
        verbose=0,
    )


def _suggest_td3(trial) -> dict:
    """Optuna search space for TD3."""
    lr           = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_units      = trial.suggest_categorical("net_units", [128, 256, 512])
    n_layers     = trial.suggest_int("net_layers", 1, 3)
    buf_size     = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])
    policy_delay = trial.suggest_int("policy_delay", 1, 4)
    target_noise = trial.suggest_float("target_policy_noise", 0.1, 0.3)
    return dict(
        learning_rate=lr,
        batch_size=batch_size,
        policy_kwargs={"net_arch": [n_units] * n_layers},
        buffer_size=buf_size,
        policy_delay=policy_delay,
        target_policy_noise=target_noise,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _make_objective(algo: str, n_episodes: int):
    """
    Return an Optuna objective function for *algo*.

    The objective trains for *n_episodes* and returns the mean episode reward
    (higher = better → ``direction="maximize"``).
    """
    from stable_baselines3 import SAC, PPO, TD3
    from stable_baselines3.common.monitor import Monitor

    algo = algo.lower()
    cls_map = {"sac": SAC, "ppo": PPO, "td3": TD3}
    cls = cls_map[algo]

    def objective(trial):
        try:
            env = make_env()
            env = Monitor(env)

            try:
                steps_per_episode = env.unwrapped.time_steps
            except AttributeError:
                steps_per_episode = 8760

            total_timesteps = n_episodes * steps_per_episode

            if algo == "sac":
                kwargs = _suggest_sac(trial)
            elif algo == "ppo":
                kwargs = _suggest_ppo(trial, steps_per_episode)
            else:
                kwargs = _suggest_td3(trial)

            model = cls(policy="MlpPolicy", env=env, **kwargs)
            model.learn(total_timesteps=total_timesteps, log_interval=999)

            # Evaluate deterministically for one episode.
            eval_env = make_env()
            model.set_env(eval_env)
            obs, _ = eval_env.reset()
            total_reward = 0.0
            while not eval_env.unwrapped.terminated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, *_ = eval_env.step(action)
                total_reward += float(reward)

            return total_reward

        except Exception as exc:
            # Prune the trial if the configuration raises an error (e.g. bad
            # batch_size / n_steps combination for PPO).
            warnings.warn(f"Trial {trial.number} failed: {exc}")
            raise optuna.exceptions.TrialPruned()

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for CityLearn RL agents."
    )
    parser.add_argument(
        "--algo",
        choices=["sac", "ppo", "td3"],
        default="sac",
        help="Algorithm to tune (default: sac).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=config.OPTUNA_N_TRIALS,
        help=f"Number of Optuna trials (default: {config.OPTUNA_N_TRIALS}).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.OPTUNA_N_EPISODES,
        help=f"Training episodes per trial (default: {config.OPTUNA_N_EPISODES}).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    algo = args.algo.lower()

    print("\n" + "=" * 60)
    print("  Optuna Hyperparameter Search")
    print("=" * 60)
    print(f"  Algorithm : {algo.upper()}")
    print(f"  Trials    : {args.trials}")
    print(f"  Episodes  : {args.episodes} per trial")
    print(f"  DB path   : {config.OPTUNA_STUDY_PATH}")
    print("=" * 60 + "\n")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    storage = f"sqlite:///{config.OPTUNA_STUDY_PATH}"
    study = optuna.create_study(
        study_name=f"hvac_{algo}",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    objective = _make_objective(algo, args.episodes)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # --- Results ---
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"  Best trial: #{best.number}  |  reward = {best.value:.4f}")
    print("  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("=" * 60)

    # Save to JSON.
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    result = {
        "algo": algo,
        "best_trial": best.number,
        "best_reward": best.value,
        "params": best.params,
    }
    with open(config.OPTUNA_BEST_PARAMS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Best params saved to: {config.OPTUNA_BEST_PARAMS_PATH}")
    print("  Copy the values into config.py for the full training run.\n")


if __name__ == "__main__":
    main()
