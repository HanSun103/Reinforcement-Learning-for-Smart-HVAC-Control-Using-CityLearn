"""
rl_agent.py
-----------
RL agent training and evaluation using Stable Baselines3.

Three algorithms are supported:

SAC  — Soft Actor-Critic (off-policy, stochastic)
  Stores all past transitions in a replay buffer; re-uses each sample many
  times → very sample-efficient.  Entropy regularisation gives built-in,
  automatic exploration.

PPO  — Proximal Policy Optimisation (on-policy, stochastic)
  Collects a fresh rollout each update (one full episode) and discards it.
  Simpler to tune; naturally episodic.  Needs more total timesteps than SAC.

TD3  — Twin Delayed DDPG (off-policy, deterministic)
  Uses two critics to reduce Q-value overestimation bias and delays policy
  updates relative to critic updates.  Lower variance than SAC but may
  need more explicit exploration noise.

Additional features
  - EvalCallback: evaluates on a separate env every episode and saves the
    best-so-far model to results/best_{algo}/.
  - TensorBoard: pass --tensorboard flag to train.py; view with
    `tensorboard --logdir results/tensorboard`.

Usage
-----
    from src.rl_agent import train_agent, evaluate_agent

    model = train_agent("sac")        # or "ppo" / "td3"
    results = evaluate_agent("sac")   # loads best model from disk
"""

import os
import sys
from typing import Dict, Any, List, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    StopTrainingOnNoModelImprovement,
)

from src import config
from src.env_setup import make_env, get_building_setpoints


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

def _algo_config(algo: str) -> dict:
    """
    Return the SB3 class and hyperparameters for *algo*.

    Parameters
    ----------
    algo : str
        One of ``"sac"``, ``"ppo"``, or ``"td3"`` (case-insensitive).

    Returns
    -------
    dict
        Keys: cls, lr, batch_size, net_arch, model_path, best_model_dir,
              monitor_dir, log_interval, extra_kwargs
    """
    algo = algo.lower()
    results = config.RESULTS_DIR

    if algo == "sac":
        return dict(
            cls=SAC,
            lr=config.SAC_LEARNING_RATE,
            batch_size=config.SAC_BATCH_SIZE,
            net_arch=config.SAC_NET_ARCH,
            model_path=config.SAC_MODEL_SAVE_PATH,
            best_model_dir=os.path.join(results, "best_sac"),
            monitor_dir=config.SAC_MONITOR_LOG_DIR,
            log_interval=config.SAC_LOG_INTERVAL,
            extra_kwargs=dict(buffer_size=config.SAC_BUFFER_SIZE),
        )
    elif algo == "ppo":
        return dict(
            cls=PPO,
            lr=config.PPO_LEARNING_RATE,
            batch_size=config.PPO_BATCH_SIZE,
            net_arch=config.PPO_NET_ARCH,
            model_path=config.PPO_MODEL_SAVE_PATH,
            best_model_dir=os.path.join(results, "best_ppo"),
            monitor_dir=config.PPO_MONITOR_LOG_DIR,
            log_interval=config.PPO_LOG_INTERVAL,
            extra_kwargs=dict(
                n_steps=config.PPO_N_STEPS,
                n_epochs=config.PPO_N_EPOCHS,
                clip_range=config.PPO_CLIP_RANGE,
                gae_lambda=config.PPO_GAE_LAMBDA,
            ),
        )
    elif algo == "td3":
        return dict(
            cls=TD3,
            lr=config.TD3_LEARNING_RATE,
            batch_size=config.TD3_BATCH_SIZE,
            net_arch=config.TD3_NET_ARCH,
            model_path=config.TD3_MODEL_SAVE_PATH,
            best_model_dir=os.path.join(results, "best_td3"),
            monitor_dir=config.TD3_MONITOR_LOG_DIR,
            log_interval=config.TD3_LOG_INTERVAL,
            extra_kwargs=dict(
                buffer_size=config.TD3_BUFFER_SIZE,
                policy_delay=config.TD3_POLICY_DELAY,
                target_policy_noise=config.TD3_TARGET_POLICY_NOISE,
                target_noise_clip=config.TD3_TARGET_NOISE_CLIP,
            ),
        )
    else:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose 'sac', 'ppo', or 'td3'.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_agent(
    algo: str = "sac",
    use_tensorboard: bool = False,
    use_eval_callback: bool = True,
    override_kwargs: Optional[dict] = None,
):
    """
    Train an RL agent on the single-building CityLearn environment.

    Parameters
    ----------
    algo : str
        ``"sac"``, ``"ppo"``, or ``"td3"`` (case-insensitive).
    use_tensorboard : bool
        If True, log training metrics to config.TENSORBOARD_LOG_DIR.
        View with: ``tensorboard --logdir results/tensorboard``
    use_eval_callback : bool
        If True, evaluate on a separate env every episode and save the
        best model to ``results/best_{algo}/best_model.zip``.
    override_kwargs : dict, optional
        Override any hyperparameter in the model constructor.  Used by
        the Optuna tuning script.

    Returns
    -------
    Trained SB3 model (also saved to disk as a .zip file).
    """
    cfg = _algo_config(algo)
    label = algo.upper()

    print(f"\n{'=' * 55}")
    print(f"  Training {label} Agent")
    print(f"{'=' * 55}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg["monitor_dir"], exist_ok=True)
    os.makedirs(cfg["best_model_dir"], exist_ok=True)

    # --- Training environment ---
    train_env = make_env()
    train_env = Monitor(
        train_env,
        filename=os.path.join(cfg["monitor_dir"], "train"),
    )

    # --- Evaluation environment (separate instance — no state bleed) ---
    eval_env = make_env() if use_eval_callback else None

    # --- Episode / timestep counts ---
    try:
        steps_per_episode = train_env.unwrapped.time_steps
    except AttributeError:
        steps_per_episode = 8760
        print(f"  [{label}] Could not read env.time_steps; assuming {steps_per_episode}.")

    # For PPO: n_steps must equal the episode length for clean episodic updates.
    if algo.lower() == "ppo":
        n_steps = steps_per_episode
        bs = cfg["batch_size"]
        if n_steps % bs != 0:
            bs = max(d for d in range(1, bs + 1) if n_steps % d == 0)
            print(f"  [PPO] batch_size adjusted to {bs} (must divide n_steps={n_steps}).")
        cfg["extra_kwargs"]["n_steps"] = n_steps
        cfg["batch_size"] = bs

    total_timesteps = config.TRAIN_EPISODES * steps_per_episode

    print(f"  Episodes        : {config.TRAIN_EPISODES}")
    print(f"  Steps/episode   : {steps_per_episode}")
    print(f"  Total timesteps : {total_timesteps}")
    print(f"  Policy network  : MLP {cfg['net_arch']}")
    print(f"  Learning rate   : {cfg['lr']}")
    print(f"  Batch size      : {cfg['batch_size']}")
    print(f"  TensorBoard     : {'yes — ' + config.TENSORBOARD_LOG_DIR if use_tensorboard else 'no'}")
    print(f"  EvalCallback    : {'yes — best model saved to ' + cfg['best_model_dir'] if use_eval_callback else 'no'}")
    print()

    # --- Model constructor kwargs ---
    model_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=cfg["lr"],
        batch_size=cfg["batch_size"],
        policy_kwargs={"net_arch": cfg["net_arch"]},
        verbose=1,
        **cfg["extra_kwargs"],
    )
    if use_tensorboard:
        os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
        model_kwargs["tensorboard_log"] = config.TENSORBOARD_LOG_DIR

    if override_kwargs:
        model_kwargs.update(override_kwargs)

    model = cfg["cls"](**model_kwargs)

    # --- Callbacks ---
    callbacks = []
    if use_eval_callback and eval_env is not None:
        # Evaluate after every episode; save the best model automatically.
        # StopTrainingOnNoModelImprovement halts if no improvement for
        # MAX_NO_IMPROVEMENT_EVALS consecutive evaluations.
        stop_cb = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=3,
            verbose=1,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=cfg["best_model_dir"],
            log_path=cfg["best_model_dir"],
            eval_freq=steps_per_episode,     # evaluate once per episode
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            callback_after_eval=stop_cb,
            verbose=1,
        )
        callbacks.append(eval_cb)

    cb = CallbackList(callbacks) if callbacks else None

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=cfg["log_interval"],
        callback=cb,
    )

    model.save(cfg["model_path"])
    print(f"\n  [{label}] Final model  : {cfg['model_path']}.zip")
    if use_eval_callback:
        best = os.path.join(cfg["best_model_dir"], "best_model.zip")
        print(f"  [{label}] Best model   : {best}")
    if use_tensorboard:
        print(f"  [{label}] TensorBoard  : tensorboard --logdir {config.TENSORBOARD_LOG_DIR}")

    return model


# Convenience wrappers
def train_sac(**kwargs): return train_agent("sac", **kwargs)
def train_ppo(**kwargs): return train_agent("ppo", **kwargs)
def train_td3(**kwargs): return train_agent("td3", **kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    algo: str = "sac",
    model=None,
    use_best: bool = True,
) -> Dict[str, Any]:
    """
    Run one deterministic evaluation episode and collect KPIs + traces.

    Parameters
    ----------
    algo : str
        ``"sac"``, ``"ppo"``, or ``"td3"`` (case-insensitive).
    model : SB3 model, optional
        A pre-loaded model.  If None, loads from disk.
    use_best : bool
        If True and the best_model.zip from EvalCallback exists, load that
        instead of the final model (the best checkpoint usually outperforms
        the final weights).

    Returns
    -------
    dict with keys:
      'kpis', 'episode_reward', 'rewards_per_step',
      'indoor_temps', 'setpoints'
    """
    import pandas as pd

    cfg = _algo_config(algo)
    label = algo.upper()

    print(f"\n{'=' * 55}")
    print(f"  Evaluating {label} Agent")
    print(f"{'=' * 55}")

    if model is None:
        best_path = os.path.join(cfg["best_model_dir"], "best_model.zip")
        final_path = cfg["model_path"] + ".zip"

        if use_best and os.path.exists(best_path):
            load_path = cfg["best_model_dir"] + "/best_model"
            print(f"  [{label}] Loading best checkpoint: {best_path}")
        elif os.path.exists(final_path):
            load_path = cfg["model_path"]
            print(f"  [{label}] Loading final model: {final_path}")
        else:
            raise FileNotFoundError(
                f"No {label} model found. Run `python src/train.py --algo {algo}` first."
            )
        model = cfg["cls"].load(load_path)

    env = make_env()
    model.set_env(env)

    observations, _ = env.reset()
    total_reward = 0.0
    rewards_per_step: List[float] = []

    step = 0
    while not env.unwrapped.terminated:
        actions, _ = model.predict(observations, deterministic=True)
        observations, reward, terminated, truncated, info = env.step(actions)

        total_reward += float(reward)
        rewards_per_step.append(float(reward))
        step += 1
        if step % 200 == 0:
            print(f"  [{label}] Step {step:4d} | cumulative reward: {total_reward:.2f}")

    print(f"  [{label}] Done — {step} steps | total reward: {total_reward:.4f}")

    # Extract temperature history after the episode (more reliable than
    # step-by-step collection for LSTMDynamicsBuilding).
    b = env.unwrapped.buildings[0]
    try:
        indoor_temps = [float(x) for x in b.indoor_dry_bulb_temperature[:step]]
    except (AttributeError, TypeError):
        indoor_temps = []
    setpoints = get_building_setpoints(b, step)

    kpis_raw = env.unwrapped.evaluate()
    try:
        kpis = kpis_raw.pivot(
            index="cost_function", columns="name", values="value"
        ).round(4)
        kpis = kpis.dropna(how="all")
    except Exception as exc:
        print(f"  [{label}] Warning: could not pivot KPI table ({exc}).")
        kpis = kpis_raw

    print(f"\n  {label} KPIs (normalised; < 1.0 = improvement):")
    print(kpis.to_string())
    print()

    return {
        "kpis": kpis,
        "episode_reward": total_reward,
        "rewards_per_step": rewards_per_step,
        "indoor_temps": indoor_temps,
        "setpoints": setpoints,
    }


# Convenience wrappers (backward compatible)
def evaluate_sac(model=None, **kwargs): return evaluate_agent("sac", model=model, **kwargs)
def evaluate_ppo(model=None, **kwargs): return evaluate_agent("ppo", model=model, **kwargs)
def evaluate_td3(model=None, **kwargs): return evaluate_agent("td3", model=model, **kwargs)
