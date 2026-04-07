"""
rl_agent.py
-----------
SAC (Soft Actor-Critic) agent training and evaluation using Stable Baselines3.

Algorithm choice: SAC
  - Continuous action space → SAC is natural; DQN would require discretisation.
  - Off-policy with a replay buffer → more sample-efficient than PPO for
    long energy simulations (8 760 steps per episode).
  - Entropy regularisation → automatic exploration without tuning epsilon.
  - Officially supported by CityLearn via NormalizedObservationWrapper +
    StableBaselines3Wrapper (see CityLearn 2.5.0 quickstart docs).

Training flow
  1. make_env() creates the wrapped single-building CityLearnEnv.
  2. A Monitor wrapper logs per-episode rewards to disk for plotting.
  3. SAC is instantiated with MlpPolicy and the hyperparameters from config.
  4. model.learn() runs for TRAIN_EPISODES × env.time_steps total timesteps.
  5. The trained model is saved to MODEL_SAVE_PATH.

Evaluation flow
  1. A fresh env is created (to reset all internal state).
  2. The saved model is loaded.
  3. One deterministic episode is rolled out.
  4. env.unwrapped.evaluate() returns normalised KPIs for comparison.
"""

import os
import sys
from typing import Dict, Any, List

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src import config
from src.env_setup import make_env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sac() -> SAC:
    """
    Train a SAC agent on the single-building CityLearn environment.

    Progress is printed to the console at the end of each episode.
    The trained model is saved to config.MODEL_SAVE_PATH.

    Returns
    -------
    stable_baselines3.SAC
        The trained SAC model (also saved to disk as a .zip file).
    """
    print("\n" + "=" * 55)
    print("  Training SAC Agent")
    print("=" * 55)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MONITOR_LOG_DIR, exist_ok=True)

    # Build the wrapped environment.
    env = make_env()

    # Wrap with Monitor so SB3 logs episode rewards and lengths to disk.
    # These logs are later read by utils.plot_training_rewards().
    env = Monitor(env, filename=os.path.join(config.MONITOR_LOG_DIR, "train"))

    # Determine total timesteps from the environment.
    # env.unwrapped gives us the raw CityLearnEnv.
    try:
        steps_per_episode = env.unwrapped.time_steps
    except AttributeError:
        # Fallback: assume one year of hourly data.
        steps_per_episode = 8760
        print(f"  [SAC] Could not read env.time_steps; assuming {steps_per_episode}.")

    total_timesteps = config.TRAIN_EPISODES * steps_per_episode
    print(f"  Episodes       : {config.TRAIN_EPISODES}")
    print(f"  Steps/episode  : {steps_per_episode}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Policy network : MLP {config.SAC_NET_ARCH}")
    print(f"  Learning rate  : {config.SAC_LEARNING_RATE}")
    print(f"  Batch size     : {config.SAC_BATCH_SIZE}")
    print(f"  Buffer size    : {config.SAC_BUFFER_SIZE}")
    print()

    # Instantiate SAC with MlpPolicy (fully connected actor/critic networks).
    # policy_kwargs lets us customise the network architecture.
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.SAC_LEARNING_RATE,
        batch_size=config.SAC_BATCH_SIZE,
        buffer_size=config.SAC_BUFFER_SIZE,
        policy_kwargs={"net_arch": config.SAC_NET_ARCH},
        verbose=1,                          # prints one line per log_interval
        # TODO: add tensorboard_log=config.RESULTS_DIR to enable TensorBoard.
    )

    # Train the agent.
    # log_interval=1 means SB3 prints a summary after every episode.
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=config.SAC_LOG_INTERVAL,
    )

    # Save the trained model to disk.
    model.save(config.MODEL_SAVE_PATH)
    print(f"\n  [SAC] Model saved to: {config.MODEL_SAVE_PATH}.zip")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_sac(model: SAC = None) -> Dict[str, Any]:
    """
    Evaluate the trained SAC agent for one deterministic episode.

    If `model` is None, the model is loaded from config.MODEL_SAVE_PATH.
    A fresh environment is always created to ensure no state leaks from
    training.

    Parameters
    ----------
    model : stable_baselines3.SAC, optional
        A trained SAC model. If None, loads from disk.

    Returns
    -------
    dict with keys:
      'kpis'            : pd.DataFrame from env.evaluate() (KPI pivot table)
      'episode_reward'  : float, total reward across the episode
      'rewards_per_step': list of float, per-timestep rewards
      'indoor_temps'    : list of float, indoor temperatures this episode
      'setpoints'       : list of float, temperature setpoints this episode
    """
    print("\n" + "=" * 55)
    print("  Evaluating SAC Agent")
    print("=" * 55)

    # Load model from disk if not provided.
    if model is None:
        if not os.path.exists(config.MODEL_SAVE_PATH + ".zip"):
            raise FileNotFoundError(
                f"No saved model found at '{config.MODEL_SAVE_PATH}.zip'. "
                "Run src/train.py first."
            )
        print(f"  [SAC] Loading model from: {config.MODEL_SAVE_PATH}.zip")
        model = SAC.load(config.MODEL_SAVE_PATH)

    # Create a fresh environment for evaluation.
    env = make_env()

    # Bind the model's internal environment to the new env for predict().
    model.set_env(env)

    observations, _ = env.reset()
    total_reward = 0.0
    rewards_per_step: List[float] = []
    indoor_temps: List[float] = []
    setpoints: List[float] = []

    step = 0
    # env.unwrapped.terminated is the CityLearnEnv termination flag.
    while not env.unwrapped.terminated:
        # deterministic=True disables the stochastic action sampling used
        # during training; the agent picks the mode of its action distribution.
        actions, _ = model.predict(observations, deterministic=True)
        observations, reward, terminated, truncated, info = env.step(actions)

        r = float(reward)
        total_reward += r
        rewards_per_step.append(r)

        # Record temperature from the underlying CityLearnEnv.
        try:
            raw_env = env.unwrapped
            b = raw_env.buildings[0]
            indoor_temps.append(float(b.indoor_dry_bulb_temperature[-1]))
            setpoints.append(float(b.indoor_dry_bulb_temperature_set_point[-1]))
        except (AttributeError, IndexError):
            pass

        step += 1
        if step % 1000 == 0:
            print(f"  [SAC] Eval step {step:5d} | cumulative reward: {total_reward:.2f}")

    print(f"  [SAC] Evaluation complete — {step} steps | total reward: {total_reward:.4f}")

    # Retrieve KPIs from the underlying CityLearnEnv.
    raw_env = env.unwrapped
    kpis_raw = raw_env.evaluate()

    try:
        kpis = kpis_raw.pivot(
            index="cost_function", columns="name", values="value"
        ).round(4)
        kpis = kpis.dropna(how="all")
    except Exception as exc:
        print(f"  [SAC] Warning: could not pivot KPI table ({exc}). Using raw output.")
        kpis = kpis_raw

    print("\n  RL Agent KPIs (relative to no-control reference = 1.0):")
    print(kpis.to_string())
    print()

    return {
        "kpis": kpis,
        "episode_reward": total_reward,
        "rewards_per_step": rewards_per_step,
        "indoor_temps": indoor_temps,
        "setpoints": setpoints,
    }
