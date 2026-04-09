"""
config.py
---------
Central configuration for the CityLearn HVAC RL project.

All tunable hyperparameters, file paths, and environment settings live here.
Edit this file to change training behaviour without touching other source files.
"""

import os

# ---------------------------------------------------------------------------
# Environment / dataset
# ---------------------------------------------------------------------------

SCHEMA_NAME: str = "citylearn_challenge_2023_phase_2_local_evaluation"

# Single-building mode: set to a building name to control only that building.
# Set MULTI_BUILDING = True below to control all buildings simultaneously.
BUILDING_TO_USE: str = "Building_1"

# Multi-building control: when True, all buildings in the schema are included
# and the action/observation space expands accordingly.
# The RL agent then acts as a centralised controller for the whole district.
MULTI_BUILDING: bool = False

# ---------------------------------------------------------------------------
# MDP — reward function
# ---------------------------------------------------------------------------

# Comfort temperature band half-width (°C).
COMFORT_BAND: float = 1.0

# Reward formula:
#   reward = -(ENERGY_WEIGHT * net_electricity
#              + COMFORT_WEIGHT * comfort_penalty
#              + CARBON_WEIGHT  * carbon_penalty)
#
# carbon_penalty = carbon_intensity (kg CO₂/kWh) × net_electricity (kWh)
# Set REWARD_CARBON_WEIGHT = 0.0 to disable the carbon term entirely.
# The three weights do not need to sum to 1.0; they are relative.
REWARD_ENERGY_WEIGHT:  float = 0.4
REWARD_COMFORT_WEIGHT: float = 0.4
REWARD_CARBON_WEIGHT:  float = 0.2   # enables carbon-aware control

# ---------------------------------------------------------------------------
# MDP — state / observation space
# ---------------------------------------------------------------------------

# Whether to include short-horizon weather and price forecasts in the state.
# CityLearn provides *_predicted_1/_2/_3/_6 variants for several signals.
# These give the agent look-ahead information — a major performance booster
# for storage scheduling tasks.
USE_PREDICTION_OBS: bool = True

ACTIVE_OBSERVATIONS: list = [
    # --- Time context ---
    "month",
    "day_type",
    "hour",
    # --- Current weather ---
    "outdoor_dry_bulb_temperature",
    "outdoor_relative_humidity",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
    # --- Weather forecasts (1 h, 2 h, 6 h ahead) ---
    # Only included when USE_PREDICTION_OBS = True (see below).
    # --- Indoor comfort ---
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_set_point",
    # --- Economic & carbon signals ---
    "electricity_pricing",
    "carbon_intensity",
    # --- Economic forecasts (1 h ahead) ---
    # Only included when USE_PREDICTION_OBS = True (see below).
    # --- Load profile ---
    "non_shiftable_load",
    "solar_generation",
    # --- Storage states ---
    "cooling_storage_soc",
    "electrical_storage_soc",
    # --- Net result ---
    "net_electricity_consumption",
]

# Prediction observations appended when USE_PREDICTION_OBS is enabled.
_PREDICTION_OBS: list = [
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_6",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_6",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_6",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "electricity_pricing_predicted_6",
]

if USE_PREDICTION_OBS:
    ACTIVE_OBSERVATIONS = ACTIVE_OBSERVATIONS + _PREDICTION_OBS

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAIN_EPISODES: int = 5
EVAL_EPISODES:  int = 1

# ---------------------------------------------------------------------------
# SAC hyperparameters (off-policy, continuous actions, entropy regularisation)
# ---------------------------------------------------------------------------

SAC_LEARNING_RATE: float = 3e-4
SAC_BATCH_SIZE:    int   = 256
SAC_BUFFER_SIZE:   int   = 100_000
SAC_NET_ARCH:      list  = [256, 256]
SAC_LOG_INTERVAL:  int   = 1

# ---------------------------------------------------------------------------
# PPO hyperparameters (on-policy, episodic rollouts)
# ---------------------------------------------------------------------------
# n_steps is set to env.time_steps at runtime to align updates with episodes.
# batch_size must divide n_steps; 60 divides 720 (the 2023 dataset length).

PPO_LEARNING_RATE: float = 3e-4
PPO_N_STEPS:       int   = 720    # overridden at runtime with env.time_steps
PPO_BATCH_SIZE:    int   = 60     # 720 / 60 = 12 minibatches per epoch
PPO_N_EPOCHS:      int   = 10
PPO_CLIP_RANGE:    float = 0.2
PPO_GAE_LAMBDA:    float = 0.95
PPO_NET_ARCH:      list  = [256, 256]
PPO_LOG_INTERVAL:  int   = 1

# ---------------------------------------------------------------------------
# TD3 hyperparameters (off-policy, deterministic policy, twin critics)
# ---------------------------------------------------------------------------
# TD3 (Twin Delayed DDPG) reduces Q-value overestimation via twin critics
# and delayed policy updates.  Deterministic policy → lower variance than SAC,
# but may need more explicit exploration noise.

TD3_LEARNING_RATE:         float = 3e-4
TD3_BATCH_SIZE:            int   = 256
TD3_BUFFER_SIZE:           int   = 100_000
TD3_NET_ARCH:              list  = [256, 256]
TD3_POLICY_DELAY:          int   = 2     # update actor every N critic steps
TD3_TARGET_POLICY_NOISE:   float = 0.2   # smoothing noise on target actions
TD3_TARGET_NOISE_CLIP:     float = 0.5
TD3_LOG_INTERVAL:          int   = 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

SAC_MODEL_SAVE_PATH: str = os.path.join(RESULTS_DIR, "sac_hvac_model")
PPO_MODEL_SAVE_PATH: str = os.path.join(RESULTS_DIR, "ppo_hvac_model")
TD3_MODEL_SAVE_PATH: str = os.path.join(RESULTS_DIR, "td3_hvac_model")
MODEL_SAVE_PATH:     str = SAC_MODEL_SAVE_PATH   # backward-compat alias

BASELINE_KPI_PATH: str = os.path.join(RESULTS_DIR, "baseline_kpis.csv")
SAC_KPI_PATH:      str = os.path.join(RESULTS_DIR, "sac_kpis.csv")
PPO_KPI_PATH:      str = os.path.join(RESULTS_DIR, "ppo_kpis.csv")
TD3_KPI_PATH:      str = os.path.join(RESULTS_DIR, "td3_kpis.csv")
RL_KPI_PATH:       str = SAC_KPI_PATH   # backward-compat alias

SAC_MONITOR_LOG_DIR: str = os.path.join(RESULTS_DIR, "monitor_logs", "sac")
PPO_MONITOR_LOG_DIR: str = os.path.join(RESULTS_DIR, "monitor_logs", "ppo")
TD3_MONITOR_LOG_DIR: str = os.path.join(RESULTS_DIR, "monitor_logs", "td3")
MONITOR_LOG_DIR:     str = SAC_MONITOR_LOG_DIR   # backward-compat alias

# TensorBoard log directory — run `tensorboard --logdir results/tensorboard`
# to launch the dashboard while training is in progress.
TENSORBOARD_LOG_DIR: str = os.path.join(RESULTS_DIR, "tensorboard")

# ---------------------------------------------------------------------------
# Hyperparameter optimisation (Optuna)
# ---------------------------------------------------------------------------

OPTUNA_N_TRIALS:   int = 20   # number of hyperparameter configurations to try
OPTUNA_N_EPISODES: int = 2    # training episodes per trial (keep short for speed)
OPTUNA_STUDY_PATH: str = os.path.join(RESULTS_DIR, "optuna_study.db")
OPTUNA_BEST_PARAMS_PATH: str = os.path.join(RESULTS_DIR, "best_params.json")

# ---------------------------------------------------------------------------
# Baseline agent (Rule-Based Controller)
# ---------------------------------------------------------------------------

RBC_CHARGE_HOURS:    list  = list(range(22, 25)) + list(range(0, 6))
RBC_DISCHARGE_HOURS: list  = list(range(12, 19))
RBC_CHARGE_ACTION:   float = 1.0
RBC_DISCHARGE_ACTION: float = -1.0
RBC_IDLE_ACTION:     float = 0.0

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FIGURE_DPI:              int = 150
TEMPERATURE_TRACE_STEPS: int = 7 * 24   # first week of the episode
