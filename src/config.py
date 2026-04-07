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

# Built-in CityLearn dataset name.
# 'citylearn_challenge_2023_phase_2_local_evaluation' ships with CityLearn
# and contains Building_1, Building_2, and Building_3.
# TODO: swap in another schema name or a path to a custom JSON file if needed.
SCHEMA_NAME: str = "citylearn_challenge_2023_phase_2_local_evaluation"

# Which building to control. All others are excluded from the simulation.
# TODO: change to "Building_2" or "Building_3" to experiment with other buildings.
BUILDING_TO_USE: str = "Building_1"

# ---------------------------------------------------------------------------
# MDP — reward function
# ---------------------------------------------------------------------------

# Half-width of the comfort temperature band around the setpoint (degrees C).
# Comfort penalty is zero when |T_indoor - T_setpoint| <= COMFORT_BAND.
COMFORT_BAND: float = 1.0

# Weights for the two terms in the reward:
#   reward = -(REWARD_ENERGY_WEIGHT * net_electricity
#              + REWARD_COMFORT_WEIGHT * comfort_penalty)
# Must sum to 1.0 (or any positive values — they are relative).
# TODO: try REWARD_ENERGY_WEIGHT=0.7 / REWARD_COMFORT_WEIGHT=0.3 to
#       prioritise energy savings, or flip the ratio to prioritise comfort.
REWARD_ENERGY_WEIGHT: float = 0.5
REWARD_COMFORT_WEIGHT: float = 0.5

# ---------------------------------------------------------------------------
# MDP — state / observation space
# ---------------------------------------------------------------------------

# Exact set of observations to activate in the schema.
# These correspond directly to the state variables described in the project doc:
#   indoor/outdoor temperature, time info, electricity consumption, storage SOC.
# All other observations in the built-in schema will be set to inactive.
# TODO: add "outdoor_dry_bulb_temperature_predicted_6h" etc. for a richer state.
ACTIVE_OBSERVATIONS: list = [
    "month",                                   # calendar — cyclical encoding
    "day_type",                                # calendar — weekday/weekend
    "hour",                                    # calendar — time of day
    "outdoor_dry_bulb_temperature",            # weather
    "indoor_dry_bulb_temperature",             # building comfort state
    "indoor_dry_bulb_temperature_set_point",   # building comfort target
    "net_electricity_consumption",             # energy signal
    "cooling_storage_soc",                     # storage state
]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Number of full simulation episodes to train on.
# Each episode is one full year of hourly data (~8760 timesteps).
# Set to 2 for a quick prototype demo; increase to 10–15 for a final run.
# TODO: increase to 10–15 for the final project run if compute allows.
TRAIN_EPISODES: int = 2

# Number of evaluation episodes after training (1 is enough for KPI comparison).
EVAL_EPISODES: int = 1

# ---------------------------------------------------------------------------
# SAC hyperparameters (Stable Baselines3)
# ---------------------------------------------------------------------------

SAC_LEARNING_RATE: float = 3e-4
SAC_BATCH_SIZE: int = 256
SAC_BUFFER_SIZE: int = 100_000

# Policy network architecture: list of hidden layer sizes.
# [256, 256] is the SB3 default; reduce to [128, 128] for faster experiments.
SAC_NET_ARCH: list = [256, 256]

# How often (in timesteps) to print a training progress update.
SAC_LOG_INTERVAL: int = 1  # log every episode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# All output (models, plots, CSVs) is written here.
RESULTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# Saved SAC model (SB3 saves as a .zip file).
MODEL_SAVE_PATH: str = os.path.join(RESULTS_DIR, "sac_hvac_model")

# Baseline KPI CSV (written by train.py, read by evaluate.py).
BASELINE_KPI_PATH: str = os.path.join(RESULTS_DIR, "baseline_kpis.csv")

# RL agent KPI CSV (written by evaluate.py).
RL_KPI_PATH: str = os.path.join(RESULTS_DIR, "rl_kpis.csv")

# SB3 Monitor log directory (used to plot training reward curve).
MONITOR_LOG_DIR: str = os.path.join(RESULTS_DIR, "monitor_logs")

# ---------------------------------------------------------------------------
# Baseline agent (Rule-Based Controller)
# ---------------------------------------------------------------------------

# Hours during which the RBC charges cooling storage (cheap off-peak power).
# Using 24-hour clock values; these hours run from 22:00 through 05:00.
RBC_CHARGE_HOURS: list = list(range(22, 25)) + list(range(0, 6))  # [22,23,24,0..5]

# Hours during which the RBC discharges storage to offset peak demand.
RBC_DISCHARGE_HOURS: list = list(range(12, 19))  # [12..18]

# Charge and discharge action values (normalised to [-1, 1]).
RBC_CHARGE_ACTION: float = 1.0
RBC_DISCHARGE_ACTION: float = -1.0
RBC_IDLE_ACTION: float = 0.0

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# DPI for saved figures.
FIGURE_DPI: int = 150

# How many timesteps of the evaluation episode to show in temperature traces.
# Set to None to show the full episode.
TEMPERATURE_TRACE_STEPS: int = 7 * 24  # first 7 days (1 week of hourly data)
