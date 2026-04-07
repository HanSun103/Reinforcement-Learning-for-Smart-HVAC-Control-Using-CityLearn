# Reinforcement Learning for Smart HVAC Control Using CityLearn

A course project that trains a **Soft Actor-Critic (SAC)** agent to control HVAC in a
single-building CityLearn environment. We compare the RL agent against a simple
**Rule-Based Controller (RBC)** baseline using CityLearn's built-in KPI framework.

---

## Project Goal

Control a building's cooling storage and heat pump to:
- **Reduce electricity consumption** (and cost)
- **Maintain thermal comfort** (keep indoor temperature near setpoint)

---

## MDP Design

| Component | Description |
|-----------|-------------|
| **State** | Indoor/outdoor temperature, hour, day type, electricity consumption, storage SOC — all min-max normalized to `[0, 1]` |
| **Action** | Continuous vector in `[-1, 1]` per device (cooling storage charge/discharge, heat pump) |
| **Reward** | `-(α × net_electricity + β × comfort_penalty)` where comfort penalty grows when indoor temp strays outside a `±1 °C` band around the setpoint |
| **Policy** | SAC (off-policy, entropy-regularized) via Stable Baselines3 |

---

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py          # all hyperparameters and file paths
│   ├── env_setup.py       # CityLearn env factory (schema + wrappers)
│   ├── reward.py          # ComfortEnergyReward (custom RewardFunction)
│   ├── baseline_agent.py  # Rule-Based Controller + run_baseline()
│   ├── rl_agent.py        # train_sac() + evaluate_sac()
│   ├── train.py           # entry point: run baseline then train RL
│   ├── evaluate.py        # entry point: compare RL vs baseline, save plots
│   └── utils.py           # plotting, metrics, logging helpers
├── results/               # auto-created; saved plots and CSVs land here
└── notebooks/             # optional Jupyter exploration notebooks
```

---

## Setup

> **Important**: do **not** run `pip install -r requirements.txt` directly.
> Use the install script below which creates a dedicated conda environment.
> This avoids three separate dependency conflicts described at the bottom of
> this section.

### 1. Open an **Anaconda Prompt** (not regular PowerShell)

Start → search "Anaconda Prompt" → open it.

### 2. Navigate to the project folder

```
cd "C:\Users\Han\OneDrive\Documents\学习\Master\MMAI\845\Project"
```

### 3. Run the install script

**Windows (Anaconda Prompt):**

```bat
install.bat
```

This creates a conda environment named `citylearn-rl`, installs all
dependencies in the correct order, and verifies each package at the end.
It takes 5–10 minutes (mostly PyTorch download).

**macOS / Linux:**

```bash
bash install.sh
```

### 4. Select the Python interpreter in Cursor / VS Code

After the install completes, press `Ctrl+Shift+P` → **Python: Select Interpreter**
and choose `citylearn-rl (conda)`.

### 5. Verify and run

```bat
conda activate citylearn-rl
python src/quick_test.py
```

---

### Manual install (step-by-step, if the script fails)

Run each line in an Anaconda Prompt:

```bash
# Create and activate a fresh environment
conda create -n citylearn-rl python=3.11 -y
conda activate citylearn-rl

# PyTorch (CPU) via conda — bundles correct MKL/TBB DLLs for Windows
conda install pytorch=2.3.1 cpuonly -c pytorch -y

# CityLearn without openstudio (not available on Windows/Python 3.11)
pip install citylearn==2.5.0 --no-deps

# CityLearn minimal deps + pinned gymnasium (CityLearn max is 0.28.1)
# platformdirs is a direct CityLearn runtime dep (skipped by --no-deps)
pip install "gymnasium==0.28.1" "scikit-learn==1.2.2" simplejson pyyaml platformdirs

# SB3 pinned to 2.2.1 (last version compatible with gymnasium 0.28)
# --no-deps prevents SB3 from auto-upgrading gymnasium to 1.x
pip install "stable-baselines3==2.2.1" --no-deps
pip install cloudpickle

# Utilities — numpy capped at <2.0.0 (CityLearn is not numpy-2 compatible)
pip install "matplotlib>=3.7.0" "pandas>=2.0.0" "numpy>=1.24.0,<2.0.0"
```

---

### Why three dependency conflicts exist

| Problem | Root cause | Fix |
|---------|-----------|-----|
| `openstudio` install fails | CityLearn declares it as a dep, but no PyPI wheel exists for Python 3.11 on Windows | `pip install citylearn --no-deps` — openstudio is only for EnergyPlus generation (not used here) |
| `gymnasium` version clash | SB3 ≥ 2.3 auto-upgrades gymnasium to 1.x; CityLearn requires ≤ 0.28.1 | Pin `stable-baselines3==2.2.1` and `gymnasium==0.28.1` |
| `torch` DLL crash in Anaconda | pip-installed PyTorch does not include the MKL/TBB DLLs that conda environments expect; also conflicts with conda-managed TBB | Use `conda install pytorch cpuonly` in a dedicated environment |

---

## Running the Project

All commands are run from the **project root** directory.

### Step 1 — Train (runs baseline first, then trains SAC)

```bash
python src/train.py
```

This will:
1. Run the Rule-Based Controller for one episode and save its KPIs.
2. Train the SAC agent for the configured number of episodes.
3. Save the trained model to `results/sac_hvac_model.zip`.

Training progress is printed to the console every episode.

### Step 2 — Evaluate and compare

```bash
python src/evaluate.py
```

This will:
1. Load the saved SAC model.
2. Run one deterministic evaluation episode.
3. Compare KPIs against the saved baseline.
4. Save plots and a summary CSV to `results/`.

---

## Configuration

Edit `src/config.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BUILDING_TO_USE` | `"Building_1"` | Which building from the dataset |
| `TRAIN_EPISODES` | `5` | Training episodes (each ~8760 timesteps) |
| `COMFORT_BAND` | `1.0` | Comfort tolerance in °C |
| `REWARD_ENERGY_WEIGHT` | `0.5` | Weight on energy penalty |
| `REWARD_COMFORT_WEIGHT` | `0.5` | Weight on comfort penalty |
| `SAC_LEARNING_RATE` | `3e-4` | SAC optimizer learning rate |

---

## Results

After running both scripts, `results/` contains:

| File | Description |
|------|-------------|
| `baseline_kpis.csv` | KPI table from the rule-based controller |
| `rl_kpis.csv` | KPI table from the trained SAC agent |
| `kpi_comparison.png` | Bar chart comparing both controllers |
| `training_rewards.png` | Episode reward curve during SAC training |
| `temperature_trace_baseline.png` | Indoor temperature vs setpoint (baseline) |
| `temperature_trace_rl.png` | Indoor temperature vs setpoint (RL agent) |
| `sac_hvac_model.zip` | Saved SAC model weights |

---

## Design Choices

- **SAC over DQN**: the action space is continuous (storage charge level), making
  SAC the natural choice. DQN would require discretizing actions.
- **SAC over PPO**: SAC is off-policy and reuses experience from a replay buffer,
  making it more sample-efficient for long energy simulations.
- **Single building, central agent**: simplifies the MDP; `central_agent=True`
  collapses multi-building observations/actions into flat vectors for SB3.
- **Custom reward**: CityLearn's default reward does not explicitly penalize comfort
  violations. Our `ComfortEnergyReward` adds a comfort band penalty.
- **NormalizedObservationWrapper**: puts all observations on the same scale `[0,1]`,
  improving neural network training stability.

---

## References

- CityLearn documentation: https://www.citylearn.net
- Stable Baselines3: https://stable-baselines3.readthedocs.io
- Haarnoja et al. (2018), "Soft Actor-Critic" — the SAC algorithm
