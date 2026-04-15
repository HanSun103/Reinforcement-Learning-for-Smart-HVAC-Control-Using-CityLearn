# Reinforcement Learning for Smart HVAC Control Using CityLearn

A course project that trains multiple RL agents to control HVAC systems in a
CityLearn building simulation environment. Three algorithms — **SAC**, **PPO**, and **TD3** —
are compared against a **Rule-Based Controller (RBC)** baseline using CityLearn's built-in
KPI framework.

---

## Project Goal

Control a building's cooling storage, electrical storage, and heat pump to:

- **Reduce electricity consumption and cost**
- **Maintain thermal comfort** (keep indoor temperature near setpoint)
- **Lower carbon emissions** (shift load away from high-carbon grid periods)

---

## MDP Design

| Component | Description |
|-----------|-------------|
| **State** | 23 observations including time context, weather (current + 1–2 h forecasts), indoor temperature, storage states, electricity pricing, carbon intensity, solar generation — all min-max normalised to `[0, 1]` |
| **Action** | Continuous vector in `[-1, 1]` per device (cooling storage, electrical storage, heat pump) |
| **Reward** | `-(α × energy + β × comfort + γ × carbon)` — balances energy efficiency, thermal comfort, and carbon footprint |
| **Policy** | SAC / PPO / TD3 via Stable Baselines3; best-checkpoint saved by EvalCallback |

---

## Algorithms

| Algorithm | Type | Key feature |
|-----------|------|-------------|
| **SAC** | Off-policy, stochastic | Entropy regularisation for automatic exploration; highly sample-efficient |
| **PPO** | On-policy, stochastic | Fresh rollout per update (one episode); simpler to tune; stable |
| **TD3** | Off-policy, deterministic | Twin critics to reduce Q-overestimation; delayed policy updates |

---

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── install.bat            # Windows setup script (creates citylearn-rl conda env)
├── install.sh             # macOS / Linux setup script
├── src/
│   ├── __init__.py
│   ├── config.py          # all hyperparameters, paths, and feature flags
│   ├── env_setup.py       # CityLearn env factory (schema loading, wrappers)
│   ├── reward.py          # ComfortEnergyReward (energy + comfort + carbon)
│   ├── baseline_agent.py  # Rule-Based Controller + run_baseline()
│   ├── rl_agent.py        # train_agent() / evaluate_agent() for SAC, PPO, TD3
│   ├── train.py           # CLI: baseline rollout + RL training
│   ├── evaluate.py        # CLI: evaluate all models, generate plots
│   ├── tune.py            # CLI: Optuna hyperparameter search
│   ├── run_all.py         # CLI: integrated pipeline (tune → train → evaluate)
│   ├── utils.py           # plotting and metrics helpers
│   └── quick_test.py      # smoke test — run before training
├── results/               # auto-created; all outputs land here
└── notebooks/             # optional Jupyter exploration
```

---

## Setup

> **Important**: do **not** run `pip install -r requirements.txt` directly.
> Use the install script below which creates a dedicated conda environment.
> See the "Why three dependency conflicts exist" table at the bottom for details.

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
It takes 5–10 minutes (mostly the PyTorch download).

**macOS / Linux:**

```bash
bash install.sh
```

### 4. Select the Python interpreter in Cursor / VS Code

After the install completes, press `Ctrl+Shift+P` → **Python: Select Interpreter**
and choose `citylearn-rl (conda)`.

### 5. Verify the environment

```bash
conda activate citylearn-rl
python src/quick_test.py
```

All 12 checks should pass before running any training.

### 6. (Optional) Install Optuna for hyperparameter tuning

```bash
pip install optuna
```

---

### Manual install (step-by-step, if the script fails)

```bash
conda create -n citylearn-rl python=3.11 -y
conda activate citylearn-rl
conda install pytorch=2.3.1 cpuonly -c pytorch -y
pip install citylearn==2.5.0 --no-deps
pip install "gymnasium==0.28.1" "scikit-learn==1.2.2" simplejson pyyaml platformdirs
pip install "stable-baselines3==2.2.1" --no-deps
pip install cloudpickle
pip install "matplotlib>=3.7.0" "pandas>=2.0.0" "numpy>=1.24.0,<2.0.0"
pip install "setuptools<70"          # keeps pkg_resources available for TensorBoard
```

---

### Why three dependency conflicts exist

| Problem | Root cause | Fix |
|---------|-----------|-----|
| `openstudio` install fails | CityLearn declares it as a dep but no PyPI wheel exists for Python 3.11 / Windows | `pip install citylearn --no-deps` — openstudio is only for EnergyPlus generation |
| `gymnasium` version clash | SB3 ≥ 2.3 auto-upgrades gymnasium to 1.x; CityLearn requires ≤ 0.28.1 | Pin `stable-baselines3==2.2.1` and `gymnasium==0.28.1` |
| `torch` DLL crash in Anaconda | pip-installed PyTorch lacks the MKL/TBB DLLs conda environments expect | Use `conda install pytorch cpuonly` in a dedicated environment |

---

## Running the Project

All commands are run from the **project root** with the `citylearn-rl` environment active.

### Option A — Integrated pipeline (recommended)

One command runs everything:

```bash
# Train SAC + PPO + TD3 then evaluate (default):
python src/run_all.py

# Single algorithm:
python src/run_all.py --algo sac

# Tune hyperparameters first, then train and evaluate:
python src/run_all.py --tune --trials 20

# Enable TensorBoard logging (then launch server separately — see TensorBoard section below):
python src/run_all.py --tensorboard

# Quick demo (1 episode, no eval callback):
python src/run_all.py --algo sac --episodes 1 --no-eval-callback
```

### Option B — Step-by-step

```bash
# 1. (Optional) Tune hyperparameters with Optuna:
python src/tune.py --algo sac --trials 20
#    → results/best_params.json  (copy values into config.py before training)

# 2. Train all algorithms (or just one):
python src/train.py                   # SAC + PPO + TD3
python src/train.py --algo sac        # SAC only
python src/train.py --tensorboard     # with TensorBoard logging

# 3. Evaluate and compare:
python src/evaluate.py
```

### TensorBoard (if enabled)

Open a **second terminal** (keep training running in the first), then:

```bash
conda activate citylearn-rl
python -m tensorboard.main --logdir results/tensorboard --port 6006
```

Then open **http://localhost:6006** in your browser.

> **Note — `pkg_resources` error on Windows**: if you see
> `ModuleNotFoundError: No module named 'pkg_resources'`, run:
> ```bash
> pip install "setuptools<70"
> ```
> Newer setuptools (v80+) removed `pkg_resources`, which TensorBoard still requires.

---

## Configuration

All settings live in `src/config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BUILDING_TO_USE` | `"Building_1"` | Which building to control |
| `MULTI_BUILDING` | `False` | Set `True` to control all 3 buildings simultaneously |
| `TRAIN_EPISODES` | `5` | Training episodes (720 steps each for the 2023 dataset) |
| `USE_PREDICTION_OBS` | `True` | Include 1–2 h weather and price forecasts in the state |
| `COMFORT_BAND` | `1.0` | Comfort temperature tolerance (°C) |
| `REWARD_ENERGY_WEIGHT` | `0.4` | Weight on energy penalty |
| `REWARD_COMFORT_WEIGHT` | `0.4` | Weight on comfort penalty |
| `REWARD_CARBON_WEIGHT` | `0.2` | Weight on carbon penalty (0.0 to disable) |
| `SAC_LEARNING_RATE` | `3e-4` | SAC learning rate |
| `PPO_N_STEPS` | `720` | PPO rollout length (set to episode length for clean updates) |
| `TD3_POLICY_DELAY` | `2` | TD3 actor update frequency |
| `OPTUNA_N_TRIALS` | `20` | Number of Optuna trials for hyperparameter search |

---

## Outputs

After running the pipeline, `results/` contains:

| File | Description |
|------|-------------|
| `baseline_kpis.csv` | KPIs from the Rule-Based Controller |
| `sac_kpis.csv` / `ppo_kpis.csv` / `td3_kpis.csv` | KPIs per RL algorithm |
| `kpi_comparison.png` | Grouped bar chart — all agents side by side |
| `training_rewards.png` | Episode reward curves (SAC, PPO, TD3 on one plot) |
| `temperature_trace_baseline.png` | Indoor temp vs setpoint — RBC |
| `temperature_trace_sac.png` / `_ppo.png` / `_td3.png` | Indoor temp — RL agents |
| `reward_comparison.png` | Per-step reward (7-day rolling mean) — all agents |
| `sac_hvac_model.zip` / `ppo_hvac_model.zip` / `td3_hvac_model.zip` | Final model weights |
| `best_sac/best_model.zip` / `best_ppo/` / `best_td3/` | Best checkpoint (EvalCallback) |
| `tensorboard/` | TensorBoard logs (if `--tensorboard` was used) |
| `best_params.json` | Optuna best hyperparameters (if `--tune` was used) |
| `optuna_study.db` | Full Optuna study database (SQLite) |

---

## Design Choices

- **Three algorithms**: SAC (off-policy, stochastic), PPO (on-policy, stochastic), and TD3
  (off-policy, deterministic) provide a thorough academic comparison across the
  on-policy vs off-policy and stochastic vs deterministic axes.

- **Carbon-aware reward**: adds a third term `γ × carbon_intensity × net_electricity`
  so the agent learns to shift load away from high-emission grid periods — a realistic
  real-world objective beyond pure energy minimisation.

- **Forecast observations**: the state includes 1 h and 2 h ahead forecasts of
  outdoor temperature, solar irradiance, and electricity price. This gives the agent
  look-ahead for storage pre-charging decisions.

- **EvalCallback**: evaluates on a separate environment after every episode and saves
  the best-performing checkpoint. `evaluate.py` automatically loads this best checkpoint
  rather than the final weights.

- **TensorBoard**: optional real-time training dashboards showing reward, entropy,
  critic loss, and actor loss across all algorithms simultaneously.

- **Optuna**: searches over learning rate, batch size, network depth/width, and
  algorithm-specific parameters. Best params saved to `results/best_params.json`.

- **Multi-building mode**: set `MULTI_BUILDING = True` in `config.py` to expand to
  all 3 buildings. The action/observation space scales automatically.

- **Single-building default**: keeps the MDP simple for the initial prototype;
  `central_agent=True` collapses observations/actions into flat vectors for SB3.

- **NormalizedObservationWrapper**: scales all observations to `[0, 1]` and
  cyclically encodes time features (hour, month) as sin/cos pairs.

---

## References

- CityLearn documentation: https://www.citylearn.net
- Stable Baselines3: https://stable-baselines3.readthedocs.io
- Haarnoja et al. (2018), "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
- Fujimoto et al. (2018), "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)
- Akiba et al. (2019), "Optuna: A Next-generation Hyperparameter Optimization Framework"
