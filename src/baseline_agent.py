"""
baseline_agent.py
-----------------
Simple Rule-Based Controller (RBC) baseline for HVAC control.

Design
------
The RBC uses a time-of-use schedule to decide when to charge or discharge
the building's cooling (and electrical) storage:

  Off-peak hours (22:00 – 05:00) → charge storage fully (+1.0)
      Cheap electricity; fill the thermal/electrical buffer overnight.

  Peak hours (12:00 – 18:00)     → discharge storage fully (−1.0)
      Expensive electricity; draw from stored energy to reduce grid demand.

  All other hours                 → idle (0.0)
      Maintain current storage level; no active storage action.

This is a well-known heuristic used in commercial buildings and serves as a
strong but interpretable baseline for comparison with the RL agent.

The `hour` observation in CityLearn ranges from 1 to 24 (not 0-indexed).
"""

import os
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import config
from src.env_setup import make_baseline_env, get_building_obs_names, get_building_setpoints


# ---------------------------------------------------------------------------
# Rule-Based Controller
# ---------------------------------------------------------------------------

class RuleBasedController:
    """
    Time-of-use Rule-Based Controller for CityLearn.

    The controller reads the `hour` from the current timestep and applies
    a fixed charge/discharge/idle action for every controllable device.

    Parameters
    ----------
    env : CityLearnEnv
        A raw (unwrapped) CityLearnEnv.  The environment is used to determine
        how many action dimensions are needed (one per active device).

    Notes
    -----
    CityLearn's `hour` observation is 1-indexed (1 = midnight, 24 = 23:00).
    We convert to 0-indexed hours internally (subtract 1) for readability.
    """

    def __init__(self, env):
        self.env = env
        self.charge_hours: set = set(config.RBC_CHARGE_HOURS)
        self.discharge_hours: set = set(config.RBC_DISCHARGE_HOURS)

    def predict(self, observations: List[List[float]]) -> List[List[float]]:
        """
        Choose an action for the current timestep.

        Parameters
        ----------
        observations : list of list of float
            Raw observation lists as returned by env.step() or env.reset().
            Outer list has one element per building; each inner list is the
            observation vector for that building.

        Returns
        -------
        list of list of float
            Actions formatted for ``central_agent=True``:
            ``[[a1, a2, ..., a_total]]`` — a single flat list of all devices
            across all buildings, wrapped in an outer list.
        """
        hour = self._get_hour(observations)
        action_value = self._hour_to_action(hour)

        # In central_agent=True mode CityLearn expects ONE flat action list
        # containing all devices across all buildings (not one list per building).
        # Clip each action to the valid bounds for that device — e.g. the
        # cooling_device has bounds [0, 1] and must not receive negative values.
        all_actions: List[float] = []
        for idx, building in enumerate(self.env.buildings):
            act_space = self.env.action_space[idx]
            for j in range(len(building.active_actions)):
                clipped = float(np.clip(action_value, act_space.low[j], act_space.high[j]))
                all_actions.append(clipped)

        return [all_actions]

    def _get_hour(self, observations: List[List[float]]) -> int:
        """
        Extract the current hour from the first building's observation vector.

        CityLearn puts `hour` at a fixed index in the observation (typically
        index 2, after `month` and `day_type`). We look it up by name to be
        robust to schema changes.

        Parameters
        ----------
        observations : list of list of float

        Returns
        -------
        int
            Hour of the day, 0-indexed (0 = midnight, 23 = 23:00).
        """
        try:
            # get_building_obs_names() is robust to LSTMDynamicsBuilding and
            # other building types that don't expose observation_names directly.
            obs_names = get_building_obs_names(self.env.buildings[0])
            if "hour" in obs_names:
                hour_idx = obs_names.index("hour")
                raw_hour = observations[0][hour_idx]
                # CityLearn hour obs is 1-indexed (1–24); convert to 0-indexed.
                return int(round(raw_hour)) % 24
        except (IndexError, ValueError, AttributeError):
            pass

        # Fallback: derive hour from current timestep (assume hourly data).
        return self.env.time_step % 24

    def _hour_to_action(self, hour_0indexed: int) -> float:
        """
        Map a 0-indexed hour to a charge/discharge/idle action value.

        Parameters
        ----------
        hour_0indexed : int
            Hour of day, 0-indexed.

        Returns
        -------
        float
            One of: CHARGE_ACTION (+1.0), DISCHARGE_ACTION (-1.0), or IDLE (0.0).
        """
        # Adjust config hours to 0-indexed for comparison.
        charge_hours_0 = {h % 24 for h in config.RBC_CHARGE_HOURS}
        discharge_hours_0 = {h % 24 for h in config.RBC_DISCHARGE_HOURS}

        if hour_0indexed in charge_hours_0:
            return config.RBC_CHARGE_ACTION
        elif hour_0indexed in discharge_hours_0:
            return config.RBC_DISCHARGE_ACTION
        else:
            return config.RBC_IDLE_ACTION


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------

def run_baseline() -> Dict[str, Any]:
    """
    Run the Rule-Based Controller for one full episode and return KPIs.

    Steps
    -----
    1. Create a fresh single-building CityLearnEnv.
    2. Step through the full episode with the RBC.
    3. Collect cumulative reward and CityLearn's built-in KPIs.
    4. Return a dict with both.

    Returns
    -------
    dict with keys:
      'kpis'            : pd.DataFrame from env.evaluate() (KPI pivot table)
      'episode_reward'  : float, total reward summed across all timesteps
      'rewards_per_step': list of float, per-timestep rewards
      'indoor_temps'    : list of float, recorded indoor temperatures
      'setpoints'       : list of float, recorded temperature setpoints
    """
    print("\n" + "=" * 55)
    print("  Running Rule-Based Controller (Baseline)")
    print("=" * 55)

    env = make_baseline_env()
    agent = RuleBasedController(env)

    observations, _ = env.reset()
    total_reward = 0.0
    rewards_per_step: List[float] = []

    step = 0
    while not env.terminated:
        actions = agent.predict(observations)
        step_result = env.step(actions)
        observations = step_result[0]
        reward = step_result[1]

        r = float(reward) if not isinstance(reward, list) else float(sum(reward))
        total_reward += r
        rewards_per_step.append(r)
        step += 1
        if step % 200 == 0:
            print(f"  [RBC] Step {step:5d} | cumulative reward: {total_reward:.2f}")

    print(f"  [RBC] Episode complete — {step} steps | total reward: {total_reward:.4f}")

    # Extract temperature history after the episode (LSTMDynamicsBuilding
    # accumulates the full run history, which is more reliable than
    # step-by-step collection with attribute checks).
    b = env.buildings[0]
    try:
        indoor_temps = [float(x) for x in b.indoor_dry_bulb_temperature[:step]]
    except (AttributeError, TypeError):
        indoor_temps = []
    setpoints = get_building_setpoints(b, step)

    # CityLearn's evaluate() returns a DataFrame of normalised KPIs.
    kpis_raw = env.evaluate()

    # Pivot to a readable table: rows = metric names, columns = building names.
    try:
        kpis = kpis_raw.pivot(
            index="cost_function", columns="name", values="value"
        ).round(4)
        kpis = kpis.dropna(how="all")
    except Exception as exc:
        print(f"  [RBC] Warning: could not pivot KPI table ({exc}). Using raw output.")
        kpis = kpis_raw

    print("\n  Baseline KPIs (relative to no-control reference = 1.0):")
    print(kpis.to_string())
    print()

    return {
        "kpis": kpis,
        "episode_reward": total_reward,
        "rewards_per_step": rewards_per_step,
        "indoor_temps": indoor_temps,
        "setpoints": setpoints,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_baseline()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    kpi_path = config.BASELINE_KPI_PATH
    results["kpis"].to_csv(kpi_path)
    print(f"Baseline KPIs saved to: {kpi_path}")
