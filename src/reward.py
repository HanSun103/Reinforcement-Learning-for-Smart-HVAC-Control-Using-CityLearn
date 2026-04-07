"""
reward.py
---------
Custom reward function for the CityLearn HVAC RL project.

The reward balances two competing objectives:
  1. Energy efficiency  — penalise net electricity consumption.
  2. Thermal comfort    — penalise indoor temperature deviations outside
                          a ±COMFORT_BAND band around the setpoint.

Formula
-------
  reward = -(alpha * energy_penalty + beta * comfort_penalty)

  where:
    energy_penalty  = net_electricity_consumption  (kWh, current timestep)
    comfort_penalty = max(0, |T_indoor - T_setpoint| - COMFORT_BAND)
    alpha           = config.REWARD_ENERGY_WEIGHT
    beta            = config.REWARD_COMFORT_WEIGHT

CityLearn 2.5.0 API for RewardFunction
---------------------------------------
  - __init__(self, env_metadata, **kwargs)
      ``env_metadata`` is a static dict of environment-level info injected
      automatically by CityLearn (includes 'central_agent', etc.).
      There is NO ``self.env``; the reward function is environment-agnostic.

  - calculate(self, observations)
      ``observations`` is a list of dicts — one per *active* building.
      Each dict maps observation name → current value (int or float).
      Returns a list of floats (one reward per building), or a single-element
      list when ``central_agent=True``.

Usage
-----
The class is referenced by its fully-qualified module path in the schema dict:
  schema["reward_function"]["type"] = "src.reward.ComfortEnergyReward"

CityLearn instantiates it automatically when the environment is created.
"""

import os
import sys
from typing import List, Mapping, Union

# Ensure project root is on the path when this module is imported directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from citylearn.reward_function import RewardFunction

from src import config


class ComfortEnergyReward(RewardFunction):
    """
    Composite reward that penalises energy use and thermal discomfort.

    CityLearn calls ``calculate(observations)`` once per timestep after
    ``env.step()``.  The ``observations`` argument is a list of dicts — one
    per active building — with observation names as keys.

    Parameters
    ----------
    env_metadata : Mapping[str, Any]
        Static environment metadata injected automatically by CityLearn.
        Must contain at least ``'central_agent'`` (bool).
    **kwargs
        Forwarded to the base class (e.g. ``exponent``,
        ``charging_constraint_penalty_coefficient``).
    """

    def __init__(self, env_metadata: Mapping, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.energy_weight: float = config.REWARD_ENERGY_WEIGHT
        self.comfort_weight: float = config.REWARD_COMFORT_WEIGHT
        self.comfort_band: float = config.COMFORT_BAND

    # ------------------------------------------------------------------
    # Core method (required by CityLearn RewardFunction interface)
    # ------------------------------------------------------------------

    def calculate(
        self,
        observations: List[Mapping[str, Union[int, float]]],
    ) -> List[float]:
        """
        Compute one reward value per active building for the current timestep.

        Parameters
        ----------
        observations : list of dict
            Each dict contains the active observations for one building,
            e.g. ``{'net_electricity_consumption': 2.3,
                    'indoor_dry_bulb_temperature': 23.1, ...}``.

        Returns
        -------
        list of float
            One reward per building, or a single summed value when
            ``central_agent=True``.
        """
        rewards: List[float] = []

        for obs in observations:
            energy_penalty = self._energy_penalty(obs)
            comfort_penalty = self._comfort_penalty(obs)

            reward = -(
                self.energy_weight * energy_penalty
                + self.comfort_weight * comfort_penalty
            )
            rewards.append(reward)

        # In central_agent mode CityLearn expects a single scalar in a list.
        if self.central_agent:
            return [sum(rewards)]
        return rewards

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _energy_penalty(
        self, obs: Mapping[str, Union[int, float]]
    ) -> float:
        """
        Return the net electricity consumption for this timestep.

        Positive = consumed from grid (penalised).
        Negative = exported to grid (rewarded via negative penalty).
        """
        val = obs.get("net_electricity_consumption", 0.0)
        return float(val) if val is not None else 0.0

    def _comfort_penalty(
        self, obs: Mapping[str, Union[int, float]]
    ) -> float:
        """
        Comfort penalty: zero inside the band, linear outside.

        Uses ``indoor_dry_bulb_temperature`` and
        ``indoor_dry_bulb_temperature_set_point`` from the observation dict.
        Returns 0.0 if either value is missing (e.g. not in active obs).
        """
        t_in = obs.get("indoor_dry_bulb_temperature")
        t_sp = obs.get("indoor_dry_bulb_temperature_set_point")

        if t_in is None or t_sp is None:
            return 0.0

        deviation = abs(float(t_in) - float(t_sp))
        return max(0.0, deviation - self.comfort_band)
