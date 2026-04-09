"""
reward.py
---------
Custom reward function for the CityLearn HVAC RL project.

The reward balances three competing objectives:

  1. Energy efficiency  — penalise net electricity consumption.
  2. Thermal comfort    — penalise indoor temperature deviations outside
                          a ±COMFORT_BAND band around the setpoint.
  3. Carbon emissions   — penalise the carbon footprint of electricity drawn
                          from the grid (carbon_intensity × net_electricity).

Formula
-------
  reward = -(alpha * energy_penalty
             + beta  * comfort_penalty
             + gamma * carbon_penalty)

  energy_penalty  = net_electricity_consumption  (kWh, current timestep)
  comfort_penalty = max(0, |T_indoor - T_setpoint| - COMFORT_BAND)
  carbon_penalty  = carbon_intensity (kg CO₂/kWh) × max(0, net_electricity)
  alpha  = config.REWARD_ENERGY_WEIGHT
  beta   = config.REWARD_COMFORT_WEIGHT
  gamma  = config.REWARD_CARBON_WEIGHT   (set to 0.0 to disable)

The carbon term drives the agent to shift load away from high-emission grid
periods — a real-world goal that goes beyond pure energy minimisation.

CityLearn 2.5.0 RewardFunction API
------------------------------------
  __init__(self, env_metadata, **kwargs)
      env_metadata is a static dict injected automatically by CityLearn.
      There is NO self.env — the reward function is fully decoupled.

  calculate(self, observations)
      observations is a list of dicts, one per active building.
      Each dict maps observation name → current value.
      Returns a list of floats (one per building), or a single summed value
      when central_agent=True.
"""

import os
import sys
from typing import List, Mapping, Union

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from citylearn.reward_function import RewardFunction

from src import config


class ComfortEnergyReward(RewardFunction):
    """
    Composite reward: energy efficiency + thermal comfort + carbon emissions.

    Parameters
    ----------
    env_metadata : Mapping
        Static environment metadata injected by CityLearn (must contain
        at least 'central_agent').
    **kwargs
        Forwarded to the base class.
    """

    def __init__(self, env_metadata: Mapping, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.energy_weight: float = config.REWARD_ENERGY_WEIGHT
        self.comfort_weight: float = config.REWARD_COMFORT_WEIGHT
        self.carbon_weight: float  = config.REWARD_CARBON_WEIGHT
        self.comfort_band: float   = config.COMFORT_BAND

    # ------------------------------------------------------------------
    # Core interface (required by CityLearn)
    # ------------------------------------------------------------------

    def calculate(
        self,
        observations: List[Mapping[str, Union[int, float]]],
    ) -> List[float]:
        """
        Compute one reward value per active building.

        Parameters
        ----------
        observations : list of dict
            One dict per building; keys are observation names, values are
            current readings (already normalised by CityLearn if applicable).

        Returns
        -------
        list of float
            Per-building rewards, or ``[sum]`` when central_agent=True.
        """
        rewards: List[float] = []

        for obs in observations:
            energy  = self._energy_penalty(obs)
            comfort = self._comfort_penalty(obs)
            carbon  = self._carbon_penalty(obs)

            reward = -(
                self.energy_weight  * energy
                + self.comfort_weight * comfort
                + self.carbon_weight  * carbon
            )
            rewards.append(reward)

        return [sum(rewards)] if self.central_agent else rewards

    # ------------------------------------------------------------------
    # Penalty components
    # ------------------------------------------------------------------

    def _energy_penalty(self, obs: Mapping) -> float:
        """
        Net electricity drawn from the grid this timestep (kWh).

        Positive → building is consuming from the grid (penalised).
        Negative → building is exporting solar surplus (rewarded).
        """
        val = obs.get("net_electricity_consumption", 0.0)
        return float(val) if val is not None else 0.0

    def _comfort_penalty(self, obs: Mapping) -> float:
        """
        Excess deviation of indoor temperature from the setpoint (°C).

        Zero when |T_indoor − T_setpoint| ≤ COMFORT_BAND; grows linearly
        beyond the band.  Returns 0.0 if either observation is unavailable.
        """
        t_in = obs.get("indoor_dry_bulb_temperature")
        t_sp = obs.get("indoor_dry_bulb_temperature_set_point")
        if t_in is None or t_sp is None:
            return 0.0
        return max(0.0, abs(float(t_in) - float(t_sp)) - self.comfort_band)

    def _carbon_penalty(self, obs: Mapping) -> float:
        """
        Carbon footprint of electricity consumed this timestep (kg CO₂).

        carbon_penalty = carbon_intensity (kg CO₂/kWh) × max(0, net_electricity)

        Only grid consumption is penalised (exports earn no carbon credit here,
        keeping the reward signal simple and unambiguous).
        Returns 0.0 if either value is unavailable or carbon_weight == 0.
        """
        if self.carbon_weight == 0.0:
            return 0.0
        intensity = obs.get("carbon_intensity")
        net_elec  = obs.get("net_electricity_consumption")
        if intensity is None or net_elec is None:
            return 0.0
        # Only penalise grid draws, not exports.
        return float(intensity) * max(0.0, float(net_elec))
