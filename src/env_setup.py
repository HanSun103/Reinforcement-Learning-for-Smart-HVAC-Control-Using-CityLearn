"""
env_setup.py
------------
Factory functions for building CityLearn environments.

Two environments are created depending on context:
  - make_env()          : wrapped for Stable Baselines3 (training / RL eval)
  - make_baseline_env() : raw CityLearnEnv (for manual RBC rollout)

The schema is loaded from CityLearn's built-in dataset, then modified
programmatically to:
  1. Keep only the building specified in config.BUILDING_TO_USE.
  2. Enable central_agent mode (required for SB3).
  3. Inject our custom ComfortEnergyReward class.
"""

import os
import sys
import copy

# Ensure the project root is on sys.path so 'src.*' imports work when
# running scripts directly from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

from src import config


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_building_obs_names(building) -> list:
    """
    Return the list of active observation names for *building*.

    CityLearn 2.5.0 uses ``LSTMDynamicsBuilding`` which does not expose
    ``observation_names`` directly.  This helper tries multiple attribute
    names so the code is robust across building types and CityLearn versions.

    Parameters
    ----------
    building : Building-like
        Any CityLearn building object.

    Returns
    -------
    list[str]
    """
    # Preferred: explicit active observations list
    for attr in ("observation_names", "active_observations"):
        val = getattr(building, attr, None)
        if val is not None:
            return list(val)

    # Fallback: observation_metadata dict (keys are obs names, value has 'active')
    meta = getattr(building, "observation_metadata", None)
    if meta and hasattr(meta, "items"):
        active = [k for k, v in meta.items()
                  if not isinstance(v, dict) or v.get("active", True)]
        if active:
            return active

    # Last resort: return a placeholder so callers don't crash
    return ["<observation_names_unavailable>"]


def get_building_setpoints(building, n_steps: int) -> list:
    """
    Return up to *n_steps* indoor temperature setpoint values for *building*.

    ``LSTMDynamicsBuilding`` in CityLearn 2.5.0 does not accumulate
    ``indoor_dry_bulb_temperature_set_point`` as a per-step history list the
    way it does for ``indoor_dry_bulb_temperature``.  The schedule lives in
    ``building.energy_simulation`` instead.

    Tries several access paths in order and returns an empty list if none work.

    Parameters
    ----------
    building : Building-like
    n_steps : int
        How many values to return (sliced from the beginning of the schedule).

    Returns
    -------
    list of float
    """
    # Path 1: history list on the building object (works for some building types)
    try:
        sp = building.indoor_dry_bulb_temperature_set_point
        if sp is not None and hasattr(sp, "__len__") and len(sp) >= n_steps:
            return [float(x) for x in sp[:n_steps]]
    except Exception:
        pass

    # Path 2: full schedule in energy_simulation (standard for LSTMDynamicsBuilding)
    for attr in ("indoor_dry_bulb_temperature_set_point",
                 "indoor_dry_bulb_temperature_setpoint"):
        try:
            esim = building.energy_simulation
            sp = getattr(esim, attr, None)
            if sp is not None and hasattr(sp, "__len__") and len(sp) > 0:
                return [float(x) for x in sp[:n_steps]]
        except Exception:
            pass

    return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_schema() -> dict:
    """
    Load the built-in CityLearn schema as a Python dict.

    CityLearn 2.5.0 downloads datasets from GitHub to a local cache
    (platformdirs user_cache_dir) on first use.  We try four approaches:

      1. DataSet.get_schema(name)          — may exist in some CityLearn builds
      2. platformdirs cache path           — reads the cached schema.json that
                                             CityLearn already downloaded
      3. Trigger download then read cache — calls DataSet.get_dataset() to
                                             download on first run, then reads
      4. Installed-package directory scan — last JSON-file fallback

    If all four fail, returns the schema name as a string so CityLearnEnv can
    still initialise (with all 3 buildings and the default reward).

    Returns
    -------
    dict or str
        Schema dict on success, schema name string as last resort.
    """
    import json
    import citylearn as _cl_module

    # --- Attempt 1: DataSet.get_schema() ---
    try:
        from citylearn.data import DataSet
        schema = DataSet.get_schema(config.SCHEMA_NAME)
        if isinstance(schema, dict):
            print("[env_setup] Loaded schema via DataSet.get_schema()")
            return schema
    except Exception:
        pass

    # --- Attempt 2: platformdirs cache (CityLearn downloads here on first use) ---
    # The cache path printed in CityLearn's INFO logs is:
    #   {user_cache_dir('citylearn','intelligent-environments-lab')}
    #     /v{version}/datasets/{name}/schema.json
    try:
        from platformdirs import user_cache_dir
        cl_version = _cl_module.__version__
        cache_base = user_cache_dir(
            appname="citylearn", appauthor="intelligent-environments-lab"
        )
        schema_path = os.path.join(
            cache_base, f"v{cl_version}", "datasets", config.SCHEMA_NAME, "schema.json"
        )

        if os.path.isfile(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            # root_directory must point to the folder that holds the data CSV files
            schema["root_directory"] = os.path.dirname(schema_path)
            print(f"[env_setup] Loaded schema from cache: {schema_path}")
            return schema

        # Cache doesn't exist yet — trigger the download then retry.
        print("[env_setup] Schema cache not found; downloading via DataSet.get_dataset()…")
        from citylearn.data import DataSet
        DataSet.get_dataset(config.SCHEMA_NAME)           # downloads to cache
        if os.path.isfile(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            schema["root_directory"] = os.path.dirname(schema_path)
            print(f"[env_setup] Loaded schema from cache after download: {schema_path}")
            return schema

    except Exception as exc:
        print(f"[env_setup] Cache-path loading failed: {exc}")

    # --- Attempt 3: JSON file inside the installed package directory ---
    try:
        cl_dir = os.path.dirname(_cl_module.__file__)
        candidates = [
            os.path.join(cl_dir, "data", config.SCHEMA_NAME, "schema.json"),
            os.path.join(cl_dir, "data", "schemas", config.SCHEMA_NAME, "schema.json"),
        ]
        data_dir = os.path.join(cl_dir, "data")
        if os.path.isdir(data_dir):
            for entry in os.scandir(data_dir):
                if entry.is_dir() and config.SCHEMA_NAME in entry.name:
                    candidates.append(os.path.join(entry.path, "schema.json"))
        for path in candidates:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                schema["root_directory"] = os.path.dirname(path)
                print(f"[env_setup] Loaded schema from package: {path}")
                return schema
    except Exception as exc:
        print(f"[env_setup] Package-directory search failed: {exc}")

    # --- Last resort: string name (CityLearnEnv handles download internally) ---
    print(
        f"\n[env_setup] WARNING: Could not load schema as a dict.\n"
        f"  Falling back to string name '{config.SCHEMA_NAME}'.\n"
        f"  Single-building filtering and custom reward will NOT be applied.\n"
        f"  The environment will use all 3 buildings and the default reward.\n"
    )
    return config.SCHEMA_NAME


def _filter_to_single_building(schema: dict, building_name: str) -> dict:
    """
    Disable all buildings in the schema except *building_name*.

    Parameters
    ----------
    schema : dict
        Schema dict as returned by _load_schema().
    building_name : str
        The key of the building to keep active (e.g. "Building_1").

    Returns
    -------
    dict
        Modified schema dict with only the target building included.
    """
    schema = copy.deepcopy(schema)

    if "buildings" not in schema:
        # Schema is a string fallback — cannot filter; return as-is.
        print(f"[env_setup] Schema has no 'buildings' key — skipping filter.")
        return schema

    available = list(schema["buildings"].keys())
    if building_name not in available:
        raise ValueError(
            f"Building '{building_name}' not found in schema. "
            f"Available buildings: {available}"
        )

    kept, excluded = 0, 0
    for name in available:
        if name == building_name:
            schema["buildings"][name]["include"] = True
            kept += 1
        else:
            schema["buildings"][name]["include"] = False
            excluded += 1

    print(f"[env_setup] Kept '{building_name}'; excluded {excluded} other building(s).")
    return schema


def _inject_reward_function(schema: dict) -> dict:
    """
    Point the schema's reward_function at our custom ComfortEnergyReward class.

    CityLearn resolves the type string as a fully-qualified Python class path.
    The class must be importable at the time the environment is constructed.

    Parameters
    ----------
    schema : dict
        Schema dict (already filtered to one building).

    Returns
    -------
    dict
        Schema dict with reward_function updated.
    """
    if not isinstance(schema, dict):
        # String-fallback path — cannot inject reward.
        return schema

    schema = copy.deepcopy(schema)
    schema["reward_function"] = {
        "type": "src.reward.ComfortEnergyReward",
        "attributes": {},
    }
    print("[env_setup] Custom reward function 'src.reward.ComfortEnergyReward' injected.")
    return schema


def _configure_observations(schema: dict) -> dict:
    """
    Activate only the observations listed in config.ACTIVE_OBSERVATIONS.

    CityLearn's schema has observations in two places:

    1. Top-level ``schema["observations"]`` — shared/weather observations
       (hour, month, outdoor temperature, pricing, solar, carbon intensity…).

    2. Per-building ``schema["buildings"][name]["observations"]`` — building-
       specific observations (indoor temperature, setpoint, storage SOC…).

    Both levels are configured so that building-level obs like
    ``indoor_dry_bulb_temperature_set_point`` are properly activated.

    Parameters
    ----------
    schema : dict

    Returns
    -------
    dict
        Schema with observation activation flags configured at both levels.
    """
    if "observations" not in schema:
        print("[env_setup] Schema has no 'observations' key — skipping obs config.")
        return schema

    schema = copy.deepcopy(schema)
    desired = set(config.ACTIVE_OBSERVATIONS)
    activated, deactivated = [], []

    # --- Level 1: top-level (shared / weather) observations ---
    for obs_name, obs_cfg in schema["observations"].items():
        if obs_name in desired:
            obs_cfg["active"] = True
            activated.append(obs_name)
        else:
            obs_cfg["active"] = False
            deactivated.append(obs_name)

    top_level_names = set(schema["observations"].keys())

    # --- Level 2: per-building observations ---
    building_level_activated = []
    for bld_name, bld_cfg in schema.get("buildings", {}).items():
        if not bld_cfg.get("include", True):
            continue
        bld_obs = bld_cfg.get("observations", {})
        for obs_name, obs_cfg in bld_obs.items():
            if obs_name in desired and obs_name not in top_level_names:
                obs_cfg["active"] = True
                if obs_name not in building_level_activated:
                    building_level_activated.append(obs_name)
            elif obs_name not in desired:
                obs_cfg["active"] = False

    if building_level_activated:
        print(f"[env_setup] Building-level obs activated: {building_level_activated}")
        activated.extend(building_level_activated)

    # Report observations requested but not found anywhere in the schema.
    all_schema_names = top_level_names | set(building_level_activated)
    still_missing = desired - all_schema_names
    if still_missing:
        print(f"[env_setup] Note: {still_missing} not found in schema "
              f"(may not be available for this building/dataset — skipping).")

    print(
        f"[env_setup] Observations: activated {len(activated)}, "
        f"deactivated {len(deactivated)}."
    )
    print(f"  Active state variables: {sorted(activated)}")
    return schema


def _build_schema() -> dict:
    """
    Compose the final single-building schema with custom reward and state space.

    Chain: _load_schema → _filter_to_single_building → _configure_observations
                        → _inject_reward_function → set central_agent

    Returns
    -------
    dict or str
        Ready-to-use schema for CityLearnEnv (dict preferred, str as fallback).
    """
    schema = _load_schema()

    # Only manipulate if we got an actual dict back.
    if isinstance(schema, dict):
        if config.MULTI_BUILDING:
            # All buildings remain active; the central agent controls all of them.
            print(f"[env_setup] Multi-building mode — all buildings included.")
        else:
            schema = _filter_to_single_building(schema, config.BUILDING_TO_USE)
        schema["central_agent"] = True
        schema = _configure_observations(schema)
        schema = _inject_reward_function(schema)

    return schema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_env() -> StableBaselines3Wrapper:
    """
    Create a CityLearn environment wrapped for Stable Baselines3.

    Wrapper stack (inside → out):
      CityLearnEnv
        └─ NormalizedObservationWrapper   (obs scaled to [0,1], cyclical obs cosine-encoded)
             └─ StableBaselines3Wrapper   (flattens multi-building obs/act to 1-D arrays)

    This is the environment passed to SB3's SAC for both training and evaluation.

    Returns
    -------
    StableBaselines3Wrapper
        Fully wrapped Gymnasium environment compatible with SB3.
    """
    schema = _build_schema()

    # central_agent kwarg is ignored if already set inside the schema dict,
    # but passing it explicitly ensures correctness on string-fallback paths.
    env = CityLearnEnv(schema, central_agent=True)

    print(
        f"[env_setup] CityLearnEnv created | "
        f"buildings: {[b.name for b in env.buildings]} | "
        f"obs_dim: {env.observation_space[0].shape[0]} | "
        f"act_dim: {env.action_space[0].shape[0]}"
    )

    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    print("[env_setup] NormalizedObservationWrapper + StableBaselines3Wrapper applied.")
    return env


def make_baseline_env() -> CityLearnEnv:
    """
    Create a raw (unwrapped) CityLearnEnv for the Rule-Based Controller.

    The RBC steps through the environment manually, so it does not need the
    SB3 wrapper stack. Observations are returned as raw (un-normalised) lists.

    Returns
    -------
    CityLearnEnv
        Bare CityLearnEnv with the single-building schema and custom reward.
    """
    schema = _build_schema()
    env = CityLearnEnv(schema, central_agent=True)

    print(
        f"[env_setup] Baseline CityLearnEnv created | "
        f"buildings: {[b.name for b in env.buildings]}"
    )
    return env


def print_env_info(env: CityLearnEnv) -> None:
    """
    Print a human-readable summary of the environment's observation and action spaces.

    Useful for debugging and for understanding the MDP state/action dimensions.

    Parameters
    ----------
    env : CityLearnEnv
        A raw (unwrapped) CityLearnEnv instance.
    """
    print("\n" + "=" * 60)
    print("  CityLearn Environment Summary")
    print("=" * 60)
    print(f"  Buildings  : {[b.name for b in env.buildings]}")
    print(f"  Time steps : {env.time_steps}")
    print()

    for b in env.buildings:
        obs_names = get_building_obs_names(b)
        act_names = b.active_actions
        print(f"  Building '{b.name}':")
        print(f"    Active observations ({len(obs_names)}): {obs_names}")
        print(f"    Active actions      ({len(act_names)}): {act_names}")
    print("=" * 60 + "\n")
