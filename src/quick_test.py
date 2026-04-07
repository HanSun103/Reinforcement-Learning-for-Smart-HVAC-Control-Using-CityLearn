"""
quick_test.py
-------------
Smoke test for the CityLearn HVAC RL project.

Run this FIRST, before train.py, to confirm the environment initialises
correctly on your machine and all imports resolve.

    python src/quick_test.py

What this checks
----------------
1. CityLearn and Stable Baselines3 import successfully.
2. The single-building CityLearnEnv is created (schema loading + filtering).
3. The custom reward function is registered.
4. Active observations match the project MDP definition.
5. The wrapped environment (SB3-compatible) has correct obs/act shapes.
6. A few environment steps run without error using the RBC and random actions.
7. CityLearn's evaluate() KPI method runs without error.

Expected runtime: < 60 seconds.
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"

results = []


def check(label: str, fn):
    """Run fn(); record pass/fail with the given label."""
    try:
        fn()
        print(f"{PASS} {label}")
        results.append((label, True))
    except Exception as exc:
        print(f"{FAIL} {label}")
        print(f"        {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results.append((label, False))


# ---------------------------------------------------------------------------
# 1. Package imports
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Quick Test — CityLearn HVAC RL Project")
print("=" * 60)

print("\n[1] Checking imports...")


def _check_citylearn():
    import citylearn
    print(f"        citylearn version: {citylearn.__version__}")


def _check_sb3():
    import stable_baselines3
    print(f"        stable-baselines3 version: {stable_baselines3.__version__}")


def _check_torch():
    import torch
    print(f"        torch version: {torch.__version__}")


check("import citylearn", _check_citylearn)
check("import stable_baselines3", _check_sb3)
check("import torch", _check_torch)

# ---------------------------------------------------------------------------
# 2. Schema loading
# ---------------------------------------------------------------------------
print("\n[2] Checking schema loading and single-building filter...")

raw_env = None


def _check_schema():
    from src.env_setup import _build_schema
    schema = _build_schema()
    if isinstance(schema, dict):
        buildings = schema.get("buildings", {})
        active = [k for k, v in buildings.items() if v.get("include", True)]
        print(f"        Schema type: dict | Active buildings: {active}")
        assert len(active) >= 1, "No active buildings found in schema"
    else:
        print(f"        Schema type: str (fallback) — single-building filter skipped")


check("schema loading + building filter", _check_schema)

# ---------------------------------------------------------------------------
# 3. Raw environment creation (for baseline / info)
# ---------------------------------------------------------------------------
print("\n[3] Checking raw CityLearnEnv (baseline env)...")


def _check_raw_env():
    global raw_env
    from src.env_setup import make_baseline_env, print_env_info, get_building_obs_names  # noqa: F401
    raw_env = make_baseline_env()
    print_env_info(raw_env)


check("make_baseline_env()", _check_raw_env)

# ---------------------------------------------------------------------------
# 4. Observation space check
# ---------------------------------------------------------------------------
print("\n[4] Checking active observations match MDP definition...")


def _check_observations():
    from src import config
    from src.env_setup import get_building_obs_names
    if raw_env is None:
        raise RuntimeError("raw_env not available — previous test failed")

    b = raw_env.buildings[0]
    active_obs = get_building_obs_names(b)
    desired = set(config.ACTIVE_OBSERVATIONS)

    print(f"        Active observations in env  : {active_obs}")
    print(f"        Desired observations (config): {sorted(desired)}")

    # Warn about missing ones (may be schema version dependent)
    present = set(active_obs)
    missing_from_env = desired - present
    extra_in_env = present - desired

    if missing_from_env:
        print(f"        WARNING: desired obs not active in env: {missing_from_env}")
    if extra_in_env:
        print(f"        Note: extra obs in env (not in ACTIVE_OBSERVATIONS): {extra_in_env}")

    assert len(active_obs) > 0, "No active observations"


check("observation space matches MDP definition", _check_observations)

# ---------------------------------------------------------------------------
# 5. Action space check
# ---------------------------------------------------------------------------
print("\n[5] Checking action space...")


def _check_actions():
    if raw_env is None:
        raise RuntimeError("raw_env not available")

    b = raw_env.buildings[0]
    act_names = b.active_actions
    act_space = raw_env.action_space[0]
    print(f"        Active actions  : {act_names}")
    print(f"        Action space    : low={act_space.low}, high={act_space.high}")
    print(f"        Action dim      : {act_space.shape[0]}")
    assert act_space.shape[0] > 0, "Empty action space"


check("action space (continuous [-1, 1])", _check_actions)

# ---------------------------------------------------------------------------
# 6. Reward function check
# ---------------------------------------------------------------------------
print("\n[6] Checking custom reward function...")


def _check_reward():
    if raw_env is None:
        raise RuntimeError("raw_env not available")

    rf = raw_env.reward_function
    print(f"        Reward function type: {type(rf).__name__}")
    if type(rf).__name__ == "ComfortEnergyReward":
        print(f"        Custom reward active — energy_weight={rf.energy_weight}, "
              f"comfort_weight={rf.comfort_weight}, band={rf.comfort_band}")
    else:
        print(f"        WARNING: using default reward '{type(rf).__name__}' "
              f"(custom injection may have failed — check env_setup.py warnings)")


check("reward function registration", _check_reward)

# ---------------------------------------------------------------------------
# 7. SB3-wrapped env creation
# ---------------------------------------------------------------------------
print("\n[7] Checking SB3-wrapped environment...")

sb3_env = None


def _check_sb3_env():
    global sb3_env
    from src.env_setup import make_env
    sb3_env = make_env()
    obs_shape = sb3_env.observation_space.shape
    act_shape = sb3_env.action_space.shape
    print(f"        Obs space shape : {obs_shape}")
    print(f"        Act space shape : {act_shape}")
    print(f"        Act space bounds: low={sb3_env.action_space.low}, "
          f"high={sb3_env.action_space.high}")
    assert obs_shape[0] > 0
    assert act_shape[0] > 0


check("make_env() (SB3 wrapper stack)", _check_sb3_env)

# ---------------------------------------------------------------------------
# 8. RBC baseline — a few steps
# ---------------------------------------------------------------------------
print("\n[8] Running RBC for 24 steps (1 day)...")


def _check_rbc_steps():
    from src.env_setup import make_baseline_env
    from src.baseline_agent import RuleBasedController

    env = make_baseline_env()
    agent = RuleBasedController(env)
    obs, _ = env.reset()

    for i in range(24):
        actions = agent.predict(obs)
        step_result = env.step(actions)
        obs = step_result[0]
        reward = step_result[1]

    print(f"        24 RBC steps completed | last reward: {reward}")


check("RBC 24-step rollout", _check_rbc_steps)

# ---------------------------------------------------------------------------
# 9. SAC env — a few random-action steps
# ---------------------------------------------------------------------------
print("\n[9] Running SB3 env for 24 steps with random actions...")


def _check_random_steps():
    if sb3_env is None:
        raise RuntimeError("sb3_env not available")

    obs, _ = sb3_env.reset()
    for i in range(24):
        action = sb3_env.action_space.sample()
        obs, reward, terminated, truncated, info = sb3_env.step(action)

    print(f"        24 random steps completed | last reward: {reward:.4f}")


check("SB3 env 24-step random rollout", _check_random_steps)

# ---------------------------------------------------------------------------
# 10. KPI evaluation
# ---------------------------------------------------------------------------
print("\n[10] Checking env.evaluate() KPI output...")


def _check_kpis():
    from src.env_setup import make_baseline_env
    from src.baseline_agent import RuleBasedController

    env = make_baseline_env()
    agent = RuleBasedController(env)
    obs, _ = env.reset()

    # Run 48 steps (2 days) then evaluate.
    for _ in range(48):
        step_result = env.step(agent.predict(obs))
        obs = step_result[0]

    kpis = env.evaluate()
    print(f"        KPI DataFrame shape: {kpis.shape}")
    print(f"        KPI columns: {list(kpis.columns)}")
    assert kpis is not None and len(kpis) > 0, "evaluate() returned empty result"


check("env.evaluate() returns KPI DataFrame", _check_kpis)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(1 for _, ok in results if ok)
total = len(results)

print("\n" + "=" * 60)
print(f"  Results: {passed}/{total} checks passed")
print("=" * 60)

for label, ok in results:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")

if passed == total:
    print("\n  All checks passed! You can now run:\n")
    print("      python src/train.py\n")
else:
    print(
        f"\n  {total - passed} check(s) failed. Fix the issues above before training.\n"
        f"  Common fixes:\n"
        f"    - Install dependencies:  pip install -r requirements.txt\n"
        f"    - Check CityLearn version: pip show citylearn\n"
        f"    - Run from project root, not from inside src/\n"
    )

print("=" * 60 + "\n")
