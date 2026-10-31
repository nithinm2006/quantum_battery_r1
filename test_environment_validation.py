
import numpy as np
from gymnasium.utils.env_checker import check_env

from env.battery_env import BatteryEnv


# --------------------------------------------------
# TEST 1: Reproducibility
# --------------------------------------------------

def test_reproducibility():

    print("\nTEST 1: Reproducibility")

    env1 = BatteryEnv()
    env2 = BatteryEnv()

    state1, _ = env1.reset(seed=42)
    state2, _ = env2.reset(seed=42)

    assert np.allclose(state1, state2)
    assert np.allclose(env1.prices, env2.prices)
    assert np.allclose(env1.demands, env2.demands)

    print("PASS: Same seed produces identical daily data.")

    env1.close()
    env2.close()


# --------------------------------------------------
# TEST 2: Maximum Battery Capacity
# --------------------------------------------------

def test_maximum_capacity():

    print("\nTEST 2: Maximum Battery Capacity")

    env = BatteryEnv()

    env.reset(seed=42)

    env.soc = 99.0

    env.step(0)  # Charge

    assert env.soc <= env.capacity

    print("PASS: Battery never exceeds maximum capacity.")
    print("Final SOC:", round(env.soc, 2), "kWh")

    env.close()


# --------------------------------------------------
# TEST 3: Minimum Battery Capacity
# --------------------------------------------------

def test_minimum_capacity():

    print("\nTEST 3: Minimum Battery Capacity")

    env = BatteryEnv()

    env.reset(seed=42)

    env.soc = 1.0

    env.step(1)  # Discharge

    assert env.soc >= 0

    print("PASS: Battery never goes below zero.")
    print("Final SOC:", round(env.soc, 2), "kWh")

    env.close()


# --------------------------------------------------
# TEST 4: Invalid Action
# --------------------------------------------------

def test_invalid_action():

    print("\nTEST 4: Invalid Action")

    env = BatteryEnv()

    env.reset(seed=42)

    try:

        env.step(5)

        raise AssertionError(
            "Invalid action was accepted."
        )

    except ValueError:

        print("PASS: Invalid actions are rejected.")

    env.close()


# --------------------------------------------------
# TEST 5: Episode Termination
# --------------------------------------------------

def test_episode_termination():

    print("\nTEST 5: Episode Termination")

    env = BatteryEnv()

    env.reset(seed=42)

    terminated = False

    for _ in range(24):

        state, reward, terminated, truncated, info = env.step(2)

    assert terminated is True
    assert truncated is False
    assert env.current_step == 24

    print("PASS: Episode terminates after 24 hours.")

    try:

        env.step(2)

        raise AssertionError(
            "Environment allowed an action after termination."
        )

    except RuntimeError:

        print("PASS: Further actions require environment reset.")

    env.close()


# --------------------------------------------------
# TEST 6: Gymnasium Compatibility
# --------------------------------------------------

def test_gymnasium_compatibility():

    print("\nTEST 6: Gymnasium Compatibility")

    env = BatteryEnv()

    check_env(env, skip_render_check=True)

    print("PASS: Gymnasium environment checker passed.")

    env.close()


# --------------------------------------------------
# RUN ALL TESTS
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("BATTERY ENVIRONMENT VALIDATION")
    print("=" * 55)

    tests = [
        test_reproducibility,
        test_maximum_capacity,
        test_minimum_capacity,
        test_invalid_action,
        test_episode_termination,
        test_gymnasium_compatibility
    ]

    passed = 0
    failed = 0

    for test in tests:

        try:

            test()
            passed += 1

        except Exception as error:

            failed += 1

            print("\nFAILED:", test.__name__)
            print("Error:", error)

    print("\n" + "=" * 55)
    print("VALIDATION SUMMARY")
    print("=" * 55)

    print("Total Tests:", len(tests))
    print("Passed:", passed)
    print("Failed:", failed)

    if failed == 0:

        print("\nALL TESTS PASSED!")

    else:

        print("\nSOME TESTS FAILED.")
