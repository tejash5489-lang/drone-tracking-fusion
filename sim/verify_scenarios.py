"""
Phase 6 — quick structural sanity check across all 7 scenarios.

Runs each scenario once and reports whether tracking behaved reasonably
(no crashes, at least some track continuity) — a fast smoke test to run
after touching sim/scenarios.py or the sensor/tracker code, distinct from
the slower per-scenario omega-label-diversity check (which additionally
runs KL/CMA-ES optimization per step and needs the real
covariance_intersection — see session notes for the ground_degraded
scenario's ~11x slower runtime under 6x clutter, worth knowing before
running that check across many seeds).

Run: venv\\Scripts\\python.exe -m sim.verify_scenarios
"""
from sim.scenarios import SCENARIOS
from sim.two_radar_simulation import run_scenario_simulation


def verify_scenario(scenario, seed=0):
    truth_paths, steps, jpda_tracker, gmlcc_tracker = run_scenario_simulation(scenario, seed=seed)
    n_targets = len(truth_paths)
    max_airborne_active = max(len(s[1]) for s in steps)
    max_ground_active = max(len(s[2]) for s in steps)
    steps_with_both = sum(1 for _, a, g in steps if a and g)
    return {
        "targets": n_targets,
        "steps": len(steps),
        "steps_with_both_sensors": steps_with_both,
        "max_concurrent_airborne_tracks": max_airborne_active,
        "max_concurrent_ground_tracks": max_ground_active,
    }


def main():
    print(f"{'scenario':25s} {'targets':>7s} {'steps':>6s} {'both':>5s} {'max_air':>8s} {'max_gnd':>8s}")
    for scenario in SCENARIOS:
        try:
            result = verify_scenario(scenario)
            print(f"{scenario.name:25s} {result['targets']:7d} {result['steps']:6d} "
                  f"{result['steps_with_both_sensors']:5d} "
                  f"{result['max_concurrent_airborne_tracks']:8d} "
                  f"{result['max_concurrent_ground_tracks']:8d}")
        except Exception as e:
            print(f"{scenario.name:25s} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
