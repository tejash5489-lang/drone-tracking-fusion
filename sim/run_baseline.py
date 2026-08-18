"""
Phase 1 deliverable — baseline pipeline (blueprint section 3, Phase 1):

    "Deliverable: baseline pipeline that runs N simulations, fuses tracks
    with fixed omega, and reports OSPA/SIAP — this is the number every
    later approach must beat."

Wires together everything built so far:
    sim.scenario    -> 3D NCV ground truth
    sim.sensors     -> airborne (moving) + ground (fixed) radar, with clutter
    trackers.jpda_tracker.JPDATracker    -> airborne radar's per-sensor track
    trackers.gmlcc_tracker.GMLCCTracker  -> ground radar's per-sensor track
    fusion.covariance_intersection       -> fixes omega=0.5, fuses the two
    eval.metrics                          -> OSPA/SIAP for each of the three
                                              (airborne-only, ground-only, fused)

Run from the project root (needs to be a module so sim/trackers/fusion/eval
resolve as packages): venv\\Scripts\\python.exe -m sim.run_baseline
"""
from sim.two_radar_simulation import run_two_radar_simulation, run_single_target_scenario
from fusion.covariance_intersection import covariance_intersection
from eval.metrics import build_metric_manager, compute_metrics
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

FIXED_OMEGA = 0.5  # the baseline every later approach (Phases 2-5) must beat


def run_once(num_steps=16, seed=None, start_time=None, scenario=None):
    """Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        None (default) preserves the original behaviour exactly (Phase 1's
        baseline scenario via run_two_radar_simulation). Pass one of
        sim.scenarios.SINGLE_TARGET_SCENARIOS for Phase 6's comparison.
    """
    if scenario is None:
        truth, steps, jpda_tracker, gmlcc_tracker = run_two_radar_simulation(
            num_steps=num_steps, seed=seed, start_time=start_time)
    else:
        truth, steps, jpda_tracker, gmlcc_tracker = run_single_target_scenario(
            scenario, seed=seed, start_time=start_time)

    fused_states = []
    for timestamp, airborne_state, ground_state in steps:
        fused = _fuse_states(airborne_state, ground_state, timestamp)
        if fused is not None:
            fused_states.append(fused)

    fused_track = Track(fused_states) if fused_states else Track()
    return truth, jpda_tracker.tracks, gmlcc_tracker.tracks, fused_track


def _fuse_states(airborne_state, ground_state, timestamp):
    """Fuse this step's two state snapshots via CI (fixed omega). Returns
    None if either sensor has no active track this step (e.g. target not
    yet initiated, or lost to clutter/deletion).
    """
    if airborne_state is None or ground_state is None:
        return None

    xa, Pa = airborne_state.state_vector, airborne_state.covar
    xb, Pb = ground_state.state_vector, ground_state.covar
    xc, Pc = covariance_intersection(xa, Pa, xb, Pb, FIXED_OMEGA)
    return GaussianState(xc, Pc, timestamp=timestamp)


def evaluate_fixed_baseline(num_steps=16, seed=None, scenario=None):
    """Real OSPA/SIAP for the fixed-omega=0.5 fused track — same shape as
    fusion.approach{1,2,3,4}'s evaluate_approachN functions, for Phase 6's
    cross-approach comparison to call all five methods uniformly."""
    truth, _, _, fused_track = run_once(num_steps=num_steps, seed=seed, scenario=scenario)
    if len(fused_track) == 0:
        raise RuntimeError("No fused states produced — airborne and ground tracks never overlapped.")
    manager = build_metric_manager()
    return compute_metrics(manager, {fused_track}, {truth})


def main():
    truth, airborne_tracks, ground_tracks, fused_track = run_once(num_steps=16, seed=2026)

    manager = build_metric_manager()
    print("=== Airborne radar only (JPDA) ===")
    for k, v in compute_metrics(manager, airborne_tracks, {truth}).items():
        print(f"  {k}: {v:.3f}")

    manager = build_metric_manager()
    print("=== Ground radar only (GM-LCC) ===")
    for k, v in compute_metrics(manager, ground_tracks, {truth}).items():
        print(f"  {k}: {v:.3f}")

    manager = build_metric_manager()
    print(f"=== Fused (Covariance Intersection, omega={FIXED_OMEGA}) ===")
    if len(fused_track) == 0:
        print("  No fused states produced — airborne and ground tracks never overlapped in time.")
    else:
        for k, v in compute_metrics(manager, {fused_track}, {truth}).items():
            print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
