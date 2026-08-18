"""
Shared two-radar simulation driver.

Factored out of sim/run_baseline.py because every approach from here on
(Phases 2-5 in fusion/) needs the same primitive: run one scenario through
both sensors/trackers and get each sensor's current track back, step by
step. Phase 1's baseline (fixed omega=0.5) and Phase 2's label generator
(optimal omega via KL) both consume this; they just do different things
with the (airborne_track, ground_track) pair at each step.
"""
from datetime import datetime

from stonesoup.types.state import GaussianState

from sim.scenario import generate_ground_truth, generate_scenario, build_transition_model
from sim.sensors import (
    build_airborne_radar, build_ground_radar, advance_moving_sensors, apply_measurement_bias,
)
from trackers.jpda_tracker import JPDATracker
from trackers.gmlcc_tracker import GMLCCTracker

# Scenario geometry tuned (see Phase 1 verification) so the target stays
# within both radars' max_range, and GM-LCC's mixture has time to converge.
DEFAULT_TARGET_INITIAL_STATE = [0, 50, 0, 50, 1000, 0]
DEFAULT_GROUND_RADAR_POSITION = [-2000, 0, 0]
DEFAULT_AIRBORNE_INITIAL_STATE = [1500, -30, 1500, -30, 2500, -20]


def run_two_radar_simulation(num_steps=16, seed=None, start_time=None):
    """Run one scenario through both sensors/trackers.

    Returns
    -------
    truth : GroundTruthPath
    steps : list of (timestamp, airborne_state, ground_state)
        One entry per simulation time step. airborne_state / ground_state
        are *snapshots* (a fresh GaussianState, not a live reference into
        the tracker) of each sensor's current track state — or None if not
        yet initiated / lost this step. Deliberately a snapshot rather than
        the Track object itself: Track objects are mutated in place as the
        simulation continues (more states get appended to the same track
        on later steps), so storing a live reference here and reading its
        `.state` later — after the loop has moved on — would silently
        return a *later* state than the one that was actually current at
        this timestamp. (This bit an earlier version of this function.)
    jpda_tracker, gmlcc_tracker : JPDATracker, GMLCCTracker
        The tracker objects themselves, post-run — use ``.tracks`` on each
        for the *full* track history (needed for OSPA/SIAP reporting,
        which wants to see short-lived/spurious tracks too, not just
        whichever track was longest at any given step).
    """
    start_time = start_time or datetime.now()
    truth = generate_ground_truth(
        initial_state=DEFAULT_TARGET_INITIAL_STATE,
        num_steps=num_steps,
        start_time=start_time,
        seed=seed,
    )
    ground_radar = build_ground_radar(position=DEFAULT_GROUND_RADAR_POSITION, clutter_rate=1.0)
    airborne_radar = build_airborne_radar(
        initial_state=DEFAULT_AIRBORNE_INITIAL_STATE, clutter_rate=1.0, start_time=start_time)

    jpda_tracker = JPDATracker(transition_model=build_transition_model())
    gmlcc_tracker = GMLCCTracker(transition_model=build_transition_model())

    steps = []
    for state in truth:
        advance_moving_sensors([airborne_radar], state.timestamp)
        airborne_detections = airborne_radar.measure({state}, timestamp=state.timestamp, noise=True)
        ground_detections = ground_radar.measure({state}, timestamp=state.timestamp, noise=True)

        jpda_tracker.step(airborne_detections, state.timestamp, airborne_radar.measurement_model)
        gmlcc_tracker.step(ground_detections, state.timestamp)

        airborne_track = max(jpda_tracker.tracks, key=len, default=None)
        ground_track = max(gmlcc_tracker.active_tracks, key=len, default=None)
        airborne_state = _snapshot(airborne_track, state.timestamp)
        ground_state = _snapshot(ground_track, state.timestamp)
        steps.append((state.timestamp, airborne_state, ground_state))

    return truth, steps, jpda_tracker, gmlcc_tracker


def _snapshot(track, timestamp):
    """Freeze a track's current (mean, covar) into a standalone GaussianState
    that won't change as the source Track is mutated by later steps."""
    if track is None:
        return None
    return GaussianState(
        track.state.state_vector.copy(), track.state.covar.copy(), timestamp=timestamp)


def run_scenario_simulation(scenario, seed=None, start_time=None):
    """Multi-target-capable driver for a sim.scenarios.ScenarioConfig
    (Phase 6). Kept as a separate function from run_two_radar_simulation
    rather than generalizing that one in place — every Phase 1-5 module was
    built and verified against its exact single-track-pair-per-step return
    shape, and reshaping it to cover multiple simultaneous targets risked
    silently breaking that already-verified work for comparatively little
    gain (some duplicated setup code below, in exchange for zero risk to
    Phases 1-5).

    Returns
    -------
    truth_paths : set of GroundTruthPath — one per target.
    steps : list of (timestamp, airborne_states, ground_states)
        airborne_states / ground_states are dicts of {track_id: GaussianState
        snapshot}, one entry per *currently active* track on that sensor
        this step — as many entries as there are confidently-tracked
        targets, which varies over the run. Snapshotting (not storing Track
        references) for the same reason as run_two_radar_simulation — see
        _snapshot's docstring.
    jpda_tracker, gmlcc_tracker : the tracker objects, post-run.
    """
    start_time = start_time or datetime.now()
    truth_paths = generate_scenario(
        scenario.target_initial_states,
        scenario.num_steps,
        start_time=start_time,
        seed=seed,
    )
    ground_radar = build_ground_radar(
        position=scenario.ground_radar_position, clutter_rate=scenario.ground_clutter_rate)
    airborne_radar = build_airborne_radar(
        initial_state=scenario.airborne_initial_state,
        clutter_rate=scenario.airborne_clutter_rate,
        start_time=start_time,
    )

    jpda_tracker = JPDATracker(transition_model=build_transition_model())
    gmlcc_tracker = GMLCCTracker(transition_model=build_transition_model())

    timestamps = sorted({s.timestamp for path in truth_paths for s in path})
    steps = []
    for timestamp in timestamps:
        advance_moving_sensors([airborne_radar], timestamp)
        truth_states_this_step = {s for path in truth_paths for s in path if s.timestamp == timestamp}
        airborne_detections = airborne_radar.measure(
            truth_states_this_step, timestamp=timestamp, noise=True)
        ground_detections = ground_radar.measure(
            truth_states_this_step, timestamp=timestamp, noise=True)
        airborne_detections = apply_measurement_bias(
            airborne_detections, scenario.airborne_measurement_bias)
        ground_detections = apply_measurement_bias(
            ground_detections, scenario.ground_measurement_bias)

        jpda_tracker.step(airborne_detections, timestamp, airborne_radar.measurement_model)
        gmlcc_tracker.step(ground_detections, timestamp)

        airborne_states = {t.id: _snapshot(t, timestamp) for t in jpda_tracker.tracks}
        ground_states = {t.id: _snapshot(t, timestamp) for t in gmlcc_tracker.active_tracks}
        steps.append((timestamp, airborne_states, ground_states))

    return truth_paths, steps, jpda_tracker, gmlcc_tracker


def run_single_target_scenario(scenario, seed=None, start_time=None):
    """Adapter: run any single-target sim.scenarios.ScenarioConfig through
    run_scenario_simulation, but return it in run_two_radar_simulation's
    single-track-pair shape — (truth, steps, jpda_tracker, gmlcc_tracker)
    with steps as (timestamp, airborne_state, ground_state) tuples, not
    dicts.

    This is what lets Phase 6's diverse scenarios (airborne_degraded,
    ground_degraded, ground_biased, ...) plug into the *existing*,
    already-verified Approach 1-4 dataset generators and evaluation
    functions — all of which were written against run_two_radar_simulation's
    shape and only ever exercised the Phase 1 baseline scenario — without
    modifying any of that code's control flow, only widening what scenario
    it can be pointed at (see each dataset generator's optional ``scenario``
    parameter).

    Raises
    ------
    ValueError
        If ``scenario`` defines more than one target — multi-target
        scenarios need track-to-track association across the two sensors'
        independent track sets before they can be reduced to a single
        track pair, which is explicitly out of scope here (see
        run_scenario_simulation's docstring and the Phase 6 README notes).
    """
    if len(scenario.target_initial_states) != 1:
        raise ValueError(
            f"run_single_target_scenario requires a single-target scenario, "
            f"but '{scenario.name}' defines {len(scenario.target_initial_states)} targets. "
            f"Multi-target scenarios need cross-sensor track-to-track association first — "
            f"see run_scenario_simulation's docstring."
        )

    truth_paths, general_steps, jpda_tracker, gmlcc_tracker = run_scenario_simulation(
        scenario, seed=seed, start_time=start_time)

    truth = next(iter(truth_paths))
    steps = []
    for timestamp, airborne_states, ground_states in general_steps:
        airborne_state = next(iter(airborne_states.values()), None)
        ground_state = next(iter(ground_states.values()), None)
        steps.append((timestamp, airborne_state, ground_state))

    return truth, steps, jpda_tracker, gmlcc_tracker
