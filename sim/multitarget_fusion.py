"""
Multi-target fusion driver (Phase 6 extension) — the piece that was missing
to fold sim.scenarios' multi-target scenarios (parallel/crossing/converging/
diverging/high_clutter_multitarget) into real OSPA/SIAP evaluation.

Wires together:
    sim.two_radar_simulation.run_scenario_simulation  -> per-step dicts of
        {track_id: GaussianState} for each sensor (already multi-target-
        capable — see its docstring)
    fusion.track_association.associate_tracks         -> which airborne
        track pairs with which ground track, this step
    fusion.covariance_intersection                    -> fuse each matched
        pair, given whatever omega an omega_fn provides

Deliberately generic on omega selection (an omega_fn callback, not a fixed
0.5) so this same driver can later carry any of Approaches 1-4's learned
omega policies through multi-target scenarios too, without rewriting this
module — only evaluate_fixed_baseline_multitarget hardcodes 0.5, as the
first, simplest thing to verify end-to-end.
"""
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track

from fusion.covariance_intersection import covariance_intersection
from fusion.track_association import associate_tracks, DEFAULT_MAX_ASSOCIATION_DISTANCE
from sim.two_radar_simulation import run_scenario_simulation


def _fixed_omega_fn(omega):
    return lambda xa, Pa, xb, Pb, timestamp: omega


def build_contiguous_segments(per_step_fused):
    """Group per-step fused states into gap-free segments (pure function, no
    Stone Soup dependency, so it can be unit-tested on its own).

    Parameters
    ----------
    per_step_fused : list[dict]
        One dict per simulation step, in chronological order, mapping
        airborne_id -> fused state produced at that step. An id is simply
        absent from a step's dict if association didn't pair it that step.

    Returns
    -------
    list[list]  — each inner list is one segment: fused states for one
    airborne id on *consecutive* steps only. If an id is unmatched for one
    or more steps and then matched again, the later matches start a NEW
    segment rather than extending the old one.

    Why: Stone Soup's TrackToTruth builds each track-truth association's
    time range from the track's own first/last matched timestamps, and
    SIAP's accuracy_at_time later indexes track[timestamp] for every
    timestamp inside that range. A Track that skipped a step mid-range
    raises IndexError there. Guaranteeing every Track is internally
    contiguous makes that impossible by construction.
    """
    segments = []
    active = {}       # airborne_id -> (segment list, last step index)
    for step_idx, fused_by_id in enumerate(per_step_fused):
        for airborne_id, state in fused_by_id.items():
            entry = active.get(airborne_id)
            if entry is not None and entry[1] == step_idx - 1:
                entry[0].append(state)
                active[airborne_id] = (entry[0], step_idx)
            else:
                segment = [state]
                segments.append(segment)
                active[airborne_id] = (segment, step_idx)
    return segments


def run_multitarget_fusion(scenario, omega_fn=None, seed=None, start_time=None,
                            max_association_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE):
    """Run a (possibly multi-target) scenario through both sensors, associate
    tracks cross-sensor at every step, and fuse each matched pair.

    Parameters
    ----------
    omega_fn : callable(xa, Pa, xb, Pb, timestamp) -> float, optional
        Called once per matched pair per step to choose the mixing
        parameter fed into covariance_intersection. None (default) fixes
        omega=0.5 (the baseline).
    max_association_distance : float
        Passed straight to fusion.track_association.associate_tracks — see
        that module's docstring for what this gates and how to tune it.

    Returns
    -------
    truth_paths : set of GroundTruthPath — one per target (from
        run_scenario_simulation, unchanged).
    fused_tracks : set of Track — gap-free fused track segments. One
        airborne identity can yield several segments if association
        pairing was lost for a step and later regained (see
        build_contiguous_segments for why segments must be contiguous). Passed directly
        to eval.metrics.compute_metrics alongside truth_paths — OSPA/SIAP
        do their own truth-to-track assignment internally, the same way
        they already do for single-target tracks, so no further bookkeeping
        of "which fused track is which truth" is needed here.
    jpda_tracker, gmlcc_tracker : the tracker objects, post-run (as
        run_scenario_simulation returns them).

    Notes
    -----
    Fused tracks are grouped by *airborne* track id, split into a new
    segment whenever that id goes unmatched for a step. Extra segments
    show up in SIAP as track-number changes / lower longest-segment
    scores — which is an honest reflection of association breaking down,
    not something to hide.
    """
    if omega_fn is None:
        omega_fn = _fixed_omega_fn(0.5)

    truth_paths, steps, jpda_tracker, gmlcc_tracker = run_scenario_simulation(
        scenario, seed=seed, start_time=start_time)

    per_step_fused = []
    for timestamp, airborne_states, ground_states in steps:
        pairs = associate_tracks(
            airborne_states, ground_states, max_distance=max_association_distance)
        fused_this_step = {}
        for airborne_id, ground_id in pairs:
            airborne_state = airborne_states[airborne_id]
            ground_state = ground_states[ground_id]
            xa, Pa = airborne_state.state_vector, airborne_state.covar
            xb, Pb = ground_state.state_vector, ground_state.covar
            omega = omega_fn(xa, Pa, xb, Pb, timestamp)
            xc, Pc = covariance_intersection(xa, Pa, xb, Pb, omega)
            fused_this_step[airborne_id] = GaussianState(xc, Pc, timestamp=timestamp)
        per_step_fused.append(fused_this_step)

    tracks = {Track(states) for states in build_contiguous_segments(per_step_fused)}
    return truth_paths, tracks, jpda_tracker, gmlcc_tracker


def evaluate_fixed_baseline_multitarget(scenario, seed=None, omega=0.5,
                                         max_association_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE):
    """Real OSPA/SIAP for the fixed-omega=0.5 fused multi-target track set —
    the multi-target counterpart of sim.run_baseline.evaluate_fixed_baseline,
    same return shape (a metrics dict), so it drops into
    eval.compare_approaches' existing evaluator pattern unchanged.
    """
    from eval.metrics import build_metric_manager, compute_metrics

    truth_paths, tracks, _, _ = run_multitarget_fusion(
        scenario, omega_fn=_fixed_omega_fn(omega), seed=seed,
        max_association_distance=max_association_distance)
    if not tracks:
        raise RuntimeError(
            "No fused tracks produced — no airborne/ground track pair was ever "
            "associated this run (check max_association_distance, or this seed's "
            "draw may just be an unlucky one — tracking is stochastic)."
        )
    manager = build_metric_manager()
    return compute_metrics(manager, tracks, truth_paths)


def associate_segments(scenario, seed=None, start_time=None,
                        max_association_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE):
    """Association only, no fusion: returns the scenario's truth paths plus
    gap-free segments of (timestamp, airborne_state, ground_state) triples —
    the same ``valid_steps`` shape the single-target evaluate_approachN
    functions work on — so any omega policy (fusion.multitarget_policies)
    can be applied to each segment afterwards.
    """
    truth_paths, steps, _, _ = run_scenario_simulation(
        scenario, seed=seed, start_time=start_time)
    per_step = []
    for timestamp, airborne_states, ground_states in steps:
        pairs = associate_tracks(
            airborne_states, ground_states, max_distance=max_association_distance)
        per_step.append({
            a_id: (timestamp, airborne_states[a_id], ground_states[g_id])
            for a_id, g_id in pairs
        })
    return truth_paths, build_contiguous_segments(per_step)


def evaluate_policy_multitarget(policy, scenario, seed=None,
                                 max_association_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE):
    """Real OSPA/SIAP for one omega policy on one multi-target scenario/seed."""
    from eval.metrics import build_metric_manager, compute_metrics

    truth_paths, segments = associate_segments(
        scenario, seed=seed, max_association_distance=max_association_distance)
    tracks = set()
    for segment in segments:
        fused = policy(segment)
        states = [GaussianState(xc, Pc, timestamp=step[0])
                  for step, (xc, Pc) in zip(segment, fused)]
        tracks.add(Track(states))
    if not tracks:
        raise RuntimeError("No fused tracks produced — no airborne/ground pair was ever associated.")
    return compute_metrics(build_metric_manager(), tracks, truth_paths)


def main():
    """Smoke test: run the fixed baseline across every multi-target scenario
    and print OSPA/SIAP, so you can sanity-check association is actually
    doing something sensible before wiring in Approaches 1-4.
    """
    from sim.scenarios import MULTI_TARGET_SCENARIOS

    for scenario in MULTI_TARGET_SCENARIOS:
        print(f"=== {scenario.name} ===")
        try:
            metrics = evaluate_fixed_baseline_multitarget(scenario, seed=2026)
        except RuntimeError as e:
            print(f"  skipped: {e}")
            continue
        for k, v in metrics.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()