"""
Cross-sensor track-to-track association (Phase 6 extension).

The gap this fills, per sim.two_radar_simulation.run_single_target_scenario's
docstring: "Multi-target scenarios need cross-sensor track-to-track
association across the two sensors' independent track sets before they can
be reduced to a single track pair." Both trackers assign their own
independent IDs to whatever they're tracking (JPDA on the airborne radar,
GM-LCC on the ground radar) — nothing ties "airborne track 3" to "ground
track 7" as the same physical target. This module is that link.

Approach: at each timestep, treat this as an assignment problem — build a
cost matrix of pairwise position distance between every currently-active
airborne track and every currently-active ground track, then solve it
optimally with the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
rather than greedy nearest-neighbor, which can pick a locally-good but
globally-inconsistent set of pairs when several targets are close together
(exactly the case the "crossing"/"converging" scenarios are designed to
stress). A gating distance then discards any assigned pair that's still too
far apart to plausibly be the same target — e.g. one sensor tracking a
target the other sensor hasn't detected at all this step, or has lost to
clutter — so a spurious pair is never forced through fusion.

This intentionally does *not* try to solve global track continuity across
time (i.e. persistently remembering "this ground ID has meant this same
target since step 12"). Each timestep is associated independently, using
only that step's positions. In practice this still produces temporally
coherent fusion (see sim.multitarget_fusion), because each JPDA/GM-LCC track
object itself is already stable across its own lifetime — the association
only needs to re-decide *which* airborne track currently pairs with *which*
ground track, not re-discover the target from scratch every step.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULT_MAX_ASSOCIATION_DISTANCE = 500.0  # metres; see module docstring on tuning this


def associate_positions(airborne_positions, ground_positions,
                         max_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE):
    """Pure, Stone-Soup-free core: match two dicts of {id: position np.ndarray}
    via optimal (Hungarian) assignment on Euclidean distance, then gate.

    Parameters
    ----------
    airborne_positions, ground_positions : dict[Any, np.ndarray]
        id -> position vector (any consistent dimensionality/units across
        both dicts — e.g. just the (x, y, z) sub-vector of a state).
    max_distance : float
        Reject (drop) any optimal pair whose distance still exceeds this —
        it's better matched to nothing than force-fused as if it were the
        same target.

    Returns
    -------
    list of (airborne_id, ground_id) pairs, each a one-to-one match.
    Unmatched ids from either side (no counterpart, or gated out) are simply
    absent from the result — same "skip, don't fuse" semantics as
    sim.two_radar_simulation's single-target _fuse_states returning None.
    """
    a_ids = list(airborne_positions)
    g_ids = list(ground_positions)
    if not a_ids or not g_ids:
        return []

    cost = np.empty((len(a_ids), len(g_ids)))
    for i, aid in enumerate(a_ids):
        pa = np.asarray(airborne_positions[aid], dtype=float).ravel()
        for j, gid in enumerate(g_ids):
            pg = np.asarray(ground_positions[gid], dtype=float).ravel()
            cost[i, j] = np.linalg.norm(pa - pg)

    row_idx, col_idx = linear_sum_assignment(cost)
    return [
        (a_ids[r], g_ids[c])
        for r, c in zip(row_idx, col_idx)
        if cost[r, c] <= max_distance
    ]


def associate_tracks(airborne_states, ground_states,
                      max_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE,
                      position_indices=(0, 2, 4)):
    """Same as associate_positions, but takes dicts of {id: GaussianState}
    (as produced by sim.two_radar_simulation.run_scenario_simulation) and
    extracts the position sub-vector itself.

    position_indices=(0, 2, 4) matches this project's state convention
    throughout (sim/scenarios.py target_initial_states, sim/scenario.py's
    transition model): [x, vx, y, vy, z, vz] — so indices 0/2/4 are the
    position components and 1/3/5 are velocity, which association
    deliberately ignores (position alone is what identifies "same place,
    same time"; velocity differences are the noisy quantity each sensor is
    still converging on independently).
    """
    airborne_positions = {
        aid: np.asarray(state.state_vector, dtype=float).ravel()[list(position_indices)]
        for aid, state in airborne_states.items()
    }
    ground_positions = {
        gid: np.asarray(state.state_vector, dtype=float).ravel()[list(position_indices)]
        for gid, state in ground_states.items()
    }
    return associate_positions(airborne_positions, ground_positions, max_distance=max_distance)