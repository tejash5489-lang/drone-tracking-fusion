"""
Phase 3, Approach 2 — training data generation.

Unlike Approach 1 (fusion/kl_objective.py), Approach 2's LSTM learns omega
directly from the training loss rather than from precomputed "optimal
omega" labels (blueprint 2.3: "trained end-to-end to minimize OSPA and
SIAP"). So this generator is simpler than Approach 1's — no KL/L-BFGS-B
step, and no dependency on fusion.covariance_intersection at all: it just
collects raw (sensor state, sensor covariance, ground truth) sequences from
many simulation runs, reusing the same sim.two_radar_simulation driver and
the same "contiguous chunk" handling as Approach 1's dataset (a chunk ends
whenever either sensor drops out for a step).
"""
import numpy as np

from sim.two_radar_simulation import run_two_radar_simulation, run_single_target_scenario

FIELDS = ("xa", "Pa", "xb", "Pb", "truth")


def generate_approach2_sequence(num_steps=16, seed=None, start_time=None, scenario=None):
    """Run one simulation and collect contiguous chunks of
    (airborne state, airborne covar, ground state, ground covar, truth
    state), one entry per time step where both sensors have an active
    track.

    Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        None (default) preserves the original single-scenario behaviour
        exactly. See sim.generate_omega_dataset.generate_omega_sequence's
        matching parameter for the full rationale — same pattern here.

    Returns
    -------
    list of dict
        Each dict has keys "xa", "Pa", "xb", "Pb", "truth", arrays of shape
        (T, n, 1) / (T, n, n) / (T, n, 1) — one contiguous chunk.
    """
    if scenario is None:
        truth, steps, _, _ = run_two_radar_simulation(
            num_steps=num_steps, seed=seed, start_time=start_time)
    else:
        truth, steps, _, _ = run_single_target_scenario(
            scenario, seed=seed, start_time=start_time)

    chunks = []
    current = {field: [] for field in FIELDS}
    for truth_state, (_, airborne_state, ground_state) in zip(truth, steps):
        if airborne_state is not None and ground_state is not None:
            current["xa"].append(airborne_state.state_vector)
            current["Pa"].append(airborne_state.covar)
            current["xb"].append(ground_state.state_vector)
            current["Pb"].append(ground_state.covar)
            current["truth"].append(truth_state.state_vector)
        elif current["xa"]:
            chunks.append(_finalize(current))
            current = {field: [] for field in FIELDS}
    if current["xa"]:
        chunks.append(_finalize(current))
    return chunks


def _finalize(current):
    return {k: np.stack(v).astype(np.float32) for k, v in current.items()}


def generate_dataset(num_runs=30, num_steps=16, seed_start=0, min_chunk_len=4, scenarios=None):
    """Run ``num_runs`` independent simulations and collect every chunk of
    at least ``min_chunk_len`` steps.

    Parameters
    ----------
    scenarios : list of sim.scenarios.ScenarioConfig, optional
        None (default): original behaviour, ``num_runs`` runs of Phase 1's
        baseline. A list: ``num_runs`` runs *per scenario*, pooled — see
        sim.generate_omega_dataset.generate_dataset's matching parameter.
    """
    scenario_list = scenarios if scenarios is not None else [None]
    sequences = []
    for scenario in scenario_list:
        for i in range(num_runs):
            for chunk in generate_approach2_sequence(
                    num_steps=num_steps, seed=seed_start + i, scenario=scenario):
                if len(chunk["xa"]) >= min_chunk_len:
                    sequences.append(chunk)
    return sequences
