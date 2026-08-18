"""
Phase 2, Approach 1, step 1 — omega-label dataset generation (blueprint 2.3):

    "run many Stone Soup simulations; at each time step, optimize omega
    offline using the L-BFGS-B algorithm to minimize the KL divergence
    between each sensor's distribution and the fused distribution — this
    produces a labeled dataset of 'optimal omega per time step.'"

Reuses sim.two_radar_simulation (same trackers/sensors as the Phase 1
baseline) for the "run a simulation" part, and fusion.kl_objective for the
"optimize omega per time step" part — this file is just the loop that ties
those two together across many runs.

Note on scale vs. the paper: the paper's own dataset was essentially one
long ~500-step run ("~500 optimized omega values used to forecast the next
50 steps"). Our scenario's sensor geometry (see sim/two_radar_simulation.py)
was tuned for Phase 1's ~16-step baseline — the ground radar's target
range grows past its 3000 m max_range well before 500 steps, since the
target flies away from it in a straight line. Rather than redesign the
scenario's motion (e.g. an orbit that keeps range roughly bounded, which
is a real follow-up worth doing before Phase 6's full comparative
evaluation), this generator instead runs *many independent* short
simulations (different seeds) and collects one short omega sequence per
run — which is also a literal reading of the blueprint's "run many
simulations" instruction. Total dataset size scales with num_runs.
"""
import numpy as np

from sim.two_radar_simulation import run_two_radar_simulation, run_single_target_scenario
from fusion.kl_objective import optimize_omega_kl


def generate_omega_sequence(num_steps=16, seed=None, start_time=None, scenario=None):
    """Run one simulation and compute the KL-optimal omega at every step
    where both sensors currently have an active track.

    Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        Defaults to None, which uses run_two_radar_simulation exactly as
        before (Phase 1's baseline scenario, ``num_steps`` honoured) —
        zero behaviour change for existing callers. Pass a single-target
        ScenarioConfig (e.g. from sim.scenarios.SINGLE_TARGET_SCENARIOS)
        to draw this sequence from a different scenario instead — Phase
        6's pooled-training use case. When given, the scenario's own
        ``num_steps`` is used and the ``num_steps`` argument is ignored.

    Returns
    -------
    list of np.ndarray
        Chunks of *contiguous* omega values (a new chunk starts whenever
        either sensor drops out for a step, e.g. before GM-LCC's mixture
        has converged, or after the target leaves a sensor's range).
        Usually a single chunk for our ~16-step scenario, but the code
        doesn't assume that.
    """
    if scenario is None:
        _, steps, _, _ = run_two_radar_simulation(
            num_steps=num_steps, seed=seed, start_time=start_time)
    else:
        _, steps, _, _ = run_single_target_scenario(
            scenario, seed=seed, start_time=start_time)

    chunks = []
    current = []
    for _, airborne_state, ground_state in steps:
        if airborne_state is not None and ground_state is not None:
            omega = optimize_omega_kl(
                airborne_state.state_vector, airborne_state.covar,
                ground_state.state_vector, ground_state.covar,
            )
            current.append(omega)
        elif current:
            chunks.append(np.array(current))
            current = []
    if current:
        chunks.append(np.array(current))
    return chunks


def generate_dataset(num_runs=30, num_steps=16, seed_start=0, min_chunk_len=4, scenarios=None):
    """Run ``num_runs`` independent simulations and collect every omega
    chunk of at least ``min_chunk_len`` steps (too-short chunks aren't
    useful for history->forecast windowing later).

    Parameters
    ----------
    scenarios : list of sim.scenarios.ScenarioConfig, optional
        Defaults to None (the original single-scenario behaviour —
        ``num_runs`` runs of Phase 1's baseline, seeded seed_start..
        seed_start+num_runs-1). Pass a list to pool ``num_runs`` runs
        *per scenario* instead (Phase 6's multi-scenario training pool) —
        the same seed range is reused for each scenario, since they're
        independent simulations with different geometry/clutter/bias, not
        repeated draws of the same one.

    Returns
    -------
    list of np.ndarray
        One array per usable chunk, across all runs (and all scenarios,
        if given).
    """
    scenario_list = scenarios if scenarios is not None else [None]
    sequences = []
    for scenario in scenario_list:
        for i in range(num_runs):
            for chunk in generate_omega_sequence(
                    num_steps=num_steps, seed=seed_start + i, scenario=scenario):
                if len(chunk) >= min_chunk_len:
                    sequences.append(chunk)
    return sequences


def save_dataset(sequences, path="data/omega_sequences.npz"):
    """Save as an .npz with keys seq_0, seq_1, ... (ragged lengths, so a
    plain 2D array won't work)."""
    np.savez(path, **{f"seq_{i}": seq for i, seq in enumerate(sequences)})


def load_dataset(path="data/omega_sequences.npz"):
    with np.load(path) as data:
        return [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]


def main():
    sequences = generate_dataset(num_runs=30, num_steps=16, seed_start=0)
    lengths = [len(s) for s in sequences]
    print(f"Generated {len(sequences)} omega sequences from 30 simulation runs.")
    if sequences:
        all_omegas = np.concatenate(sequences)
        print(f"  sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
        print(f"  omega values: mean={all_omegas.mean():.3f}, std={all_omegas.std():.3f}, "
              f"min={all_omegas.min():.3f}, max={all_omegas.max():.3f}")
        save_dataset(sequences)
        print("Saved to data/omega_sequences.npz")
    else:
        print("No usable sequences produced (all chunks shorter than min_chunk_len) — "
              "check sensor geometry / GM-LCC convergence before proceeding.")


if __name__ == "__main__":
    main()
