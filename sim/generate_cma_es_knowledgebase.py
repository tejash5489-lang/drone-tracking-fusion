"""
Phase 5, Approach 4 — offline knowledge-base generation (blueprint 2.3):

    "30 simulations x 100 time steps each generate paired sensor state
    means/covariances; for every time step, CMA-ES searches for the omega
    that minimizes log(det(Pcc))... This produces a database of ~3,000
    (state-condition -> optimal omega) records."

Same "many shorter runs" adaptation as Approach 1's dataset (see
sim/generate_omega_dataset.py's docstring for why) — our scenario's sensor
geometry doesn't support 100-step runs without redesigning the scenario's
motion. Reaching close to the paper's ~3,000-record target here means more
runs, not longer ones — num_runs is easy to scale up once there's a
scenario built for longer-duration tracking (see Phase 6).

Each record's "state-condition" fingerprint is the same flattened-covariance
feature representation used by Approaches 2/3
(models.dynamic_fusing_lstm.flatten_covariance's numpy twin, in
fusion/cma_es_lookup.py) — kept consistent because it's what the
nearest-neighbor lookup matches against at inference time.
"""
import numpy as np

from sim.two_radar_simulation import run_two_radar_simulation, run_single_target_scenario
from fusion.cma_es_objective import optimize_omega_cma_es
from fusion.cma_es_lookup import flatten_covariance_np


def generate_knowledgebase_records(
    num_steps=16, seed=None, start_time=None, x0=0.5, sigma0=0.5, maxiter=100, scenario=None,
):
    """Run one simulation and compute the CMA-ES-optimal omega + feature
    fingerprint at every step where both sensors have an active track.

    Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        None (default) preserves the original single-scenario behaviour
        exactly — see sim.generate_omega_dataset's matching parameter.

    Returns
    -------
    list of (feature: np.ndarray shape (42,), omega: float)
    """
    if scenario is None:
        _, steps, _, _ = run_two_radar_simulation(
            num_steps=num_steps, seed=seed, start_time=start_time)
    else:
        _, steps, _, _ = run_single_target_scenario(scenario, seed=seed, start_time=start_time)

    records = []
    for _, airborne_state, ground_state in steps:
        if airborne_state is None or ground_state is None:
            continue
        xa, Pa = airborne_state.state_vector, airborne_state.covar
        xb, Pb = ground_state.state_vector, ground_state.covar
        omega = optimize_omega_cma_es(
            xa, Pa, xb, Pb, x0=x0, sigma0=sigma0, maxiter=maxiter, seed=seed)
        feature = np.concatenate([flatten_covariance_np(Pa), flatten_covariance_np(Pb)])
        records.append((feature, omega))
    return records


def build_knowledgebase(num_runs=30, num_steps=16, seed_start=0, scenarios=None, **cma_kwargs):
    """Parameters
    ----------
    scenarios : list of sim.scenarios.ScenarioConfig, optional
        None (default): original behaviour. A list: ``num_runs`` runs *per
        scenario*, pooled — see sim.generate_omega_dataset's matching
        parameter.
    """
    scenario_list = scenarios if scenarios is not None else [None]
    records = []
    for scenario in scenario_list:
        for i in range(num_runs):
            records.extend(generate_knowledgebase_records(
                num_steps=num_steps, seed=seed_start + i, scenario=scenario, **cma_kwargs))
    return records


def save_knowledgebase(records, path="data/cma_es_knowledgebase.npz"):
    features = np.stack([r[0] for r in records])
    omegas = np.array([r[1] for r in records])
    np.savez(path, features=features, omegas=omegas)


def main():
    records = build_knowledgebase(num_runs=30, num_steps=16, seed_start=0)
    print(f"Generated {len(records)} knowledge-base records from 30 runs "
          f"(paper's own dataset target: ~3,000 — see module docstring).")
    if records:
        omegas = np.array([r[1] for r in records])
        print(f"  omega: mean={omegas.mean():.3f}, std={omegas.std():.3f}, "
              f"min={omegas.min():.3f}, max={omegas.max():.3f}")
        save_knowledgebase(records)
        print("Saved to data/cma_es_knowledgebase.npz")
    else:
        print("No records produced — check sensor geometry / tracker convergence.")


if __name__ == "__main__":
    main()
