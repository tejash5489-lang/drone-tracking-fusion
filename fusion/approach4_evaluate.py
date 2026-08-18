"""
Approach 4 evaluation — Phase 5 deliverable (blueprint 3, Phase 5):

    "Approach-4 pipeline benchmarked for both accuracy and (importantly)
    inference latency vs. the LSTM approaches."

Accuracy: real OSPA/SIAP (via eval.metrics), same evaluation pattern as
Phase 1's baseline and Approaches 1-3 — feed a fresh simulation's sensor
tracks through the knowledge-base lookup to get omega(t), fuse with the
real numpy covariance_intersection, build a Track, score it.

Latency: the paper's own selling point for CMA-ES (blueprint 2.5 —
"delivered comparable, stable accuracy at a fraction of the runtime cost
since it needs no live optimization") is specifically about *inference*
speed, not training speed — a knowledge-base lookup (nearest-neighbor
search over ~300 rows) against an LSTM forward pass (matrix multiplies
through a trained network). Both are fast in absolute terms at this scale;
what matters is the comparison, not either number in isolation.
"""
import time

import numpy as np
import torch


def evaluate_approach4(knowledge_base, num_steps=16, seed=None, scenario=None):
    """Real (non-differentiable) evaluation — mirrors evaluate_approach2/3:
    knowledge-base lookup for omega(t), fuse with the real numpy CI, report
    real OSPA/SIAP.

    Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        None (default) uses Phase 1's baseline scenario — pass one of
        sim.scenarios.SINGLE_TARGET_SCENARIOS for Phase 6's comparison.
    """
    from fusion.covariance_intersection import covariance_intersection
    from sim.two_radar_simulation import run_two_radar_simulation, run_single_target_scenario
    from eval.metrics import build_metric_manager, compute_metrics
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track

    if scenario is None:
        truth, steps, _, _ = run_two_radar_simulation(num_steps=num_steps, seed=seed)
    else:
        truth, steps, _, _ = run_single_target_scenario(scenario, seed=seed)
    valid_steps = [(ts, a, g) for ts, a, g in steps if a is not None and g is not None]
    if not valid_steps:
        raise RuntimeError("No steps with both sensors active — can't evaluate.")

    fused_states = []
    for timestamp, airborne_state, ground_state in valid_steps:
        xa, Pa = airborne_state.state_vector, airborne_state.covar
        xb, Pb = ground_state.state_vector, ground_state.covar
        omega, _ = knowledge_base.lookup(Pa, Pb)
        xc, Pc = covariance_intersection(xa, Pa, xb, Pb, omega)
        fused_states.append(GaussianState(xc, Pc, timestamp=timestamp))

    fused_track = Track(fused_states)
    manager = build_metric_manager()
    return compute_metrics(manager, {fused_track}, {truth})


def benchmark_lookup_latency(knowledge_base, Pa, Pb, num_calls=200):
    """Average time per knowledge-base lookup (no live optimization)."""
    # warm-up (first call pays for e.g. numpy's lazy imports)
    knowledge_base.lookup(Pa, Pb)
    start = time.perf_counter()
    for _ in range(num_calls):
        knowledge_base.lookup(Pa, Pb)
    return (time.perf_counter() - start) / num_calls


def benchmark_lstm_latency(model, features, num_calls=200):
    """Average time per LSTM forward pass, for comparison — pass in
    whatever single-step feature tensor a trained Approach 1/2/3 model
    expects (already batched/shaped correctly)."""
    model.eval()
    with torch.no_grad():
        model(features)  # warm-up
        start = time.perf_counter()
        for _ in range(num_calls):
            model(features)
    return (time.perf_counter() - start) / num_calls


def main():
    from sim.generate_cma_es_knowledgebase import build_knowledgebase, save_knowledgebase
    from fusion.cma_es_lookup import OmegaKnowledgeBase
    from models.dynamic_fusing_lstm import DynamicFusingLSTM, make_features

    print("Building knowledge base (30 runs)...")
    records = build_knowledgebase(num_runs=30, num_steps=16, seed_start=0)
    save_knowledgebase(records)
    print(f"  {len(records)} records saved to data/cma_es_knowledgebase.npz")

    kb = OmegaKnowledgeBase.load()

    print("\n=== Approach 4 (real OSPA/SIAP, via covariance_intersection.py) ===")
    for k, v in evaluate_approach4(kb, num_steps=16, seed=2026).items():
        print(f"  {k}: {v:.3f}")

    print("\n=== Inference latency: knowledge-base lookup vs. LSTM forward pass ===")
    n = 6
    Pa = np.eye(n, dtype=np.float64) * 3.0
    Pb = np.eye(n, dtype=np.float64) * 15.0
    lookup_latency = benchmark_lookup_latency(kb, Pa, Pb)
    print(f"  Approach 4 (knowledge-base lookup): {lookup_latency * 1e6:.1f} us/call")

    dummy_model = DynamicFusingLSTM()
    Pa_t = torch.from_numpy(Pa[np.newaxis, np.newaxis].astype(np.float32))
    Pb_t = torch.from_numpy(Pb[np.newaxis, np.newaxis].astype(np.float32))
    feat = make_features(Pa_t, Pb_t)
    lstm_latency = benchmark_lstm_latency(dummy_model, feat)
    print(f"  Approach 2-style LSTM forward pass (untrained, same architecture/cost "
          f"as trained): {lstm_latency * 1e6:.1f} us/call")
    print(f"  ratio: knowledge-base lookup is {lstm_latency / lookup_latency:.1f}x "
          f"faster" if lookup_latency < lstm_latency else
          f"  ratio: LSTM forward pass is {lookup_latency / lstm_latency:.1f}x faster")


if __name__ == "__main__":
    main()
