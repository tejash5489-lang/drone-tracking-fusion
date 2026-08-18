"""
Phase 6 — comparative evaluation (blueprint 3, Phase 6):

    "Run all 5 methods (Fixed, Approach 1-4) x 5 repetitions x all
    scenarios; collect SIAP (ambiguity, completeness, positional accuracy,
    spuriousness) and OSPA distance for each run."

Scope: the 4 single-target scenarios (sim.scenarios.SINGLE_TARGET_SCENARIOS)
— the 4 multi-target geometric scenarios (parallel/crossing/converging/
diverging) and high_clutter_multitarget are excluded here. Approaches 1-4's
per-step fusion assumes a single track pair per sensor per step; extending
to multi-target needs cross-sensor track-to-track association (matching
which of sensor A's tracks corresponds to which of sensor B's), which is
real, well-scoped future work, not attempted in this pass — see
sim.two_radar_simulation.run_scenario_simulation's docstring.

Training strategy: each approach is trained *once*, on data pooled across
all four scenarios (not per-scenario), so the comparison measures how well
each method generalizes across conditions it was exposed to during
training, then evaluated with 5 repetitions (different seeds) per scenario
— matching the blueprint's "5 repetitions" structure as evaluation reps,
not independent retraining runs (the standard ML methodology, and the only
tractable reading given Approaches 1-4 each already take real wall-clock
time to train).

Everything here eventually calls the real numpy fusion.covariance_intersection
(via each approach's evaluate_approachN), so none of it produces results
until that's implemented — same gating as everything since Phase 1.
"""
import numpy as np
import pandas as pd

from sim.scenarios import SINGLE_TARGET_SCENARIOS
from sim.run_baseline import evaluate_fixed_baseline
from fusion.approach1_forecast import (
    make_windows, train_val_split, train as train_approach1, evaluate_approach1,
)
from fusion.approach2_train import train as train_approach2, evaluate_approach2
from fusion.approach3_train import refine_dataset, train_fusing_lstm, evaluate_approach3
from fusion.approach3_innovation import transition_matrix_torch, train_innovation_lstm
from fusion.approach4_evaluate import evaluate_approach4
from models.omega_lstm import OmegaForecastLSTM
from models.dynamic_fusing_lstm import DynamicFusingLSTM
from models.innovation_lstm import InnovationLSTM

NUM_TRAINING_RUNS_PER_SCENARIO = 10
NUM_REPETITIONS = 5
TRAIN_SEED_START = 0
EVAL_SEED_START = 5000  # disjoint from training seeds, so evaluation never reuses training draws

METHOD_NAMES = (
    "Fixed (omega=0.5)",
    "Approach 1 (KL-LSTM)",
    "Approach 2 (Dynamic Fusing LSTM)",
    "Approach 3 (IMM-LSTM)",
    "Approach 4 (CMA-ES)",
)


def train_all_approaches(scenarios=SINGLE_TARGET_SCENARIOS, num_runs=NUM_TRAINING_RUNS_PER_SCENARIO):
    """Train Approaches 1-4 once, pooling training data across all given
    scenarios (see module docstring for why this is "once", not per-scenario).

    Returns
    -------
    dict with keys: model1, model2, model_a, model_b, model3, F, knowledge_base
    """
    print(f"Building training data pooled across {len(scenarios)} scenarios "
          f"({num_runs} runs each)...")

    from sim.generate_omega_dataset import generate_dataset as gen_omega_dataset
    omega_sequences = gen_omega_dataset(
        num_runs=num_runs, seed_start=TRAIN_SEED_START, scenarios=scenarios)
    X, Y = make_windows(omega_sequences)
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y)
    model1 = OmegaForecastLSTM()
    train_approach1(model1, X_train, Y_train, X_val, Y_val)
    print(f"  Approach 1: trained on {len(omega_sequences)} omega sequences "
          f"({len(X_train)} train / {len(X_val)} val windows)")

    from sim.generate_approach2_dataset import generate_dataset as gen_a2_dataset
    a2_sequences = gen_a2_dataset(
        num_runs=num_runs, seed_start=TRAIN_SEED_START, scenarios=scenarios)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(a2_sequences))
    n_val = max(1, int(len(a2_sequences) * 0.2))
    val_seqs = [a2_sequences[i] for i in idx[:n_val]]
    train_seqs = [a2_sequences[i] for i in idx[n_val:]]
    model2 = DynamicFusingLSTM()
    train_approach2(model2, train_seqs, val_seqs)
    print(f"  Approach 2: trained on {len(a2_sequences)} sequences "
          f"({len(train_seqs)} train / {len(val_seqs)} val)")

    F = transition_matrix_torch()
    model_a = InnovationLSTM()
    train_innovation_lstm(model_a, train_seqs, "xa", "Pa", F)
    model_b = InnovationLSTM()
    train_innovation_lstm(model_b, train_seqs, "xb", "Pb", F)
    refined_train = refine_dataset(model_a, model_b, train_seqs, F)
    refined_val = refine_dataset(model_a, model_b, val_seqs, F)
    model3 = DynamicFusingLSTM()
    train_fusing_lstm(model3, refined_train, refined_val)
    print("  Approach 3: LSTM-1/2/3 trained (reused Approach 2's dataset)")

    from sim.generate_cma_es_knowledgebase import build_knowledgebase
    from fusion.cma_es_lookup import OmegaKnowledgeBase
    records = build_knowledgebase(num_runs=num_runs, seed_start=TRAIN_SEED_START, scenarios=scenarios)
    knowledge_base = OmegaKnowledgeBase(
        np.stack([r[0] for r in records]), np.array([r[1] for r in records]))
    print(f"  Approach 4: knowledge base built with {len(records)} records")

    return {
        "model1": model1, "model2": model2,
        "model_a": model_a, "model_b": model_b, "model3": model3, "F": F,
        "knowledge_base": knowledge_base,
    }


def evaluate_all_methods(trained, scenario, seed):
    """Run all 5 methods on one (scenario, seed). A method that fails on
    this particular draw (e.g. no confirmed track this run) is *skipped*,
    not treated as failing the whole repetition for every other method —
    tracking is stochastic (clutter, noise), so occasional no-track draws
    are expected, especially for the degraded-sensor scenarios.

    Returns
    -------
    dict of {method_name: metrics_dict}, only for methods that succeeded.
    """
    evaluators = {
        "Fixed (omega=0.5)": lambda: evaluate_fixed_baseline(seed=seed, scenario=scenario),
        "Approach 1 (KL-LSTM)": lambda: evaluate_approach1(
            trained["model1"], seed=seed, scenario=scenario),
        "Approach 2 (Dynamic Fusing LSTM)": lambda: evaluate_approach2(
            trained["model2"], seed=seed, scenario=scenario),
        "Approach 3 (IMM-LSTM)": lambda: evaluate_approach3(
            trained["model_a"], trained["model_b"], trained["model3"], trained["F"],
            seed=seed, scenario=scenario),
        "Approach 4 (CMA-ES)": lambda: evaluate_approach4(
            trained["knowledge_base"], seed=seed, scenario=scenario),
    }
    results = {}
    for name, evaluator in evaluators.items():
        try:
            results[name] = evaluator()
        except RuntimeError as e:
            print(f"    {name}: skipped ({e})")
    return results


def run_comparison(scenarios=SINGLE_TARGET_SCENARIOS, num_repetitions=NUM_REPETITIONS,
                    num_training_runs=NUM_TRAINING_RUNS_PER_SCENARIO):
    """Full Phase 6 comparison: train once, evaluate 5 methods x num_repetitions
    x scenarios. Returns a tidy pandas DataFrame — one row per
    (scenario, method, repetition), metric columns from eval.metrics'
    OSPA/SIAP names.
    """
    trained = train_all_approaches(scenarios, num_runs=num_training_runs)

    rows = []
    for scenario in scenarios:
        for rep in range(num_repetitions):
            seed = EVAL_SEED_START + rep
            print(f"Evaluating scenario={scenario.name!r} repetition={rep} (seed={seed})...")
            method_results = evaluate_all_methods(trained, scenario, seed)
            for method, metrics in method_results.items():
                row = {"scenario": scenario.name, "method": method, "repetition": rep, "seed": seed}
                row.update(metrics)
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = run_comparison()
    df.to_csv("data/phase6_comparison_results.csv", index=False)
    print(f"\nSaved {len(df)} result rows to data/phase6_comparison_results.csv")
    print(df.groupby(["scenario", "method"])[["OSPA distances", "SIAP Completeness"]].mean())


if __name__ == "__main__":
    main()
