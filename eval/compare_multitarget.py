"""
Five-method comparison on the multi-target scenarios — the multi-target
counterpart of eval.compare_approaches (which stays untouched).

Training reuses eval.compare_approaches.train_all_approaches unchanged, so
the models are the same ones Phase 6 evaluated: trained once on the pooled
single-target scenarios. Evaluation is what changes: cross-sensor
association (fusion.track_association) builds gap-free segments, each
approach's omega policy (fusion.multitarget_policies) runs on every
segment, and the fused tracks are scored against all truths at once.

Evaluation seeds reuse eval.compare_approaches' EVAL_SEED_START so they stay
disjoint from training seeds.

Run:  python -m eval.compare_multitarget
"""
import pandas as pd

from eval.compare_approaches import (
    train_all_approaches, EVAL_SEED_START, NUM_REPETITIONS, METHOD_NAMES,
)
from fusion.multitarget_policies import (
    fixed_policy, approach1_policy, approach2_policy, approach3_policy, approach4_policy,
)
from sim.multitarget_fusion import evaluate_policy_multitarget
from sim.scenarios import MULTI_TARGET_SCENARIOS


def build_policies(trained):
    return {
        "Fixed (omega=0.5)": fixed_policy(0.5),
        "Approach 1 (KL-LSTM)": approach1_policy(trained["model1"]),
        "Approach 2 (Dynamic Fusing LSTM)": approach2_policy(trained["model2"]),
        "Approach 3 (IMM-LSTM)": approach3_policy(
            trained["model_a"], trained["model_b"], trained["model3"], trained["F"]),
        "Approach 4 (CMA-ES)": approach4_policy(trained["knowledge_base"]),
    }


def run_multitarget_comparison(scenarios=MULTI_TARGET_SCENARIOS, num_repetitions=NUM_REPETITIONS):
    trained = train_all_approaches()
    policies = build_policies(trained)
    assert set(policies) == set(METHOD_NAMES)

    rows = []
    for scenario in scenarios:
        for rep in range(num_repetitions):
            seed = EVAL_SEED_START + rep
            print(f"Evaluating scenario={scenario.name!r} repetition={rep} (seed={seed})...")
            for name, policy in policies.items():
                try:
                    metrics = evaluate_policy_multitarget(policy, scenario, seed=seed)
                except RuntimeError as e:
                    print(f"    {name}: skipped ({e})")
                    continue
                row = {"scenario": scenario.name, "method": name, "repetition": rep, "seed": seed}
                row.update(metrics)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = run_multitarget_comparison()
    df.to_csv("data/multitarget_comparison_results.csv", index=False)
    print(f"\nSaved {len(df)} result rows to data/multitarget_comparison_results.csv")
    cols = ["OSPA distances", "SIAP Completeness", "SIAP Ambiguity", "SIAP Spuriousness"]
    print(df.groupby(["scenario", "method"])[cols].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()