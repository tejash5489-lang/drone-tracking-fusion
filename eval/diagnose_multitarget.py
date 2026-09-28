"""
Diagnostic for eval.compare_multitarget: are the fused segments long enough
for the learned omega policies to matter?

Per (scenario, seed) it reports:
  steps            simulation steps in the run
  truths           number of ground-truth targets
  a_tracks/g_tracks  mean tracks held per step by the airborne / ground tracker
  pairs            mean associated pairs per step
  paired_steps     fraction of steps with at least one associated pair
  segments         number of gap-free segments association produced
  len min/med/max  segment length in steps
  short(<=5)       segments no longer than Approach 1's KL warmup (5 steps),
                   where Approach 1 never actually uses its LSTM
  post_warmup      fraction of all segment-steps beyond step 5 of their segment

Read it like this: if segments are mostly short, the LSTM approaches barely
get to act, and the multi-target comparison says little about them. If
pairs is well below min(a_tracks, g_tracks), association (or its gate) is
dropping matches that probably exist.

Run:  python -m eval.diagnose_multitarget
"""
import numpy as np

from fusion.track_association import associate_tracks, DEFAULT_MAX_ASSOCIATION_DISTANCE
from sim.multitarget_fusion import build_contiguous_segments

WARMUP_STEPS = 5  # matches fusion.multitarget_policies.approach1_policy default


def summarize_run(steps, num_truths, max_distance=DEFAULT_MAX_ASSOCIATION_DISTANCE):
    """Pure summary of one run's association behaviour (no simulation here)."""
    a_counts, g_counts, pair_counts, per_step = [], [], [], []
    for timestamp, airborne_states, ground_states in steps:
        pairs = associate_tracks(airborne_states, ground_states, max_distance=max_distance)
        a_counts.append(len(airborne_states))
        g_counts.append(len(ground_states))
        pair_counts.append(len(pairs))
        per_step.append({a_id: timestamp for a_id, _ in pairs})

    segments = build_contiguous_segments(per_step)
    lengths = [len(s) for s in segments]
    total = sum(lengths)
    post = sum(max(0, n - WARMUP_STEPS) for n in lengths)
    return {
        "steps": len(steps),
        "truths": num_truths,
        "a_tracks": float(np.mean(a_counts)) if a_counts else 0.0,
        "g_tracks": float(np.mean(g_counts)) if g_counts else 0.0,
        "pairs": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "paired_steps": float(np.mean([c > 0 for c in pair_counts])) if pair_counts else 0.0,
        "segments": len(segments),
        "len_min": min(lengths) if lengths else 0,
        "len_med": float(np.median(lengths)) if lengths else 0.0,
        "len_max": max(lengths) if lengths else 0,
        "short": sum(n <= WARMUP_STEPS for n in lengths),
        "post_warmup": (post / total) if total else 0.0,
    }


def main(seeds=(5000, 5001, 5002, 5003, 5004)):
    import pandas as pd
    from sim.scenarios import MULTI_TARGET_SCENARIOS
    from sim.two_radar_simulation import run_scenario_simulation

    rows = []
    for scenario in MULTI_TARGET_SCENARIOS:
        for seed in seeds:
            truth_paths, steps, _, _ = run_scenario_simulation(scenario, seed=seed)
            row = {"scenario": scenario.name, "seed": seed}
            row.update(summarize_run(steps, len(truth_paths)))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("data/multitarget_diagnostics.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    print("\nMeans per scenario:")
    print(df.drop(columns="seed").groupby("scenario").mean().round(2).to_string())
    print("\nSaved data/multitarget_diagnostics.csv")


if __name__ == "__main__":
    main()