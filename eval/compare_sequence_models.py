"""
Phase 7 — LSTM vs GRU vs TCN for Approach 2 (Dynamic Fusing).

Protocol (deliberately identical to Phase 6 so numbers are comparable):
  * Training data: sim.generate_approach2_dataset, pooled across the four
    single-target scenarios, 10 runs each, seeds from TRAIN_SEED_START, with
    Phase 6's 80/20 split (RandomState(0)). Every architecture trains on
    exactly the same sequences with fusion.approach2_train.train (Adam,
    lr 0.001, 50 epochs).
  * Evaluation: the four single-target scenarios x NUM_REPETITIONS runs,
    seeds from EVAL_SEED_START — disjoint from training seeds. Real
    covariance_intersection + real OSPA/SIAP via eval.metrics.

Two things differ from Phase 6, on purpose:
  * Each architecture is trained from several random initialisations
    (--init-seeds, default 3). With ~32 training sequences, one lucky or
    unlucky initialisation can outweigh the architecture, so a single run
    per architecture would mostly measure luck.
  * Each (scenario, seed) is simulated once and every trained model is scored
    on that same run, instead of re-simulating per model — the simulation
    dominates runtime, and it makes the comparison properly paired.
    _score_run below copies the tail of fusion.approach2_train.
    evaluate_approach2; if you change that function, mirror it here.

A fixed omega=0.5 reference is scored on the same runs.

Run:  python -m eval.compare_sequence_models [--init-seeds 3] [--reps 5]
"""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import friedmanchisquare, wilcoxon

from eval.compare_approaches import (
    NUM_TRAINING_RUNS_PER_SCENARIO, NUM_REPETITIONS, TRAIN_SEED_START, EVAL_SEED_START,
)
from fusion.approach2_train import train as train_approach2
from models.dynamic_fusing_lstm import DynamicFusingLSTM, make_features
from models.sequence_variants import DynamicFusingGRU, DynamicFusingTCN
from sim.scenarios import SINGLE_TARGET_SCENARIOS

ARCHITECTURES = {
    "LSTM": DynamicFusingLSTM,
    "GRU": DynamicFusingGRU,
    "TCN (raw)": DynamicFusingTCN,
    "TCN (log input)": lambda: DynamicFusingTCN(input_transform="signed_log"),
}
FIXED = "Fixed (omega=0.5)"
METRICS = ("OSPA distances", "SIAP Completeness", "SIAP Spuriousness", "SIAP Position Accuracy")


def build_training_split():
    from sim.generate_approach2_dataset import generate_dataset
    sequences = generate_dataset(
        num_runs=NUM_TRAINING_RUNS_PER_SCENARIO, seed_start=TRAIN_SEED_START,
        scenarios=SINGLE_TARGET_SCENARIOS)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * 0.2))
    return [sequences[i] for i in idx[n_val:]], [sequences[i] for i in idx[:n_val]]


def train_models(init_seeds):
    """{(architecture, init_seed): (model, params, final_val_loss)}."""
    train_seqs, val_seqs = build_training_split()
    print(f"Training data: {len(train_seqs)} train / {len(val_seqs)} val sequences")
    models = {}
    for name, cls in ARCHITECTURES.items():
        for s in range(init_seeds):
            torch.manual_seed(s)
            model = cls()
            _, val_losses = train_approach2(model, train_seqs, val_seqs)
            n_params = sum(p.numel() for p in model.parameters())
            models[(name, s)] = (model, n_params, val_losses[-1])
            print(f"  {name} init_seed={s}: params={n_params}, final val loss={val_losses[-1]:.3f}")
    return models


def _score_run(valid_steps, omega_seq, truth):
    from fusion.covariance_intersection import covariance_intersection
    from eval.metrics import build_metric_manager, compute_metrics
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track

    fused = []
    for (timestamp, a, g), omega in zip(valid_steps, omega_seq):
        xc, Pc = covariance_intersection(
            a.state_vector, a.covar, g.state_vector, g.covar, float(omega))
        fused.append(GaussianState(xc, Pc, timestamp=timestamp))
    return compute_metrics(build_metric_manager(), {Track(fused)}, {truth})


def evaluate(models, num_reps):
    from sim.two_radar_simulation import run_single_target_scenario

    rows = []
    for scenario in SINGLE_TARGET_SCENARIOS:
        for rep in range(num_reps):
            seed = EVAL_SEED_START + rep
            print(f"Evaluating scenario={scenario.name!r} repetition={rep} (seed={seed})...")
            truth, steps, _, _ = run_single_target_scenario(scenario, seed=seed)
            valid = [(ts, a, g) for ts, a, g in steps if a is not None and g is not None]
            if not valid:
                print("    no steps with both sensors active — skipped")
                continue
            Pa = torch.from_numpy(np.stack([a.covar for _, a, _ in valid]).astype(np.float32))
            Pb = torch.from_numpy(np.stack([g.covar for _, _, g in valid]).astype(np.float32))
            features = make_features(Pa, Pb).unsqueeze(0)

            def add(method, init_seed, omega_seq):
                row = {"scenario": scenario.name, "method": method, "init_seed": init_seed,
                       "repetition": rep, "seed": seed}
                row.update(_score_run(valid, omega_seq, truth))
                row["omega_mean"] = float(np.mean(omega_seq))
                row["omega_std"] = float(np.std(omega_seq))
                rows.append(row)

            add(FIXED, -1, np.full(len(valid), 0.5))
            for (name, s), (model, _, _) in models.items():
                model.eval()
                with torch.no_grad():
                    add(name, s, model(features).squeeze(0).numpy())
    return pd.DataFrame(rows)


def compare_architectures(df, reference="LSTM"):
    """Paired statistics across architectures (pure; no simulation needed).

    Scores are first averaged over init seeds within each (scenario,
    repetition) block, so a block is one simulated run and the comparison is
    paired on it. Friedman is run over the architectures (fixed excluded),
    then each other architecture is tested against ``reference`` with a
    paired Wilcoxon signed-rank test.

    Returns {metric: {"blocks": n, "friedman_p": p, "vs_reference": {arch: (mean_diff, p)}}}
    """
    archs = [a for a in ARCHITECTURES if a in set(df["method"])]
    out = {}
    for metric in METRICS:
        wide = (df[df.method.isin(archs)]
                .groupby(["scenario", "repetition", "method"])[metric].mean()
                .unstack("method").dropna())
        entry = {"blocks": len(wide), "friedman_p": float("nan"), "vs_reference": {}}
        if len(wide) >= 3 and len(archs) >= 3:
            entry["friedman_p"] = float(friedmanchisquare(*[wide[a] for a in archs]).pvalue)
        for arch in archs:
            if arch == reference or reference not in wide:
                continue
            diff = wide[arch] - wide[reference]
            p = (float("nan") if len(wide) < 6 or np.allclose(diff, 0)
                 else float(wilcoxon(wide[arch], wide[reference]).pvalue))
            entry["vs_reference"][arch] = (float(diff.mean()), p)
        out[metric] = entry
    return out


def write_report(df, models, out_path="data/phase7_sequence_models_report.md"):
    lines = ["# Phase 7 — LSTM vs GRU vs TCN (Approach 2)\n"]
    n_seeds = df[df.method != FIXED]["init_seed"].nunique()
    lines.append(f"Init seeds per architecture: {n_seeds}. Evaluation runs: "
                 f"{df[['scenario', 'repetition']].drop_duplicates().shape[0]} "
                 f"(scenario x repetition).\n")

    lines.append("## Model size and final validation loss (mean over init seeds)\n")
    for name in ARCHITECTURES:
        entries = [v for (n, _), v in models.items() if n == name]
        lines.append(f"- {name}: {entries[0][1]} parameters, "
                     f"val loss {np.mean([e[2] for e in entries]):.3f} "
                     f"(+/- {np.std([e[2] for e in entries]):.3f} across inits)")

    lines.append("\n## Pooled mean +/- std across scenarios and runs (init seeds averaged first)\n")
    per_block = (df.groupby(["scenario", "repetition", "method"])[list(METRICS)].mean()
                 .reset_index())
    lines.append(per_block.groupby("method")[list(METRICS)].agg(["mean", "std"]).round(3).to_string())

    lines.append("\n\n## Per scenario (mean)\n")
    lines.append(per_block.groupby(["scenario", "method"])[list(METRICS)].mean().round(3).to_string())

    lines.append("\n\n## Paired tests between architectures\n")
    lines.append("Blocks = scenario x repetition, init seeds averaged. Friedman across "
                 "the four architectures; then each vs LSTM by Wilcoxon signed-rank. Three comparisons per "
                 "metric, so a Bonferroni threshold of p<0.017 per metric is sensible.\n")
    for metric, e in compare_architectures(df).items():
        fp = "n/a" if np.isnan(e["friedman_p"]) else f"{e['friedman_p']:.4f}"
        lines.append(f"- {metric} (blocks={e['blocks']}, Friedman p={fp})")
        for arch, (d, p) in e["vs_reference"].items():
            ptxt = "n/a" if np.isnan(p) else f"{p:.4f}"
            lines.append(f"    - {arch} minus LSTM: mean diff {d:+.3f}, p={ptxt}")

    lines.append("\n## Omega behaviour (mean over runs of each run's mean / std of omega)\n")
    lines.append("A std near 0 means the model outputs an almost constant omega; a mean "
                 "pinned near 0 or 1 means the output is saturated.\n")
    om = df.groupby("method")[["omega_mean", "omega_std"]].mean().round(3)
    lines.append(om.to_string())

    lines.append("\n## Caveats\n")
    lines.append("- Small data (about 32 training sequences) and few evaluation runs: "
                 "only large, consistent differences are worth reading into.")
    lines.append("- Same scenarios for training and evaluation, different seeds: this "
                 "measures fit to these four conditions, not generalisation to new ones.")
    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-seeds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=NUM_REPETITIONS)
    args = parser.parse_args()

    models = train_models(args.init_seeds)
    df = evaluate(models, args.reps)
    df.to_csv("data/phase7_sequence_models.csv", index=False)
    print(f"\nSaved {len(df)} result rows to data/phase7_sequence_models.csv\n")
    print(write_report(df, models))
    print("\nReport saved to data/phase7_sequence_models_report.md")


if __name__ == "__main__":
    main()