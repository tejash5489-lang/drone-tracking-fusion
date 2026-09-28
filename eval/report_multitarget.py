"""
Charts + written report for the multi-target comparison
(data/multitarget_comparison_results.csv, from eval.compare_multitarget).

Reuses the pure helpers in eval.report_results (summary_table,
friedman_tests, rank_methods) but writes its own files, so the Phase 6
single-target outputs (data/phase6_*) are never overwritten. It also
differs from eval.report_results.write_report in two deliberate ways:

  * It does not name an overall "recommended method". With 5 repetitions
    per scenario and short fused segments, naming a winner from pooled means
    would overstate what the data supports. A metric is only called decided
    if its Friedman test is significant.
  * A Friedman test that can't be computed (e.g. Ambiguity when every
    value is exactly 1.0) is reported as "no variation to test", not as
    "not significant".

It adds a paired Wilcoxon signed-rank test of each method against the fixed
omega=0.5 baseline on OSPA, since "does adaptive omega beat fixed?" is the
question the comparison is really asking.

Run:  python -m eval.report_multitarget
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from eval.report_results import summary_table, friedman_tests, rank_methods, KEY_METRICS

FIXED = "Fixed (omega=0.5)"


def wilcoxon_vs_fixed(df, metric="OSPA distances"):
    """Paired signed-rank test of each non-fixed method vs Fixed, pairing on
    (scenario, repetition) — i.e. the same simulated run scored under two
    methods. Returns {method: (mean_diff, p_value, n_pairs)}, where
    mean_diff = method - fixed (negative = better for a lower-is-better metric).
    """
    wide = df.pivot_table(index=["scenario", "repetition"], columns="method", values=metric)
    out = {}
    for method in wide.columns:
        if method == FIXED or FIXED not in wide.columns:
            continue
        pair = wide[[method, FIXED]].dropna()
        diff = pair[method] - pair[FIXED]
        if len(pair) < 6 or np.allclose(diff, 0):
            out[method] = (float(diff.mean()) if len(pair) else float("nan"), float("nan"), len(pair))
            continue
        out[method] = (float(diff.mean()), float(wilcoxon(pair[method], pair[FIXED]).pvalue), len(pair))
    return out


def plot_charts(df, out_dir="data"):
    import matplotlib.pyplot as plt

    methods = sorted(df["method"].unique())
    scenarios = sorted(df["scenario"].unique())

    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(methods)
    x = np.arange(len(scenarios))
    for i, method in enumerate(methods):
        sub = [df[(df.scenario == s) & (df.method == method)]["OSPA distances"] for s in scenarios]
        ax.bar(x + i * width, [v.mean() for v in sub], width,
               yerr=[v.std() for v in sub], capsize=2, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Mean OSPA distance (lower is better); bars = +/-1 std")
    ax.set_title("Multi-target: mean OSPA by method and scenario")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/multitarget_ospa_bar_chart.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["OSPA distances", "SIAP Completeness"]):
        ax.boxplot([df[df.method == m][metric].dropna().values for m in methods])
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(f"{metric} (pooled across scenarios)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/multitarget_box_plots.png")
    plt.close(fig)


def write_report(df, out_path="data/multitarget_report.md"):
    friedman = friedman_tests(df)
    winners = rank_methods(df)
    versus = wilcoxon_vs_fixed(df)

    lines = ["# Multi-target comparison report\n"]
    lines.append(f"Scenarios: {sorted(df['scenario'].unique())}")
    lines.append(f"Methods: {sorted(df['method'].unique())}")
    lines.append(f"Result rows: {len(df)} "
                 f"({df['repetition'].nunique()} repetitions per scenario/method)\n")

    lines.append("## Mean +/- std per scenario per method\n")
    lines.append(summary_table(df).to_string())

    lines.append("\n\n## Friedman test across methods (blocks = scenario x repetition)\n")
    for metric, (stat, p, n) in friedman.items():
        if np.isnan(p):
            verdict = "no variation to test (or too few complete blocks)"
        else:
            verdict = "significant (p<0.05)" if p < 0.05 else "not significant"
        lines.append(f"- {metric}: p={p:.4f}, n_blocks={n} -> {verdict}")

    lines.append("\n## Each method vs fixed omega=0.5 on OSPA (paired Wilcoxon signed-rank)\n")
    lines.append("Difference = method minus fixed; negative means lower (better) OSPA. "
                 "Four comparisons are run, so a Bonferroni-adjusted threshold is p<0.0125.\n")
    for method, (diff, p, n) in versus.items():
        ptxt = "n/a" if np.isnan(p) else f"{p:.4f}"
        lines.append(f"- {method}: mean diff {diff:+.1f}, p={ptxt}, n_pairs={n}")

    lines.append("\n## Best pooled mean per metric (descriptive only)\n")
    for metric, winner in winners.items():
        p = friedman[metric][1]
        note = "" if (not np.isnan(p) and p < 0.05) else "  (difference not established)"
        lines.append(f"- {metric}: {winner}{note}")

    lines.append("\n## Caveats\n")
    lines.append("- Models were trained on single-target sequences and applied to short "
                 "multi-target segments; see data/multitarget_diagnostics*.csv for segment lengths.")
    lines.append("- Approach 1 uses real KL optimisation for its first 5 steps of each segment, "
                 "so short segments barely exercise its LSTM.")
    lines.append("- Runs are short and the ground tracker holds fewer tracks than there are "
                 "targets, so low completeness here reflects tracking, not only fusion.")

    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main():
    df = pd.read_csv("data/multitarget_comparison_results.csv")
    plot_charts(df)
    print(write_report(df))
    print("\nCharts saved to data/multitarget_ospa_bar_chart.png, data/multitarget_box_plots.png")
    print("Report saved to data/multitarget_report.md")


if __name__ == "__main__":
    main()