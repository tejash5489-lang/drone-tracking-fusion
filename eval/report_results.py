"""
Phase 6 — aggregation, significance testing, charts, report (blueprint 3,
Phase 6):

    "Aggregate results into comparison tables (avg +/- std. dev. per method
    per scenario) and run the Friedman test (scipy.stats.friedmanchisquare)
    to check statistical significance of differences between methods.
    Produce comparison charts (bar charts and box plots per metric across
    scenarios/methods) ... Deliverable: a final results report/dashboard
    with tables + charts + a written discussion of which method wins on
    which metric and why, plus recommended default method(s)."

Takes the tidy DataFrame produced by eval.compare_approaches.run_comparison
(one row per scenario x method x repetition) and produces everything the
blueprint's Phase 6 deliverable asks for.
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

KEY_METRICS = (
    "OSPA distances",
    "SIAP Completeness",
    "SIAP Ambiguity",
    "SIAP Spuriousness",
    "SIAP Position Accuracy",
)

# Direction each metric is "better" in — needed to pick a winner, since
# some (Completeness) are better *higher* and most (OSPA, Spuriousness,
# Position Accuracy, Ambiguity) are better *lower*.
HIGHER_IS_BETTER = {"SIAP Completeness"}


def summary_table(df, metrics=KEY_METRICS):
    """Mean +/- std per (scenario, method), for each metric — the
    blueprint's "comparison tables (avg +/- std. dev. per method per
    scenario)" deliverable."""
    agg = df.groupby(["scenario", "method"])[list(metrics)].agg(["mean", "std"])
    return agg


def friedman_tests(df, metrics=KEY_METRICS):
    """Friedman test per metric: is there a significant difference between
    the 5 methods, treating each (scenario, repetition) as a block?

    Returns
    -------
    dict of {metric: (statistic, p_value, n_blocks)}
        n_blocks is the number of complete blocks actually used (a block —
        one scenario+repetition — is dropped if any method is missing a
        value for it, since Friedman needs a complete table; see
        eval.compare_approaches.evaluate_all_methods for why a method can
        be missing — occasional no-track draws are expected).
    """
    results = {}
    methods = sorted(df["method"].unique())
    for metric in metrics:
        wide = df.pivot_table(
            index=["scenario", "repetition"], columns="method", values=metric)
        complete = wide.dropna(subset=methods)
        if len(complete) < 3:
            results[metric] = (float("nan"), float("nan"), len(complete))
            continue
        groups = [complete[m].values for m in methods]
        statistic, p_value = friedmanchisquare(*groups)
        results[metric] = (statistic, p_value, len(complete))
    return results


def rank_methods(df, metrics=KEY_METRICS):
    """For each metric, the method with the best mean value overall
    (pooled across scenarios/repetitions) — used for the "which method
    wins on which metric" discussion."""
    winners = {}
    for metric in metrics:
        means = df.groupby("method")[metric].mean()
        winners[metric] = means.idxmin() if metric not in HIGHER_IS_BETTER else means.idxmax()
    return winners


def plot_comparison_charts(df, out_dir="data"):
    """Bar chart of mean OSPA per method per scenario, plus box plots of
    OSPA and Completeness distribution per method (pooled across
    scenarios) — the blueprint's "bar charts and box plots per metric
    across scenarios/methods"."""
    import matplotlib.pyplot as plt

    methods = sorted(df["method"].unique())
    scenarios = sorted(df["scenario"].unique())

    # Bar chart: mean OSPA per method per scenario
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(methods)
    x = np.arange(len(scenarios))
    for i, method in enumerate(methods):
        means = [
            df[(df.scenario == s) & (df.method == method)]["OSPA distances"].mean()
            for s in scenarios
        ]
        ax.bar(x + i * width, means, width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Mean OSPA distance (lower is better)")
    ax.set_title("Mean OSPA by method and scenario")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/phase6_ospa_bar_chart.png")
    plt.close(fig)

    # Box plots: OSPA and Completeness distribution per method, pooled across scenarios
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["OSPA distances", "SIAP Completeness"]):
        data = [df[df.method == m][metric].dropna().values for m in methods]
        ax.boxplot(data, tick_labels=methods)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/phase6_box_plots.png")
    plt.close(fig)


def write_report(df, out_path="data/phase6_report.md"):
    """Written results report — tables + Friedman results + winners +
    a recommended default method, per the blueprint's Phase 6 deliverable."""
    summary = summary_table(df)
    friedman = friedman_tests(df)
    winners = rank_methods(df)

    lines = ["# Phase 6 — Comparative Evaluation Report\n"]
    lines.append(f"Scenarios: {sorted(df['scenario'].unique())}")
    lines.append(f"Methods: {sorted(df['method'].unique())}")
    lines.append(f"Total evaluation runs: {len(df)}\n")

    lines.append("## Mean +/- std per scenario per method\n")
    lines.append(summary.to_string())
    lines.append("")

    lines.append("\n## Friedman test (significance of differences between methods)\n")
    for metric, (stat, p, n) in friedman.items():
        sig = "significant (p<0.05)" if p < 0.05 else "not significant"
        lines.append(f"- {metric}: chi2={stat:.3f}, p={p:.4f}, n_blocks={n} -> {sig}")

    lines.append("\n## Winner per metric (best pooled mean across scenarios/repetitions)\n")
    for metric, winner in winners.items():
        lines.append(f"- {metric}: **{winner}**")

    lines.append("\n## Recommendation\n")
    win_counts = pd.Series(list(winners.values())).value_counts()
    top_method = win_counts.idxmax()
    lines.append(
        f"{top_method} wins on {win_counts[top_method]}/{len(winners)} metrics tracked here. "
        f"See the Friedman results above before treating any single win as conclusive — "
        f"a metric where the difference isn't statistically significant shouldn't drive "
        f"the recommendation on its own."
    )

    report = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main():
    df = pd.read_csv("data/phase6_comparison_results.csv")
    plot_comparison_charts(df)
    report = write_report(df)
    print(report)
    print("\nCharts saved to data/phase6_ospa_bar_chart.png, data/phase6_box_plots.png")
    print("Report saved to data/phase6_report.md")


if __name__ == "__main__":
    main()
