
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

SUMMARY_FILE = RESULTS_DIR / "quantum_robustness_summary.csv"
PAIRED_FILE = RESULTS_DIR / "quantum_robustness_paired_comparison.csv"


def load_data():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {SUMMARY_FILE}")

    if not PAIRED_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {PAIRED_FILE}")

    summary = pd.read_csv(SUMMARY_FILE)
    paired = pd.read_csv(PAIRED_FILE)

    required_summary = {
        "scenario",
        "agent",
        "mean_reward",
        "std_reward",
        "mean_energy_cost",
        "terminal_soc_success_rate",
    }

    required_paired = {
        "scenario",
        "comparison_agent",
        "comparison_agent_wins",
        "quantum_wins",
        "ties",
        "mean_reward_difference_vs_quantum",
        "reward_difference_ci_95_lower",
        "reward_difference_ci_95_upper",
    }

    missing_summary = required_summary - set(summary.columns)
    missing_paired = required_paired - set(paired.columns)

    if missing_summary:
        raise ValueError(
            f"Missing columns in robustness summary: {sorted(missing_summary)}"
        )

    if missing_paired:
        raise ValueError(
            f"Missing columns in paired comparison: {sorted(missing_paired)}"
        )

    return summary, paired


def create_agent_ranking(summary):
    ranking = (
        summary.groupby("agent")
        .agg(
            scenarios_evaluated=("scenario", "nunique"),
            average_reward=("mean_reward", "mean"),
            average_reward_std=("std_reward", "mean"),
            average_energy_cost=("mean_energy_cost", "mean"),
            average_terminal_soc_success=(
                "terminal_soc_success_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    ranking["reward_rank"] = (
        ranking["average_reward"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ranking["terminal_soc_rank"] = (
        ranking["average_terminal_soc_success"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    ranking["energy_cost_rank"] = (
        ranking["average_energy_cost"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    ranking = ranking.sort_values(
        ["reward_rank", "agent"]
    ).reset_index(drop=True)

    return ranking


def analyze_paired_comparisons(paired):
    analysis = (
        paired.groupby("comparison_agent")
        .agg(
            scenarios_compared=("scenario", "nunique"),
            total_comparison_wins=("comparison_agent_wins", "sum"),
            total_quantum_wins=("quantum_wins", "sum"),
            total_ties=("ties", "sum"),
            average_reward_difference=(
                "mean_reward_difference_vs_quantum",
                "mean",
            ),
            scenarios_favouring_comparison=(
                "mean_reward_difference_vs_quantum",
                lambda values: int((values > 0).sum()),
            ),
            scenarios_favouring_quantum=(
                "mean_reward_difference_vs_quantum",
                lambda values: int((values < 0).sum()),
            ),
        )
        .reset_index()
    )

    def classify_interval(row):
        lower = row["reward_difference_ci_95_lower"]
        upper = row["reward_difference_ci_95_upper"]

        if pd.isna(lower) or pd.isna(upper):
            return "Unavailable"

        if lower > 0:
            return "Favours comparison agent"

        if upper < 0:
            return "Favours quantum agent"

        return "Inconclusive"

    paired = paired.copy()
    paired["confidence_interval_interpretation"] = paired.apply(
        classify_interval,
        axis=1,
    )

    return analysis, paired


def write_report(ranking, paired_analysis, paired_details):
    report_file = ANALYSIS_DIR / "agent_performance_report.md"

    lines = [
        "# Agent Performance Ranking and Statistical Analysis",
        "",
        "## Evaluation overview",
        "",
        "The ranking uses average mean reward across the evaluated robustness "
        "scenarios. Terminal SOC success and energy cost are reported as "
        "separate performance indicators.",
        "",
        "## Overall agent ranking",
        "",
        "| Agent | Average reward | Reward rank | Terminal SOC success | "
        "Average energy cost |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in ranking.iterrows():
        lines.append(
            f"| {row['agent']} "
            f"| {row['average_reward']:.3f} "
            f"| {row['reward_rank']} "
            f"| {row['average_terminal_soc_success']:.3f} "
            f"| {row['average_energy_cost']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparison against the Quantum Agent",
            "",
            "Positive reward differences favour the comparison agent. "
            "Negative differences favour the Quantum Agent.",
            "",
            "| Agent | Scenarios compared | Comparison wins | "
            "Quantum wins | Ties | Average reward difference |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in paired_analysis.iterrows():
        lines.append(
            f"| {row['comparison_agent']} "
            f"| {row['scenarios_compared']} "
            f"| {row['total_comparison_wins']} "
            f"| {row['total_quantum_wins']} "
            f"| {row['total_ties']} "
            f"| {row['average_reward_difference']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Scenario-level confidence interval interpretation",
            "",
            "A 95% bootstrap confidence interval entirely above zero favours "
            "the comparison agent. An interval entirely below zero favours "
            "the Quantum Agent. An interval containing zero is inconclusive.",
            "",
            "| Scenario | Comparison agent | 95% CI lower | 95% CI upper | "
            "Interpretation |",
            "|---|---|---:|---:|---|",
        ]
    )

    for _, row in paired_details.iterrows():
        lines.append(
            f"| {row['scenario']} "
            f"| {row['comparison_agent']} "
            f"| {row['reward_difference_ci_95_lower']:.3f} "
            f"| {row['reward_difference_ci_95_upper']:.3f} "
            f"| {row['confidence_interval_interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- The overall ranking is based on average reward, not a "
            "combined score.",
            "- Terminal SOC success and energy cost are separate metrics "
            "and should be considered when interpreting the ranking.",
            "- Paired confidence intervals describe uncertainty in the "
            "estimated reward differences.",
            "- These results do not establish that quantum policies have "
            "a general advantage over classical policies.",
            "- The conclusions are limited to the evaluated agents, "
            "environment settings, and scenarios.",
            "",
        ]
    )

    report_file.write_text("\n".join(lines), encoding="utf-8")
    return report_file


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary, paired = load_data()

    ranking = create_agent_ranking(summary)
    paired_analysis, paired_details = analyze_paired_comparisons(paired)

    ranking_file = ANALYSIS_DIR / "agent_performance_ranking.csv"
    paired_file = ANALYSIS_DIR / "agent_paired_statistical_analysis.csv"

    ranking.to_csv(ranking_file, index=False)
    paired_analysis.to_csv(paired_file, index=False)

    report_file = write_report(
        ranking,
        paired_analysis,
        paired_details,
    )

    print("\nAGENT PERFORMANCE RANKING")
    print(ranking.to_string(index=False))

    print("\nPAIRED STATISTICAL ANALYSIS")
    print(paired_analysis.to_string(index=False))

    print("\nSCENARIO-LEVEL CONFIDENCE INTERVALS")
    print(
        paired_details[
            [
                "scenario",
                "comparison_agent",
                "reward_difference_ci_95_lower",
                "reward_difference_ci_95_upper",
                "confidence_interval_interpretation",
            ]
        ].to_string(index=False)
    )

    print("\nGenerated files:")
    print(ranking_file)
    print(paired_file)
    print(report_file)


if __name__ == "__main__":
    main()
