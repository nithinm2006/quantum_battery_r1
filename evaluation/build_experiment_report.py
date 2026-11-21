
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
FIGURES = RESULTS / "figures"

SUMMARY_FILE = RESULTS / "quantum_robustness_summary.csv"
PAIRED_FILE = RESULTS / "quantum_robustness_paired_comparison.csv"
RANKING_FILE = ANALYSIS / "agent_performance_ranking.csv"
AUDIT_FILE = RESULTS / "reproducibility_audit.md"

OUTPUT_FILE = ANALYSIS / "comprehensive_experiment_report.md"


def load_csv(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    data = pd.read_csv(path)
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    return data


def load_inputs():
    summary = load_csv(
        SUMMARY_FILE,
        [
            "scenario",
            "agent",
            "episodes",
            "mean_reward",
            "std_reward",
            "mean_energy_cost",
            "mean_wear_penalty",
            "mean_terminal_penalty",
            "mean_final_soc",
            "terminal_soc_success_rate",
        ],
    )

    paired = load_csv(
        PAIRED_FILE,
        [
            "scenario",
            "comparison_agent",
            "comparison_agent_wins",
            "quantum_wins",
            "ties",
            "mean_reward_difference_vs_quantum",
            "reward_difference_ci_95_lower",
            "reward_difference_ci_95_upper",
        ],
    )

    ranking = None

    if RANKING_FILE.exists():
        ranking = pd.read_csv(RANKING_FILE)

    return summary, paired, ranking


def format_number(value, digits=3):
    if pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}"


def build_scenario_table(summary):
    lines = [
        "| Scenario | Agent | Episodes | Mean reward | "
        "Reward std. dev. | Energy cost | Final SOC | "
        "Terminal SOC success |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary.sort_values(
        ["scenario", "agent"]
    ).iterrows():
        lines.append(
            f"| {row['scenario']} "
            f"| {row['agent']} "
            f"| {int(row['episodes'])} "
            f"| {format_number(row['mean_reward'])} "
            f"| {format_number(row['std_reward'])} "
            f"| {format_number(row['mean_energy_cost'])} "
            f"| {format_number(row['mean_final_soc'])} "
            f"| {format_number(row['terminal_soc_success_rate'] * 100, 1)}% |"
        )

    return lines


def build_ranking_table(summary, ranking):
    if ranking is not None and {
        "agent",
        "average_reward",
        "reward_rank",
        "average_terminal_soc_success",
        "average_energy_cost",
    }.issubset(ranking.columns):
        lines = [
            "| Agent | Average reward | Reward rank | "
            "Terminal SOC success | Average energy cost |",
            "|---|---:|---:|---:|---:|",
        ]

        for _, row in ranking.sort_values("reward_rank").iterrows():
            lines.append(
                f"| {row['agent']} "
                f"| {format_number(row['average_reward'])} "
                f"| {int(row['reward_rank'])} "
                f"| {format_number(row['average_terminal_soc_success'] * 100, 1)}% "
                f"| {format_number(row['average_energy_cost'])} |"
            )

        return lines

    fallback = (
        summary.groupby("agent")
        .agg(
            average_reward=("mean_reward", "mean"),
            average_energy_cost=("mean_energy_cost", "mean"),
            average_terminal_soc_success=(
                "terminal_soc_success_rate",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("average_reward", ascending=False)
    )

    lines = [
        "| Agent | Average reward | Reward rank | "
        "Terminal SOC success | Average energy cost |",
        "|---|---:|---:|---:|---:|",
    ]

    for rank, (_, row) in enumerate(fallback.iterrows(), start=1):
        lines.append(
            f"| {row['agent']} "
            f"| {format_number(row['average_reward'])} "
            f"| {rank} "
            f"| {format_number(row['average_terminal_soc_success'] * 100, 1)}% "
            f"| {format_number(row['average_energy_cost'])} |"
        )

    return lines


def build_paired_table(paired):
    lines = [
        "| Scenario | Comparison agent | Comparison wins | "
        "Quantum wins | Ties | Mean reward difference | "
        "95% CI | Interpretation |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]

    for _, row in paired.sort_values(
        ["scenario", "comparison_agent"]
    ).iterrows():
        lower = row["reward_difference_ci_95_lower"]
        upper = row["reward_difference_ci_95_upper"]

        if pd.isna(lower) or pd.isna(upper):
            interpretation = "Unavailable"
        elif lower > 0:
            interpretation = "Favours comparison agent"
        elif upper < 0:
            interpretation = "Favours quantum agent"
        else:
            interpretation = "Inconclusive"

        interval = (
            f"[{format_number(lower)}, {format_number(upper)}]"
        )

        lines.append(
            f"| {row['scenario']} "
            f"| {row['comparison_agent']} "
            f"| {int(row['comparison_agent_wins'])} "
            f"| {int(row['quantum_wins'])} "
            f"| {int(row['ties'])} "
            f"| {format_number(row['mean_reward_difference_vs_quantum'])} "
            f"| {interval} "
            f"| {interpretation} |"
        )

    return lines


def build_artifact_section():
    lines = [
        "## Generated experiment artifacts",
        "",
        "### Analysis files",
        "",
    ]

    analysis_files = [
        "agent_performance_ranking.csv",
        "agent_paired_statistical_analysis.csv",
        "agent_performance_report.md",
    ]

    for filename in analysis_files:
        path = ANALYSIS / filename
        if path.exists():
            lines.append(f"- `analysis/{filename}`")

    lines.extend(["", "### Figures", ""])

    figure_files = sorted(FIGURES.glob("robustness_*.png"))

    if figure_files:
        for path in figure_files:
            relative_path = path.relative_to(RESULTS).as_posix()
            lines.append(f"- `{relative_path}`")
    else:
        lines.append("- No robustness visualization figures found.")

    lines.extend(["", "### Reproducibility audit", ""])

    if AUDIT_FILE.exists():
        lines.append("- `reproducibility_audit.md`")
    else:
        lines.append("- Reproducibility audit report not found.")

    return lines


def generate_report(summary, paired, ranking):
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    scenarios = sorted(summary["scenario"].unique())
    agents = sorted(summary["agent"].unique())
    total_episode_runs = int(summary["episodes"].sum())

    lines = [
        "# Comprehensive Quantum Battery RL Experiment Report",
        "",
        "## 1. Executive overview",
        "",
        "This report consolidates the robustness evaluation, agent "
        "performance ranking, and paired statistical comparisons "
        "for the battery energy management environment.",
        "",
        f"- Scenarios evaluated: {len(scenarios)}",
        f"- Agents evaluated: {len(agents)}",
        f"- Total agent-episode runs represented in the summary: "
        f"{total_episode_runs}",
        "",
        "## 2. Evaluation coverage",
        "",
        "### Scenarios",
        "",
    ]

    for scenario in scenarios:
        lines.append(f"- `{scenario}`")

    lines.extend(["", "### Agents", ""])

    for agent in agents:
        lines.append(f"- `{agent}`")

    lines.extend(
        [
            "",
            "## 3. Overall agent performance ranking",
            "",
            "Ranking is based on average mean reward across the evaluated "
            "scenarios. Terminal SOC success and energy cost are included "
            "as separate metrics and are not combined into the reward rank.",
            "",
        ]
    )

    lines.extend(build_ranking_table(summary, ranking))

    lines.extend(
        [
            "",
            "## 4. Scenario-wise robustness results",
            "",
            "The following table reports the available metrics for each "
            "agent and scenario.",
            "",
        ]
    )

    lines.extend(build_scenario_table(summary))

    lines.extend(
        [
            "",
            "## 5. Paired statistical comparisons",
            "",
            "Reward differences are calculated as comparison-agent reward "
            "minus Quantum Agent reward. Positive values favour the "
            "comparison agent; negative values favour the Quantum Agent.",
            "",
            "A 95% confidence interval entirely above zero favours the "
            "comparison agent. An interval entirely below zero favours "
            "the Quantum Agent. An interval containing zero is inconclusive.",
            "",
        ]
    )

    lines.extend(build_paired_table(paired))

    lines.extend(
        [
            "",
            "## 6. Interpretation and limitations",
            "",
            "- Higher mean reward indicates better performance under the "
            "environment's defined reward function.",
            "- Terminal SOC success measures how often an agent satisfies "
            "the terminal battery condition.",
            "- Energy cost, wear penalty, and terminal penalty provide "
            "additional context for interpreting reward.",
            "- Paired comparisons use the reported scenario-level "
            "confidence intervals and win counts.",
            "- Results are limited to the agents, scenarios, seeds, and "
            "environment configurations evaluated.",
            "- The evaluation does not establish a general quantum "
            "advantage over classical reinforcement learning methods.",
            "- The `unseen_scenarios` evaluation uses different random "
            "realizations; it should not automatically be interpreted "
            "as testing a completely unseen noise distribution.",
            "",
        ]
    )

    lines.extend(build_artifact_section())

    lines.extend(
        [
            "",
            "## 7. Reproducibility",
            "",
            "Refer to the reproducibility audit artifact, when available, "
            "for experiment configuration and reproducibility checks.",
            "",
            "---",
            "",
            "Report generated from the existing experiment result files. "
            "No agent training or experiment evaluation is performed "
            "by this report-generation script.",
            "",
        ]
    )

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    return OUTPUT_FILE


def main():
    summary, paired, ranking = load_inputs()

    report_path = generate_report(summary, paired, ranking)

    print("\nCOMPREHENSIVE EXPERIMENT REPORT GENERATED")
    print(f"Output: {report_path}")

    print("\nEvaluation coverage:")
    print(f"Scenarios: {summary['scenario'].nunique()}")
    print(f"Agents: {summary['agent'].nunique()}")
    print(f"Paired comparisons: {len(paired)}")

    print("\nReport size:")
    print(f"{report_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
