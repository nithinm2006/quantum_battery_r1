
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import experiment_config as config


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_PATH = RESULTS_DIR / "experiment_report.md"

# Each entry describes one experiment and its expected output files.
EXPERIMENTS = [
    {
        "name": "Quantum Policy Training",
        "description": "SPSA training history and validation progress.",
        "files": ["quantum_training_history.csv"],
    },
    {
        "name": "Agent Statistical Evaluation",
        "description": "Reward, cost, terminal SOC and pairwise agent comparisons.",
        "files": [
            "quantum_agent_comparison_summary.csv",
            "quantum_agent_comparison_details.csv",
            "quantum_agent_pairwise_comparison.csv",
        ],
    },
    {
        "name": "Quantum Policy Robustness",
        "description": "Policy performance under changed battery and market conditions.",
        "files": [
            "quantum_robustness_summary.csv",
            "quantum_robustness_details.csv",
        ],
    },
    {
        "name": "Quantum Circuit Ablation",
        "description": "Comparison of quantum circuit depths.",
        "files": ["quantum_ablation_summary.csv"],
    },
    {
        "name": "Classical vs Quantum Benchmark",
        "description": "Comparison of a classical neural policy and quantum policy.",
        "files": [
            "classical_quantum_benchmark.csv",
            "classical_quantum_benchmark_details.csv",
        ],
    },
]

# Figures are included only when the files exist.
FIGURE_FILES = [
    "agent_mean_reward.png",
    "agent_reward_distribution.png",
    "agent_energy_cost.png",
    "agent_final_soc.png",
    "agent_terminal_soc_success.png",
    "quantum_policy_win_rate.png",
]


def markdown_table(dataframe, max_rows=20):
    """Convert a DataFrame into a Markdown table without extra dependencies."""

    if dataframe.empty:
        return "_No rows available._"

    display_df = dataframe.head(max_rows).copy()
    display_df = display_df.fillna("")

    columns = [str(column) for column in display_df.columns]

    def clean(value):
        value = str(value)
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for _, row in display_df.iterrows():
        lines.append(
            "| " + " | ".join(clean(value) for value in row.tolist()) + " |"
        )

    if len(dataframe) > max_rows:
        lines.append(
            f"\n_Showing {max_rows} of {len(dataframe)} rows._"
        )

    return "\n".join(lines)


def load_csv(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        print(f"Warning: Could not read {filename}: {error}")
        return None


def format_value(value, digits=3):
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"

    return str(value)


def get_config_summary():
    names = [
        "EXPERIMENT_NAME",
        "TRAINING_SEED",
        "EVALUATION_SEED",
        "TRAINING_ITERATIONS",
        "EPISODES_PER_ESTIMATE",
        "SPSA_LEARNING_RATE",
        "SPSA_PERTURBATION",
        "VALIDATION_EPISODES",
        "EVALUATION_EPISODES",
    ]

    rows = []

    for name in names:
        if hasattr(config, name):
            rows.append({
                "Setting": name,
                "Value": getattr(config, name),
            })

    return pd.DataFrame(rows)


def build_training_section():
    df = load_csv("quantum_training_history.csv")

    if df is None or df.empty:
        return (
            "## Quantum Training\n\n"
            "_Training history CSV is missing or empty._\n"
        )

    content = [
        "## Quantum Training",
        "",
        f"- Recorded iterations: {len(df)}",
    ]

    if "validation_reward" in df.columns:
        values = pd.to_numeric(
            df["validation_reward"], errors="coerce"
        ).dropna()

        if not values.empty:
            content.extend([
                f"- Best recorded validation reward: {values.max():.3f}",
                f"- Final validation reward: {values.iloc[-1]:.3f}",
            ])

    if "training_reward" in df.columns:
        values = pd.to_numeric(
            df["training_reward"], errors="coerce"
        ).dropna()

        if not values.empty:
            content.append(
                f"- Mean recorded training reward: {values.mean():.3f}"
            )

    content.extend([
        "",
        "### Recent training records",
        "",
        markdown_table(df.tail(10)),
        "",
    ])

    return "\n".join(content)


def build_experiment_sections():
    sections = []
    availability = []

    for experiment in EXPERIMENTS:
        sections.extend([
            f"## {experiment['name']}",
            "",
            experiment["description"],
            "",
        ])

        for filename in experiment["files"]:
            df = load_csv(filename)

            if df is None:
                availability.append({
                    "Experiment": experiment["name"],
                    "File": filename,
                    "Status": "Missing or unreadable",
                })
                sections.extend([
                    f"### {filename}",
                    "",
                    "_File not available. Run the corresponding experiment first._",
                    "",
                ])
                continue

            availability.append({
                "Experiment": experiment["name"],
                "File": filename,
                "Status": f"Available ({len(df)} rows)",
            })

            sections.extend([
                f"### {filename}",
                "",
                f"Rows: {len(df)} | Columns: {len(df.columns)}",
                "",
                markdown_table(df),
                "",
            ])

            if filename == "classical_quantum_benchmark.csv":
                sections.extend(build_benchmark_observations(df))

    return "\n".join(sections), pd.DataFrame(availability)


def build_benchmark_observations(df):
    """Highlight benchmark patterns without claiming unsupported conclusions."""

    observations = [
        "### Benchmark observations",
        "",
    ]

    required = {
        "agent",
        "mean_evaluation_reward",
        "training_time_seconds",
        "mean_inference_time_ms",
    }

    if not required.issubset(df.columns):
        observations.append(
            "_The benchmark CSV does not contain all expected columns; "
            "automatic comparisons were skipped._"
        )
        observations.append("")
        return observations

    reward_values = pd.to_numeric(
        df["mean_evaluation_reward"], errors="coerce"
    )
    training_values = pd.to_numeric(
        df["training_time_seconds"], errors="coerce"
    )
    inference_values = pd.to_numeric(
        df["mean_inference_time_ms"], errors="coerce"
    )

    if len(df) >= 2 and reward_values.notna().all():
        if reward_values.nunique() == 1:
            observations.append(
                "- All benchmarked agents have identical mean evaluation "
                "rewards. This should be investigated before drawing "
                "performance conclusions."
            )

    if len(df) >= 2 and training_values.notna().all():
        fastest = training_values.idxmin()
        observations.append(
            f"- Fastest training in this run: "
            f"{df.loc[fastest, 'agent']} "
            f"({training_values.loc[fastest]:.3f} seconds)."
        )

    if len(df) >= 2 and inference_values.notna().all():
        fastest = inference_values.idxmin()
        observations.append(
            f"- Lowest measured inference time: "
            f"{df.loc[fastest, 'agent']} "
            f"({inference_values.loc[fastest]:.4f} ms)."
        )

    observations.extend([
        "",
        "_These observations describe the recorded experiment only. "
        "They do not establish general quantum advantage._",
        "",
    ])

    return observations


def build_figures_section():
    existing_figures = [
        filename for filename in FIGURE_FILES
        if (FIGURES_DIR / filename).exists()
    ]

    lines = ["## Available Figures", ""]

    if not existing_figures:
        lines.append("_No configured comparison figures were found._")
        lines.append("")
        return "\n".join(lines)

    for filename in existing_figures:
        relative_path = Path("figures") / filename
        lines.extend([
            f"### {filename}",
            "",
            f"![{filename}]({relative_path.as_posix()})",
            "",
        ])

    return "\n".join(lines)


def generate_report():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating consolidated experiment report...")

    experiment_content, availability_df = build_experiment_sections()

    report_parts = [
        "# Hybrid Quantum Battery Reinforcement Learning",
        "",
        "## Consolidated Experiment Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "This report consolidates the experiment outputs currently available "
        "in the project's results directory. It does not rerun training "
        "or evaluation.",
        "",
        "## Experiment Configuration",
        "",
        markdown_table(get_config_summary()),
        "",
        "## Output Availability",
        "",
        markdown_table(availability_df),
        "",
        build_training_section(),
        experiment_content,
        build_figures_section(),
        "## Limitations and Interpretation",
        "",
        "- Results depend on the environment, seeds, training budget and "
        "policy implementation used in each experiment.",
        "- Missing CSV files indicate that an experiment output was not "
        "available when this report was generated.",
        "- Identical results across policies should be investigated; they "
        "should not automatically be interpreted as equivalent learning "
        "performance.",
        "- Simulation results alone do not establish real-world battery "
        "performance or quantum advantage.",
        "",
        "## Reproducibility",
        "",
        "The configuration table above records the settings available in "
        "`config/experiment_config.py`. Individual experiment scripts may "
        "also define experiment-specific settings.",
        "",
    ]

    report = "\n".join(report_parts)
    OUTPUT_PATH.write_text(report, encoding="utf-8")

    print("\nReport generated successfully.")
    print(f"Report: {OUTPUT_PATH}")
    print("\nExperiment output availability:")
    print(availability_df.to_string(index=False))

    return OUTPUT_PATH


if __name__ == "__main__":
    generate_report()
