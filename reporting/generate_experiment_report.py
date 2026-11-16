
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import experiment_config as config


# --------------------------------------------------
# Paths
# --------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

OUTPUT_PATH = RESULTS_DIR / "experiment_report.md"


# --------------------------------------------------
# Experiment Definitions
# --------------------------------------------------

EXPERIMENTS = [
    {
        "name": "Quantum Policy Training",
        "description": "SPSA training history and validation progress.",
        "files": [
            "quantum_training_history.csv"
        ],
    },
    {
        "name": "Agent Statistical Evaluation",
        "description": (
            "Reward, energy cost, terminal SOC and "
            "pairwise agent comparisons."
        ),
        "files": [
            "quantum_agent_comparison_summary.csv",
            "quantum_agent_comparison_details.csv",
            "quantum_agent_pairwise_comparison.csv",
        ],
    },
    {
        "name": "Quantum Policy Robustness",
        "description": (
            "Policy performance under changed battery "
            "and market conditions."
        ),
        "files": [
            "quantum_robustness_summary.csv",
            "quantum_robustness_details.csv",
        ],
    },
    {
        "name": "Quantum Circuit Ablation",
        "description": "Comparison of quantum circuit configurations.",
        "files": [
            "quantum_ablation_summary.csv"
        ],
    },
    {
        "name": "Deterministic Classical vs Quantum Benchmark",
        "description": (
            "Comparison of deterministic classical neural "
            "and quantum policies."
        ),
        "files": [
            "classical_quantum_benchmark.csv",
            "classical_quantum_benchmark_details.csv",
            "classical_quantum_paired_comparison.csv",
        ],
    },
    {
        "name": "Stochastic Classical vs Quantum Benchmark",
        "description": (
            "Comparison of stochastic action sampling, "
            "including paired statistical results."
        ),
        "files": [
            "classical_quantum_stochastic_summary.csv",
            "classical_quantum_stochastic_details.csv",
            "classical_quantum_stochastic_comparison.csv",
        ],
    },
    {
        "name": "Policy Behavior Diagnostics",
        "description": (
            "Action distribution, action switching, "
            "dominant actions and policy collapse detection."
        ),
        "files": [
            "policy_behavior_summary.csv",
            "policy_behavior_episode_details.csv",
        ],
    },
]


# --------------------------------------------------
# Figure Definitions
# --------------------------------------------------

FIGURE_FILES = [
    "agent_mean_reward.png",
    "agent_reward_distribution.png",
    "agent_energy_cost.png",
    "agent_final_soc.png",
    "agent_terminal_soc_success.png",
    "quantum_policy_win_rate.png",
    "policy_action_distribution.png",
    "policy_action_switches.png",
]


# --------------------------------------------------
# General Utilities
# --------------------------------------------------

def markdown_table(dataframe, max_rows=30):
    """Convert a DataFrame into a Markdown table."""

    if dataframe is None or dataframe.empty:
        return "_No rows available._"

    display_df = dataframe.head(max_rows).copy()
    display_df = display_df.fillna("")

    columns = [
        str(column)
        for column in display_df.columns
    ]

    def clean(value):
        value = str(value)
        return (
            value.replace("|", "\\|")
            .replace("\n", " ")
        )

    lines = [
        "| " + " | ".join(
            clean(column) for column in columns
        ) + " |",

        "| " + " | ".join(
            "---" for _ in columns
        ) + " |",
    ]

    for _, row in display_df.iterrows():
        lines.append(
            "| " + " | ".join(
                clean(value) for value in row.tolist()
            ) + " |"
        )

    if len(dataframe) > max_rows:
        lines.append(
            f"\n_Showing {max_rows} of {len(dataframe)} rows._"
        )

    return "\n".join(lines)


def load_csv(filename):
    """Load an experiment CSV if available."""

    path = RESULTS_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception as error:
        print(
            f"Warning: Could not read {filename}: {error}"
        )
        return None


def format_value(value, digits=3):
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"

    return str(value)


def numeric_column(dataframe, column):
    """Return a numeric Series or an empty Series."""

    if column not in dataframe.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        dataframe[column],
        errors="coerce"
    ).dropna()


# --------------------------------------------------
# Configuration
# --------------------------------------------------

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


# --------------------------------------------------
# Training Analysis
# --------------------------------------------------

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

        values = numeric_column(
            df,
            "validation_reward"
        )

        if not values.empty:
            content.extend([
                (
                    f"- Best validation reward: "
                    f"{values.max():.3f}"
                ),
                (
                    f"- Final validation reward: "
                    f"{values.iloc[-1]:.3f}"
                ),
            ])

    if "training_reward" in df.columns:

        values = numeric_column(
            df,
            "training_reward"
        )

        if not values.empty:
            content.append(
                f"- Mean training reward: {values.mean():.3f}"
            )

    content.extend([
        "",
        "### Recent training records",
        "",
        markdown_table(df.tail(10)),
        "",
    ])

    return "\n".join(content)


# --------------------------------------------------
# General Experiment Sections
# --------------------------------------------------

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
                    (
                        "_File not available. Run the "
                        "corresponding experiment first._"
                    ),
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
                (
                    f"Rows: {len(df)} | "
                    f"Columns: {len(df.columns)}"
                ),
                "",
                markdown_table(df),
                "",
            ])

            if filename == "classical_quantum_benchmark.csv":
                sections.extend(
                    build_benchmark_observations(df)
                )

            elif filename == "classical_quantum_stochastic_summary.csv":
                sections.extend(
                    build_stochastic_observations(df)
                )

            elif filename == "classical_quantum_stochastic_comparison.csv":
                sections.extend(
                    build_paired_comparison_observations(df)
                )

            elif filename == "policy_behavior_summary.csv":
                sections.extend(
                    build_behavior_observations(df)
                )

    return (
        "\n".join(sections),
        pd.DataFrame(availability)
    )


# --------------------------------------------------
# Deterministic Benchmark Observations
# --------------------------------------------------

def build_benchmark_observations(df):

    observations = [
        "### Deterministic benchmark observations",
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
            "_Expected benchmark columns are unavailable; "
            "automatic comparison was skipped._"
        )

        observations.append("")
        return observations

    rewards = numeric_column(
        df,
        "mean_evaluation_reward"
    )

    training = numeric_column(
        df,
        "training_time_seconds"
    )

    inference = numeric_column(
        df,
        "mean_inference_time_ms"
    )

    if len(df) >= 2 and len(rewards) == len(df):

        if rewards.nunique() == 1:
            observations.append(
                "- All agents have identical mean deterministic "
                "rewards. This may indicate convergence to "
                "similar actions or behavior."
            )

    if len(training) == len(df) and not training.empty:

        fastest = training.idxmin()

        observations.append(
            f"- Fastest training: {df.loc[fastest, 'agent']} "
            f"({training.loc[fastest]:.3f} seconds)."
        )

    if len(inference) == len(df) and not inference.empty:

        fastest = inference.idxmin()

        observations.append(
            f"- Lowest measured inference time: "
            f"{df.loc[fastest, 'agent']} "
            f"({inference.loc[fastest]:.4f} ms)."
        )

    observations.extend([
        "",
        "_These observations describe the recorded run only. "
        "They do not establish general quantum advantage._",
        "",
    ])

    return observations


# --------------------------------------------------
# Stochastic Benchmark Observations
# --------------------------------------------------

def build_stochastic_observations(df):

    observations = [
        "### Stochastic benchmark observations",
        "",
    ]

    required = {
        "agent",
        "mean_reward",
        "mean_energy_cost",
        "terminal_soc_success_rate",
    }

    if not required.issubset(df.columns):

        observations.append(
            "_Expected stochastic benchmark columns are "
            "unavailable; automatic analysis was skipped._"
        )

        observations.append("")
        return observations

    rewards = numeric_column(
        df,
        "mean_reward"
    )

    if len(df) >= 2 and len(rewards) == len(df):

        best_index = rewards.idxmax()

        observations.append(
            f"- Highest mean stochastic reward: "
            f"{df.loc[best_index, 'agent']} "
            f"({rewards.loc[best_index]:.3f})."
        )

    if "std_reward" in df.columns:

        deviations = numeric_column(
            df,
            "std_reward"
        )

        if len(deviations) == len(df) and not deviations.empty:

            stable_index = deviations.idxmin()

            observations.append(
                f"- Lowest reward variability: "
                f"{df.loc[stable_index, 'agent']} "
                f"({deviations.loc[stable_index]:.3f} standard deviation)."
            )

    observations.extend([
        "",
        "Stochastic evaluation samples actions from policy "
        "probability distributions. Its results can differ "
        "from deterministic argmax evaluation.",
        "",
    ])

    return observations


# --------------------------------------------------
# Paired Statistical Comparison
# --------------------------------------------------

def build_paired_comparison_observations(df):

    observations = [
        "### Paired statistical comparison observations",
        "",
    ]

    required = {
        "scenarios_compared",
        "quantum_wins",
        "classical_wins",
        "ties",
        "mean_reward_difference",
        "reward_difference_ci_95_lower",
        "reward_difference_ci_95_upper",
    }

    if not required.issubset(df.columns):

        observations.append(
            "_Required paired comparison columns are unavailable._"
        )

        observations.append("")
        return observations

    for _, row in df.iterrows():

        mean_difference = float(
            row["mean_reward_difference"]
        )

        lower = float(
            row["reward_difference_ci_95_lower"]
        )

        upper = float(
            row["reward_difference_ci_95_upper"]
        )

        observations.extend([
            (
                f"- Scenarios compared: "
                f"{int(row['scenarios_compared'])}"
            ),
            (
                f"- Quantum wins: {int(row['quantum_wins'])}"
            ),
            (
                f"- Classical wins: {int(row['classical_wins'])}"
            ),
            (
                f"- Ties: {int(row['ties'])}"
            ),
            (
                f"- Mean reward difference "
                f"(Quantum - Classical): {mean_difference:.3f}"
            ),
            (
                f"- 95% bootstrap confidence interval: "
                f"[{lower:.3f}, {upper:.3f}]"
            ),
        ])

        if lower > 0:
            observations.append(
                "- The recorded bootstrap interval is above zero, "
                "favoring the quantum policy on these evaluated scenarios."
            )

        elif upper < 0:
            observations.append(
                "- The recorded bootstrap interval is below zero, "
                "favoring the classical policy on these evaluated scenarios."
            )

        else:
            observations.append(
                "- The recorded bootstrap interval includes zero; "
                "the comparison does not clearly distinguish the "
                "mean reward difference from zero."
            )

        observations.append("")

    observations.append(
        "_The bootstrap interval describes uncertainty across "
        "the sampled paired scenarios. It does not establish "
        "generalization to all environments or prove quantum advantage._"
    )

    observations.append("")

    return observations


# --------------------------------------------------
# Policy Behavior Observations
# --------------------------------------------------

def build_behavior_observations(df):

    observations = [
        "### Policy behavior observations",
        "",
    ]

    required = {
        "agent",
        "evaluation_mode",
        "charge_percentage",
        "discharge_percentage",
        "hold_percentage",
        "average_switches_per_episode",
        "dominant_action",
        "dominant_action_percentage",
        "potentially_stuck",
    }

    if not required.issubset(df.columns):

        observations.append(
            "_Expected policy behavior columns are unavailable._"
        )

        observations.append("")
        return observations

    for _, row in df.iterrows():

        observations.append(
            f"**{row['agent']} — {row['evaluation_mode']}**"
        )

        observations.append(
            f"- Charge: {float(row['charge_percentage']):.2f}%"
        )

        observations.append(
            f"- Discharge: {float(row['discharge_percentage']):.2f}%"
        )

        observations.append(
            f"- Hold: {float(row['hold_percentage']):.2f}%"
        )

        observations.append(
            f"- Average action switches per episode: "
            f"{float(row['average_switches_per_episode']):.2f}"
        )

        observations.append(
            f"- Dominant action: {row['dominant_action']} "
            f"({float(row['dominant_action_percentage']):.2f}%)"
        )

        observations.append(
            f"- Potentially stuck: {row['potentially_stuck']}"
        )

        observations.append("")

    observations.append(
        "_Policy collapse is flagged when one action accounts "
        f"for at least 95% of recorded actions. This is a "
        "diagnostic threshold, not proof of a faulty policy._"
    )

    observations.append("")

    return observations


# --------------------------------------------------
# Figures
# --------------------------------------------------

def build_figures_section():

    existing_figures = [
        filename
        for filename in FIGURE_FILES
        if (FIGURES_DIR / filename).exists()
    ]

    lines = [
        "## Available Figures",
        "",
    ]

    if not existing_figures:

        lines.append(
            "_No configured comparison figures were found._"
        )

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


# --------------------------------------------------
# Report Generation
# --------------------------------------------------

def generate_report():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Generating consolidated experiment report...")

    experiment_content, availability_df = (
        build_experiment_sections()
    )

    report_parts = [

        "# Hybrid Quantum Battery Reinforcement Learning",
        "",

        "## Consolidated Experiment Report",
        "",

        (
            f"Generated: "
            f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
        ),

        "",

        (
            "This report consolidates the experiment outputs "
            "currently available in the project's results directory. "
            "It does not rerun training or evaluation."
        ),

        "",

        "## Experiment Configuration",
        "",

        markdown_table(
            get_config_summary()
        ),

        "",

        "## Output Availability",
        "",

        markdown_table(
            availability_df
        ),

        "",

        build_training_section(),

        experiment_content,

        build_figures_section(),

        "## Limitations and Interpretation",
        "",

        (
            "- Results depend on the environment, random seeds, "
            "training budget and policy implementation used."
        ),

        (
            "- Deterministic and stochastic evaluations use "
            "different action-selection procedures and should "
            "be interpreted separately."
        ),

        (
            "- Paired comparisons are based on the evaluated "
            "scenarios and do not guarantee generalization."
        ),

        (
            "- Missing CSV files indicate that an experiment "
            "output was unavailable when this report was generated."
        ),

        (
            "- Policy behavior diagnostics identify action "
            "concentration but do not independently establish "
            "the cause of that behavior."
        ),

        (
            "- Simulation results alone do not establish "
            "real-world battery performance or quantum advantage."
        ),

        "",

        "## Reproducibility",
        "",

        (
            "The configuration table records settings available "
            "in `config/experiment_config.py`. Individual "
            "experiment scripts may also define experiment-specific "
            "settings."
        ),

        (
            "CSV files in the results directory are used as "
            "the source of recorded experiment results. "
            "The report generator does not retrain policies "
            "or modify experiment checkpoints."
        ),

        "",
    ]

    report = "\n".join(report_parts)

    OUTPUT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print("\nReport generated successfully.")

    print(f"Report: {OUTPUT_PATH}")

    print("\nExperiment output availability:")

    print(
        availability_df.to_string(index=False)
    )

    print(
        f"\nReport size: {OUTPUT_PATH.stat().st_size} bytes"
    )

    return OUTPUT_PATH


if __name__ == "__main__":
    generate_report()
