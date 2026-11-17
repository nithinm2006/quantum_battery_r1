
"""Validate generated quantum battery experiment artifacts.

This script is read-only. It does not train models or modify results.
Run from the project root:

    python validation/validate_experiment_results.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


REQUIRED_CSV_FILES = {
    "Deterministic benchmark summary": (
        "classical_quantum_benchmark.csv"
    ),
    "Deterministic benchmark details": (
        "classical_quantum_benchmark_details.csv"
    ),
    "Deterministic paired comparison": (
        "classical_quantum_paired_comparison.csv"
    ),
    "Stochastic benchmark summary": (
        "classical_quantum_stochastic_summary.csv"
    ),
    "Stochastic benchmark details": (
        "classical_quantum_stochastic_details.csv"
    ),
    "Stochastic paired comparison": (
        "classical_quantum_stochastic_comparison.csv"
    ),
    "Policy behavior summary": (
        "policy_behavior_summary.csv"
    ),
    "Policy behavior episode details": (
        "policy_behavior_episode_details.csv"
    ),
}


REQUIRED_COLUMNS = {
    "classical_quantum_benchmark.csv": {
        "agent",
        "mean_evaluation_reward",
    },
    "classical_quantum_stochastic_summary.csv": {
        "agent",
        "mean_reward",
    },
    "classical_quantum_paired_comparison.csv": {
        "scenarios_compared",
        "mean_reward_difference",
    },
    "classical_quantum_stochastic_comparison.csv": {
        "scenarios_compared",
        "mean_reward_difference",
    },
    "policy_behavior_summary.csv": {
        "agent",
    },
    "policy_behavior_episode_details.csv": {
        "agent",
        "episode",
    },
}


REQUIRED_FIGURES = [
    "policy_action_distribution.png",
    "policy_action_switches.png",
]


def validate_csv_files():
    """Check required CSV artifacts and their basic integrity."""

    errors = []
    warnings = []
    loaded_data = {}

    print("\n[1/4] Checking CSV files...")

    for label, filename in REQUIRED_CSV_FILES.items():
        path = RESULTS_DIR / filename

        if not path.exists():
            errors.append(f"Missing CSV: {filename}")
            print(f"[FAIL] {label}: file missing")
            continue

        try:
            dataframe = pd.read_csv(path)
        except Exception as exc:
            errors.append(f"Cannot read {filename}: {exc}")
            print(f"[FAIL] {label}: unreadable")
            continue

        if dataframe.empty:
            errors.append(f"CSV is empty: {filename}")
            print(f"[FAIL] {label}: empty")
            continue

        loaded_data[filename] = dataframe

        required = REQUIRED_COLUMNS.get(filename, set())
        missing_columns = required - set(dataframe.columns)

        if missing_columns:
            errors.append(
                f"{filename} missing columns: "
                f"{sorted(missing_columns)}"
            )
            print(
                f"[FAIL] {label}: missing columns "
                f"{sorted(missing_columns)}"
            )
            continue

        print(
            f"[PASS] {label}: "
            f"{len(dataframe)} rows, "
            f"{len(dataframe.columns)} columns"
        )

    return loaded_data, errors, warnings


def validate_numeric_values(dataframes):
    """Detect infinite values in numeric CSV columns."""

    errors = []

    print("\n[2/4] Checking numeric values...")

    for filename, dataframe in dataframes.items():
        numeric = dataframe.select_dtypes(include=[np.number])

        if numeric.empty:
            continue

        values = numeric.to_numpy(dtype=float)

        if np.isinf(values).any():
            errors.append(
                f"Infinite numeric values found in {filename}"
            )
            print(f"[FAIL] {filename}: infinite values detected")
        else:
            print(f"[PASS] {filename}: no infinite values")

        nan_count = int(numeric.isna().sum().sum())

        if nan_count:
            print(
                f"[WARN] {filename}: "
                f"{nan_count} missing numeric value(s)"
            )

    return errors


def validate_figures():
    """Check that required diagnostic figures exist and are non-empty."""

    errors = []

    print("\n[3/4] Checking diagnostic figures...")

    for filename in REQUIRED_FIGURES:
        path = FIGURES_DIR / filename

        if not path.exists():
            errors.append(f"Missing figure: {filename}")
            print(f"[FAIL] {filename}: missing")
        elif path.stat().st_size == 0:
            errors.append(f"Empty figure: {filename}")
            print(f"[FAIL] {filename}: empty")
        else:
            print(
                f"[PASS] {filename}: "
                f"{path.stat().st_size:,} bytes"
            )

    return errors


def validate_report():
    """Check that the experiment report exists and has content."""

    errors = []

    print("\n[4/4] Checking experiment report...")

    report_path = RESULTS_DIR / "experiment_report.md"

    if not report_path.exists():
        errors.append("Experiment report is missing.")
        print("[FAIL] experiment_report.md: missing")
        return errors

    try:
        content = report_path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"Cannot read experiment report: {exc}")
        print("[FAIL] experiment_report.md: unreadable")
        return errors

    if not content.strip():
        errors.append("Experiment report is empty.")
        print("[FAIL] experiment_report.md: empty")
        return errors

    if "#" not in content:
        errors.append("Experiment report has no Markdown headings.")
        print("[FAIL] experiment_report.md: no headings")
    else:
        print("[PASS] Experiment report exists and has headings")

    return errors


def main():
    print("=" * 60)
    print("QUANTUM BATTERY EXPERIMENT RESULT VALIDATION")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print("[FAIL] Results directory does not exist.")
        return 1

    dataframes, csv_errors, _ = validate_csv_files()

    numeric_errors = validate_numeric_values(dataframes)
    figure_errors = validate_figures()
    report_errors = validate_report()

    all_errors = (
        csv_errors
        + numeric_errors
        + figure_errors
        + report_errors
    )

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"CSV files checked: {len(dataframes)}")
    print(f"Errors found: {len(all_errors)}")

    if all_errors:
        print("\nValidation failed:")

        for error in all_errors:
            print(f"- {error}")

        print("\nFix the reported issues and run validation again.")
        return 1

    print("\nALL EXPERIMENT ARTIFACTS PASSED VALIDATION.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
