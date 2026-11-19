
"""Audit experiment metadata and reproducibility artifacts.

Read-only with respect to experiment inputs and model checkpoints.
Writes separate JSON and Markdown audit reports.
"""

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import experiment_config
from env.battery_env import BatteryEnv
from reporting.experiment_metadata import (
    get_environment_defaults,
    get_git_information,
)

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

METADATA_PATH = RESULTS_DIR / "experiment_metadata.json"

JSON_OUTPUT = RESULTS_DIR / "reproducibility_audit.json"
MARKDOWN_OUTPUT = RESULTS_DIR / "reproducibility_audit.md"

REQUIRED_SECTIONS = [
    "experiment",
    "software",
    "configuration",
    "environment",
    "git",
]

REQUIRED_CHECKPOINTS = [
    "quantum_policy_best_weights.npy",
    "quantum_policy_weights.npy",
    "quantum_training_checkpoint.npz",
    "q_learning_table.npy",
]


class ReproducibilityAuditor:
    """Collect reproducibility checks and generate audit reports."""

    def __init__(self):
        self.checks = []

    def add_check(self, name, status, message):
        self.checks.append({
            "check": name,
            "status": status,
            "message": str(message),
        })

        print(f"[{status}] {name}: {message}")

    def check_metadata_file(self):
        if not METADATA_PATH.exists():
            self.add_check(
                "Metadata file",
                "FAIL",
                "experiment_metadata.json is missing.",
            )
            return None

        try:
            with METADATA_PATH.open(
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

        except (json.JSONDecodeError, OSError) as exc:
            self.add_check(
                "Metadata file",
                "FAIL",
                f"Cannot read metadata: {exc}",
            )
            return None

        self.add_check(
            "Metadata file",
            "PASS",
            "Valid JSON metadata file found.",
        )

        missing_sections = [
            section
            for section in REQUIRED_SECTIONS
            if section not in metadata
        ]

        if missing_sections:
            self.add_check(
                "Metadata structure",
                "FAIL",
                f"Missing sections: {missing_sections}",
            )
        else:
            self.add_check(
                "Metadata structure",
                "PASS",
                "All required metadata sections are present.",
            )

        return metadata

    def check_metadata_timestamp(self, metadata):
        experiment = metadata.get("experiment", {})
        timestamp = experiment.get("timestamp_utc")

        if not timestamp:
            self.add_check(
                "Metadata timestamp",
                "FAIL",
                "Experiment timestamp is missing.",
            )
            return

        try:
            parsed = datetime.fromisoformat(
                timestamp.replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                self.add_check(
                    "Metadata timestamp",
                    "WARNING",
                    "Timestamp has no timezone information.",
                )
                return

            age_days = (
                datetime.now(timezone.utc) - parsed
            ).total_seconds() / 86400

            if age_days < 0:
                self.add_check(
                    "Metadata timestamp",
                    "WARNING",
                    "Metadata timestamp is in the future.",
                )
            else:
                self.add_check(
                    "Metadata timestamp",
                    "PASS",
                    f"Valid UTC timestamp; age is {age_days:.1f} days.",
                )

        except (ValueError, TypeError):
            self.add_check(
                "Metadata timestamp",
                "FAIL",
                "Invalid ISO 8601 timestamp.",
            )

    def check_configuration(self, metadata):
        recorded = metadata.get("configuration", {})

        current = {
            name: value
            for name, value in vars(experiment_config).items()
            if name.isupper()
        }

        if not recorded:
            self.add_check(
                "Experiment configuration",
                "FAIL",
                "Recorded configuration is missing.",
            )
            return

        missing = sorted(set(current) - set(recorded))

        mismatches = {
            key: {
                "recorded": recorded.get(key),
                "current": current[key],
            }
            for key in current
            if key in recorded and recorded[key] != current[key]
        }

        if missing or mismatches:
            self.add_check(
                "Experiment configuration",
                "WARNING",
                f"Missing settings: {missing}; "
                f"changed settings: {mismatches}",
            )
        else:
            self.add_check(
                "Experiment configuration",
                "PASS",
                "Recorded configuration matches current settings.",
            )

        for key in (
            "TRAINING_SEED",
            "EVALUATION_SEED",
            "TRAINING_ITERATIONS",
            "EVALUATION_EPISODES",
        ):
            if key not in recorded:
                self.add_check(
                    f"Recorded {key}",
                    "FAIL",
                    "Required experiment setting is missing.",
                )
            else:
                self.add_check(
                    f"Recorded {key}",
                    "PASS",
                    f"Recorded value: {recorded[key]}",
                )

    def check_software(self, metadata):
        software = metadata.get("software", {})

        recorded_python = software.get("python_version")
        recorded_numpy = software.get("numpy_version")
        recorded_platform = software.get("platform")

        if not all((
            recorded_python,
            recorded_numpy,
            recorded_platform,
        )):
            self.add_check(
                "Software metadata",
                "FAIL",
                "One or more software version fields are missing.",
            )
            return

        self.add_check(
            "Recorded software",
            "PASS",
            f"Python {recorded_python.split()[0]}, "
            f"NumPy {recorded_numpy}, "
            f"{recorded_platform}",
        )

        current_python = platform.python_version()

        if recorded_python.split()[0] != current_python:
            self.add_check(
                "Python version consistency",
                "WARNING",
                f"Recorded {recorded_python.split()[0]}, "
                f"current {current_python}.",
            )
        else:
            self.add_check(
                "Python version consistency",
                "PASS",
                f"Python {current_python} matches.",
            )

        if recorded_numpy != np.__version__:
            self.add_check(
                "NumPy version consistency",
                "WARNING",
                f"Recorded {recorded_numpy}, "
                f"current {np.__version__}.",
            )
        else:
            self.add_check(
                "NumPy version consistency",
                "PASS",
                f"NumPy {np.__version__} matches.",
            )

    def check_environment(self, metadata):
        environment = metadata.get("environment", {})

        recorded_class = environment.get("class")
        recorded_defaults = environment.get(
            "default_parameters",
            {},
        )

        if recorded_class != BatteryEnv.__name__:
            self.add_check(
                "Environment class",
                "WARNING",
                f"Recorded {recorded_class}; "
                f"current {BatteryEnv.__name__}.",
            )
        else:
            self.add_check(
                "Environment class",
                "PASS",
                f"Environment class is {recorded_class}.",
            )

        current_defaults = get_environment_defaults(BatteryEnv)

        mismatches = {
            key: {
                "recorded": recorded_defaults.get(key),
                "current": current_defaults[key],
            }
            for key in current_defaults
            if (
                key not in recorded_defaults
                or recorded_defaults[key] != current_defaults[key]
            )
        }

        missing = sorted(
            set(current_defaults) - set(recorded_defaults)
        )

        if mismatches:
            self.add_check(
                "Environment parameters",
                "WARNING",
                f"Parameter differences: {mismatches}; "
                f"missing: {missing}",
            )
        else:
            self.add_check(
                "Environment parameters",
                "PASS",
                "Recorded default parameters match current defaults.",
            )

    def check_git_provenance(self, metadata):
        recorded_git = metadata.get("git", {})
        current_git = get_git_information(PROJECT_ROOT)

        recorded_commit = recorded_git.get("commit_hash")
        current_commit = current_git.get("commit_hash")

        if not recorded_commit:
            self.add_check(
                "Recorded Git commit",
                "WARNING",
                "No Git commit hash in experiment metadata.",
            )
        elif recorded_commit != current_commit:
            self.add_check(
                "Git commit consistency",
                "WARNING",
                f"Metadata commit {recorded_commit[:12]} differs "
                f"from current commit {current_commit[:12]}.",
            )
        else:
            self.add_check(
                "Git commit consistency",
                "PASS",
                "Recorded and current Git commits match.",
            )

        if current_git.get("working_tree_clean") is None:
            self.add_check(
                "Git working tree",
                "WARNING",
                "Unable to determine working-tree status.",
            )
        elif current_git["working_tree_clean"]:
            self.add_check(
                "Git working tree",
                "PASS",
                "Current working tree is clean.",
            )
        else:
            self.add_check(
                "Git working tree",
                "WARNING",
                "Current working tree contains local changes.",
            )

    def check_checkpoints(self):
        print("\nChecking model checkpoints...")

        for filename in REQUIRED_CHECKPOINTS:
            path = MODELS_DIR / filename

            if not path.exists():
                self.add_check(
                    f"Checkpoint: {filename}",
                    "WARNING",
                    "Checkpoint file is missing.",
                )
            elif path.stat().st_size == 0:
                self.add_check(
                    f"Checkpoint: {filename}",
                    "FAIL",
                    "Checkpoint file is empty.",
                )
            else:
                self.add_check(
                    f"Checkpoint: {filename}",
                    "PASS",
                    f"Available ({path.stat().st_size:,} bytes).",
                )

    def build_report(self, metadata):
        counts = {
            status: sum(
                check["status"] == status
                for check in self.checks
            )
            for status in ("PASS", "WARNING", "FAIL")
        }

        if counts["FAIL"]:
            overall = "FAIL"
        elif counts["WARNING"]:
            overall = "WARNING"
        else:
            overall = "PASS"

        return {
            "audit_timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "experiment_name": metadata.get(
                "experiment", {}
            ).get("name"),
            "overall_status": overall,
            "summary": counts,
            "checks": self.checks,
        }

    def write_reports(self, report):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with JSON_OUTPUT.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(report, file, indent=4)

        lines = [
            "# Experiment Reproducibility Audit",
            "",
            f"**Experiment:** {report['experiment_name']}",
            "",
            f"**Audit timestamp (UTC):** "
            f"{report['audit_timestamp_utc']}",
            "",
            f"**Overall status:** {report['overall_status']}",
            "",
            "## Summary",
            "",
            "| Status | Count |",
            "|---|---:|",
        ]

        for status, count in report["summary"].items():
            lines.append(f"| {status} | {count} |")

        lines.extend([
            "",
            "## Detailed Checks",
            "",
            "| Check | Status | Message |",
            "|---|---|---|",
        ])

        for check in report["checks"]:
            message = (
                check["message"]
                .replace("|", "\\|")
                .replace("\n", " ")
            )

            lines.append(
                f"| {check['check']} "
                f"| {check['status']} "
                f"| {message} |"
            )

        lines.extend([
            "",
            "## Interpretation",
            "",
            "- PASS means the specific check succeeded.",
            "- WARNING indicates a reproducibility limitation "
            "that should be reviewed.",
            "- FAIL indicates a missing or invalid required artifact.",
            "",
            "This audit checks available metadata and files. "
            "It does not prove that a previous training run can "
            "be reproduced exactly.",
            "",
        ])

        MARKDOWN_OUTPUT.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        print("\nAudit reports saved:")
        print(JSON_OUTPUT)
        print(MARKDOWN_OUTPUT)


def main():
    print("=" * 65)
    print("QUANTUM BATTERY EXPERIMENT REPRODUCIBILITY AUDIT")
    print("=" * 65)

    auditor = ReproducibilityAuditor()

    metadata = auditor.check_metadata_file()

    if metadata is not None:
        auditor.check_metadata_timestamp(metadata)
        auditor.check_configuration(metadata)
        auditor.check_software(metadata)
        auditor.check_environment(metadata)
        auditor.check_git_provenance(metadata)

    auditor.check_checkpoints()

    if metadata is None:
        metadata = {}

    report = auditor.build_report(metadata)
    auditor.write_reports(report)

    print("\n" + "=" * 65)
    print("AUDIT SUMMARY")
    print("=" * 65)

    print(f"Overall status: {report['overall_status']}")

    for status, count in report["summary"].items():
        print(f"{status}: {count}")

    if report["overall_status"] == "FAIL":
        return 1

    print("\nREPRODUCIBILITY AUDIT COMPLETED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
