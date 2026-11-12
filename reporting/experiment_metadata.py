
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path

import numpy as np


def get_git_information(project_root):
    """Collect the current Git commit and working-tree status."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        return {
            "commit_hash": commit_hash,
            "working_tree_clean": not bool(git_status),
        }

    except (subprocess.SubprocessError, FileNotFoundError):
        return {
            "commit_hash": None,
            "working_tree_clean": None,
        }


def get_environment_defaults(environment_class):
    """Read default environment parameters from its constructor."""
    parameters = signature(environment_class.__init__).parameters

    defaults = {}

    for name, parameter in parameters.items():
        if name == "self":
            continue

        if parameter.default != parameter.empty:
            defaults[name] = parameter.default

    return defaults


def collect_experiment_metadata(
    project_root,
    experiment_config,
    environment_class,
    extra_metadata=None,
):
    """Collect experiment, software, environment and Git metadata."""

    config_values = {
        name: value
        for name, value in vars(experiment_config).items()
        if name.isupper()
    }

    metadata = {
        "experiment": {
            "name": config_values.get("EXPERIMENT_NAME", "unnamed_experiment"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "software": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "configuration": config_values,
        "environment": {
            "class": environment_class.__name__,
            "default_parameters": get_environment_defaults(environment_class),
        },
        "git": get_git_information(project_root),
    }

    if extra_metadata:
        metadata["additional_information"] = extra_metadata

    return metadata


def save_experiment_metadata(metadata, output_path):
    """Save collected metadata as formatted JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, default=str)

    return output_path
