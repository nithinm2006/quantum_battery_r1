
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import experiment_config
from env.battery_env import BatteryEnv
from reporting.experiment_metadata import (
    collect_experiment_metadata,
    save_experiment_metadata,
)


def main():
    metadata = collect_experiment_metadata(
        project_root=ROOT,
        experiment_config=experiment_config,
        environment_class=BatteryEnv,
        extra_metadata={
            "project": "Quantum Battery Reinforcement Learning",
            "description": (
                "Hybrid quantum reinforcement learning "
                "for battery energy management."
            ),
        },
    )

    output_path = ROOT / "results" / "experiment_metadata.json"

    save_experiment_metadata(metadata, output_path)

    print("\nExperiment metadata generated successfully.")
    print(f"Output: {output_path}")
    print(f"Experiment: {metadata['experiment']['name']}")
    print(f"Python: {metadata['software']['python_version'].split()[0]}")
    print(f"Git commit: {metadata['git']['commit_hash']}")
    print(
        f"Working tree clean: "
        f"{metadata['git']['working_tree_clean']}"
    )


if __name__ == "__main__":
    main()
