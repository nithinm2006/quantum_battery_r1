
import numpy as np
import pytest

from evaluation.evaluate_quantum_robustness import (
    paired_bootstrap_comparison,
)


def test_paired_comparison_calculates_correct_wins():
    quantum = [10, 20, 30, 40]
    baseline = [15, 20, 25, 50]

    result = paired_bootstrap_comparison(
        quantum,
        baseline,
        n_bootstrap=1000,
        seed=42,
    )

    assert result["comparison_agent_wins"] == 2
    assert result["quantum_wins"] == 1
    assert result["ties"] == 1
    assert result["scenarios_compared"] == 4


def test_paired_comparison_mean_difference():
    quantum = [10, 20, 30]
    baseline = [15, 15, 35]

    result = paired_bootstrap_comparison(
        quantum,
        baseline,
        n_bootstrap=1000,
        seed=42,
    )

    assert result["mean_reward_difference_vs_quantum"] == pytest.approx(
        1.6666666667
    )


def test_paired_comparison_is_reproducible():
    quantum = [10, 20, 30, 40]
    baseline = [12, 18, 35, 38]

    first = paired_bootstrap_comparison(
        quantum,
        baseline,
        seed=42,
    )

    second = paired_bootstrap_comparison(
        quantum,
        baseline,
        seed=42,
    )

    assert first == second


def test_paired_comparison_rejects_different_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_comparison(
            [10, 20],
            [15],
        )


def test_paired_comparison_rejects_empty_arrays():
    with pytest.raises(ValueError):
        paired_bootstrap_comparison([], [])


def test_paired_comparison_rejects_non_finite_rewards():
    with pytest.raises(ValueError):
        paired_bootstrap_comparison(
            [10, np.nan],
            [15, 20],
        )


def test_paired_comparison_confidence_interval_is_ordered():
    result = paired_bootstrap_comparison(
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        n_bootstrap=1000,
    )

    assert (
        result["reward_difference_ci_95_lower"]
        <= result["reward_difference_ci_95_upper"]
    )
