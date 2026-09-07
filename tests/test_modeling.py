from __future__ import annotations

import numpy as np
import pytest

import modeling


@pytest.fixture
def draws() -> np.ndarray:
    rng = np.random.default_rng(645)
    return np.asarray(
        [sorted(rng.choice(np.arange(1, 46), 6, replace=False)) for _ in range(120)]
    )


def test_recommendations_are_valid_unique_and_deterministic(draws: np.ndarray) -> None:
    first = modeling.generate_recommendations(draws, count=5, seed=1234)
    second = modeling.generate_recommendations(draws, count=5, seed=1234)

    assert first == second
    assert len(first) == 5
    assert len({tuple(numbers) for numbers in first}) == 5
    for numbers in first:
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers)) == 6
        assert all(1 <= number <= 45 for number in numbers)


def test_scores_cover_all_numbers_and_sum_to_one(draws: np.ndarray) -> None:
    scores = modeling.calculate_number_scores(draws)

    assert scores.shape == (45,)
    assert np.all(scores > 0)
    assert scores.sum() == pytest.approx(1.0)


def test_invalid_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="중복 없는"):
        modeling.generate_recommendations([[1, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
