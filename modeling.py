"""Fast, transparent recommendation engine for Lotto 6/45.

Lottery draws are independent random events. This module deliberately returns
balanced, data-inspired combinations without presenting them as better odds.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


def _as_draw_matrix(data: Iterable[Iterable[int]]) -> np.ndarray:
    matrix = np.asarray(list(data), dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] != 6 or len(matrix) < 2:
        raise ValueError("추천에는 최소 2개 회차의 6개 당첨번호가 필요합니다.")
    for row in matrix:
        if len(set(row.tolist())) != 6 or np.any((row < 1) | (row > 45)):
            raise ValueError("당첨번호는 중복 없는 1~45의 숫자 6개여야 합니다.")
    return np.sort(matrix, axis=1)


def _minmax(values: np.ndarray) -> np.ndarray:
    spread = float(values.max() - values.min())
    if spread == 0:
        return np.ones_like(values, dtype=float) * 0.5
    return (values - values.min()) / spread


def calculate_number_scores(
    data: Iterable[Iterable[int]], lookback: int = 80
) -> np.ndarray:
    """Blend long-run frequency, recency and overdue diversity into 45 weights."""
    draws = _as_draw_matrix(data)
    lookback = max(10, min(int(lookback), len(draws)))
    recent = draws[-lookback:]

    long_counts = np.bincount(draws.ravel(), minlength=46)[1:].astype(float)
    recent_counts = np.zeros(45, dtype=float)
    decay_weights = np.exp(np.linspace(-2.0, 0.0, len(recent)))
    for draw, weight in zip(recent, decay_weights):
        recent_counts[draw - 1] += weight

    gaps = np.full(45, len(draws), dtype=float)
    for reverse_index, draw in enumerate(draws[::-1]):
        unseen = gaps == len(draws)
        present = np.zeros(45, dtype=bool)
        present[draw - 1] = True
        gaps[unseen & present] = reverse_index

    blended = (
        0.48 * _minmax(recent_counts)
        + 0.32 * _minmax(long_counts)
        + 0.20 * _minmax(gaps)
    )
    # A floor prevents visualized trends from excluding any valid number.
    weights = 0.25 + blended
    return weights / weights.sum()


def _default_seed(draws: np.ndarray) -> int:
    digest = hashlib.sha256(draws[-20:].tobytes()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _is_balanced(numbers: np.ndarray, sum_bounds: tuple[float, float]) -> bool:
    odd_count = int(np.sum(numbers % 2 == 1))
    low_count = int(np.sum(numbers <= 22))
    total = int(numbers.sum())
    consecutive_pairs = int(np.sum(np.diff(numbers) == 1))
    return (
        2 <= odd_count <= 4
        and 2 <= low_count <= 4
        and sum_bounds[0] <= total <= sum_bounds[1]
        and consecutive_pairs <= 2
    )


def generate_recommendations(
    data: Iterable[Iterable[int]],
    count: int = 5,
    lookback: int = 80,
    seed: int | None = None,
) -> list[list[int]]:
    """Generate valid, varied and reproducible weighted combinations."""
    draws = _as_draw_matrix(data)
    count = max(1, min(int(count), 10))
    weights = calculate_number_scores(draws, lookback)
    rng = np.random.default_rng(_default_seed(draws) if seed is None else seed)
    draw_sums = draws.sum(axis=1)
    sum_bounds = (float(np.quantile(draw_sums, 0.08)), float(np.quantile(draw_sums, 0.92)))

    recommendations: list[list[int]] = []
    attempts = 0
    while len(recommendations) < count and attempts < count * 500:
        attempts += 1
        candidate = np.sort(rng.choice(np.arange(1, 46), size=6, replace=False, p=weights))
        candidate_list = candidate.tolist()
        if not _is_balanced(candidate, sum_bounds):
            continue
        if candidate_list in recommendations:
            continue
        if any(len(set(candidate_list) & set(existing)) > 4 for existing in recommendations):
            continue
        recommendations.append(candidate_list)

    # Unusual tiny datasets may reject many samples. Always honor the count
    # while retaining the hard number validity constraints.
    while len(recommendations) < count:
        candidate = sorted(rng.choice(np.arange(1, 46), size=6, replace=False).tolist())
        if candidate not in recommendations:
            recommendations.append(candidate)
    return recommendations


def describe_trends(
    data: Iterable[Iterable[int]], lookback: int = 80
) -> tuple[list[int], list[int]]:
    scores = calculate_number_scores(data, lookback)
    ranked = np.argsort(scores) + 1
    return ranked[-6:][::-1].tolist(), ranked[:6].tolist()


def predict_next_numbers(
    data: Iterable[Iterable[int]] | None = None,
    data_path: str | Path = "lotto_data.csv",
) -> np.ndarray:
    """Backward-compatible single-set API used by older callers."""
    if data is None:
        import csv

        with Path(data_path).open(encoding="utf-8") as source:
            rows = csv.DictReader(source)
            data = [ast.literal_eval(row["numbers"]) for row in rows]
    return np.asarray(generate_recommendations(data, count=1)[0], dtype=int)


def initialize_or_update_model(data: Iterable[Iterable[int]]) -> None:
    """Backward-compatible no-op: recommendations no longer retrain on clicks."""
    _as_draw_matrix(data)
