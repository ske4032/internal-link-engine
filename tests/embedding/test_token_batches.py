from __future__ import annotations

import itertools
import math
import random

import pytest

from linking_engine.embedding.voyage_client import token_batches


def check_partition(counts: list[int], ranges: list[range], budget: int, max_items: int) -> None:
    covered = [i for r in ranges for i in r]
    assert covered == list(range(len(counts))), (
        "ranges must be contiguous and cover every index once"
    )
    for r in ranges:
        assert r.step == 1, r
        assert len(r) >= 1, f"empty batch {r}"
        assert len(r) <= max_items, f"{r} holds {len(r)} items, cap {max_items}"
        total = sum(counts[i] for i in r)
        assert total <= budget, f"{r} holds {total} tokens, budget {budget}"


def test_random_counts_always_respect_budget_and_item_cap() -> None:
    rng = random.Random(42)
    for _ in range(300):
        budget = rng.randint(1, 500)
        max_items = rng.randint(1, 12)
        counts = [rng.randint(0, budget) for _ in range(rng.randint(1, 80))]
        ranges = list(token_batches(counts, budget=budget, max_items=max_items))
        check_partition(counts, ranges, budget, max_items)


def test_consecutive_batches_could_not_have_been_merged() -> None:
    # Packing is greedy: each batch closes only because the next item would not fit.
    rng = random.Random(7)
    for _ in range(200):
        budget, max_items = 300, 5
        counts = [rng.randint(1, budget) for _ in range(rng.randint(2, 60))]
        ranges = list(token_batches(counts, budget=budget, max_items=max_items))
        for current, following in itertools.pairwise(ranges):
            grown = sum(counts[i] for i in current) + counts[following.start]
            assert grown > budget or len(current) == max_items, (
                f"{current} closed early: next item fits under budget {budget} and cap {max_items}"
            )


def test_one_count_over_budget_raises() -> None:
    with pytest.raises(ValueError):
        list(token_batches([10, 121, 10], budget=120, max_items=4))


def test_empty_input_yields_nothing() -> None:
    assert list(token_batches([], budget=120, max_items=4)) == []


def test_exact_budget_fits_one_batch() -> None:
    assert list(token_batches([60, 60], budget=120, max_items=4)) == [range(0, 2)]
    assert list(token_batches([120], budget=120, max_items=4)) == [range(0, 1)]


def test_one_token_over_budget_splits() -> None:
    assert list(token_batches([60, 61], budget=120, max_items=4)) == [range(0, 1), range(1, 2)]


def test_item_cap_splits_even_when_tokens_fit() -> None:
    ranges = list(token_batches([1] * 9, budget=120, max_items=4))
    check_partition([1] * 9, ranges, 120, 4)
    assert len(ranges) == 3, f"9 one-token items at cap 4 need 3 batches, got {ranges}"


def test_5000_and_200_token_pages_share_one_batch() -> None:
    assert list(token_batches([5_000, 200], budget=110_000, max_items=1_000)) == [range(0, 2)]


def test_realistic_pages_fill_batches_to_the_budget() -> None:
    counts = [3_300] * 40
    ranges = list(token_batches(counts, budget=110_000, max_items=1_000))
    check_partition(counts, ranges, 110_000, 1_000)
    per_batch = 110_000 // 3_300
    assert len(ranges) == math.ceil(40 / per_batch), (
        f"expected 2 batches of <=33 pages, got {ranges}"
    )


@pytest.mark.parametrize(("budget", "max_items"), [(0, 4), (120, 0)])
def test_non_positive_limits_raise(budget: int, max_items: int) -> None:
    with pytest.raises(ValueError):
        list(token_batches([1], budget=budget, max_items=max_items))
