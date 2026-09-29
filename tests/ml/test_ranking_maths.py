"""#24-#27 maths: the source split, query groups, NDCG@10, bootstrap intervals, LambdaMART on
the planted graded pairs, chunked prediction, gain importance, the promotion gate and #24's
planted labels. Pure functions, no stores."""

from __future__ import annotations

import math
import tracemalloc
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas
import pytest
import ranking_factories as make
from ranking_seed import (
    NOISE_URLS,
    ORPHAN_INDEX,
    URLS,
    graded_frame,
    noise_url,
    page_url,
    planted_grades,
)

from linking_engine.discovery.features import FEATURE_COLUMNS
from linking_engine.ml.quality import LINK_DERIVED_COLUMNS
from linking_engine.ml.ranking import (
    DOMINANT_SHARE,
    EXCL_PLACEMENT_COLUMNS,
    EXCLUDED_COLUMNS,
    GROUP_COLUMNS,
    LIKE_FOR_LIKE_COLUMNS,
    MODEL_COLUMNS,
    PLACEMENT_COLUMNS,
    PREDICT_CHUNK,
    PRODUCT_COLUMNS,
    Trained,
    bootstrap_ci,
    dominant_feature,
    gini,
    group_ndcg,
    importance,
    ndcg_at,
    params_for,
    placement_gain_share,
    placement_shares,
    positive_groups,
    precision_at,
    predict,
    product_measures,
    promotion,
    ranking_metrics,
    seed_result,
    seed_split,
    split_sources,
    summarise_ranker,
    train,
    with_groups,
    without_orphan_targets,
)
from linking_engine.models import HeldOutSettings, RankerParams, ScorerName

if TYPE_CHECKING:
    from collections.abc import Iterator

# The columns the planted frame fills from its truth.
PLANTED_COLUMNS = (
    "content_cosine",
    "same_hub",
    "target_is_orphan",
    "target_hub_size",
    "target_inbound_count",
)
# #24's grades are not relevance-monotone by design: the bridge gap outranks the same topic.
PARAMS = RankerParams(max_rounds=300, early_stopping_rounds=30, monotone_increasing=())
RELEVANCE = ("content_cosine", "anchor_target_fit", "context_relevance")
HUMP_COLUMNS = (*RELEVANCE, "same_hub")
LOG3 = math.log2(3)


def split(frame: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    valid = split_sources(frame["source_url"], share=0.3, seed=7)
    held = frame["source_url"].isin(valid)
    return frame[~held], frame[held]


@pytest.fixture(scope="module")
def planted() -> pandas.DataFrame:
    return graded_frame()


@pytest.fixture(scope="module")
def trained(planted: pandas.DataFrame) -> Trained:
    return train(*split(planted), PLANTED_COLUMNS, PARAMS)


# ── columns ─────────────────────────────────────────────────────────────────


def test_model_columns_drop_the_leaking_crawl_depth_and_like_for_like_the_link_counts() -> None:
    assert dict(EXCLUDED_COLUMNS) == {
        "target_crawl_depth": "stored BFS over links that still include the hidden ones"
    }
    assert tuple(c for c in FEATURE_COLUMNS if c != "target_crawl_depth") == MODEL_COLUMNS
    assert tuple(c for c in MODEL_COLUMNS if c not in LINK_DERIVED_COLUMNS) == LIKE_FOR_LIKE_COLUMNS
    assert "content_cosine" in LIKE_FOR_LIKE_COLUMNS
    assert "target_inbound_count" not in LIKE_FOR_LIKE_COLUMNS
    assert GROUP_COLUMNS == ("round", "source_url")
    assert PLACEMENT_COLUMNS == ("context_relevance", "anchor_target_fit")
    assert tuple(c for c in MODEL_COLUMNS if c not in PLACEMENT_COLUMNS) == EXCL_PLACEMENT_COLUMNS


def test_placement_shares_by_label_show_the_anchors_the_protocol_hands_positives() -> None:
    nan = math.nan
    frame = pandas.DataFrame(
        {
            "label": [1, 1, 1, 0, 0, 0, 0],
            "context_relevance": [0.7, nan, nan, 0.6, nan, nan, nan],
            "anchor_target_fit": [nan, 0.8, nan, nan, nan, nan, nan],
        }
    )

    assert placement_shares(frame) == pytest.approx((2 / 3, 1 / 4))
    assert placement_shares(frame[frame["label"] == 0]) == (None, 0.25)
    assert placement_shares(frame[frame["label"] == 1]) == (pytest.approx(2 / 3), None)
    entries = make.importance(context_relevance=0.5, content_cosine=0.2, same_hub=0.2)
    assert placement_gain_share(entries) == pytest.approx(0.5)
    assert placement_gain_share(make.importance()) == 0.0


# ── split and groups ────────────────────────────────────────────────────────


def test_split_sources_stable_disjoint_and_reproducible() -> None:
    sources = [f"example.com/p{i}" for i in range(2000)]

    test = split_sources(sources, share=0.2, seed=7)
    rest = [s for s in sources if s not in test]
    valid = split_sources(rest, share=0.1, seed=8)

    assert test <= set(sources)
    assert not test & valid, "a source page on both sides"
    # Binomial(2000, 0.2): mean 400, sd 18.
    assert 330 <= len(test) <= 470
    assert split_sources(sources[::-1] + sources[:50], share=0.2, seed=7) == test
    grown = [*sources, *(f"example.com/new{i}" for i in range(300))]
    assert split_sources(grown, share=0.2, seed=7) & set(sources) == test, (
        "a source changed side when others were added"
    )
    assert split_sources(sources, share=0.2, seed=8) != test, "the seed does not drive the split"
    assert split_sources(sources, share=0.3, seed=7) > test, "a larger share only adds sources"


@pytest.mark.parametrize("share", [0.0, 1.0, -0.2])
def test_split_share_must_be_strictly_between_zero_and_one(share: float) -> None:
    with pytest.raises(ValueError, match="share"):
        split_sources(["a"], share=share, seed=7)


def test_groups_sorted_sizes_match_rows() -> None:
    rows = [
        (1, "s2", "t1", 0),
        (0, "s2", "t2", 1),
        (0, "s1", "t3", 0),
        (1, "s1", "t1", 0),
        (0, "s1", "t1", 1),
        (0, "s2", "t1", 0),
        (1, "s2", "t0", 0),
    ]
    frame = pandas.DataFrame(rows, columns=["round", "source_url", "target_url", "label"])

    ordered, sizes = with_groups(frame)

    keys = list(zip(ordered["round"], ordered["source_url"], ordered["target_url"], strict=True))
    assert keys == sorted(keys)
    assert sizes.tolist() == [2, 2, 1, 2]
    assert sizes.sum() == len(frame)
    kept = positive_groups(ordered)
    assert list(zip(kept["round"], kept["source_url"], strict=True)) == [
        (0, "s1"),
        (0, "s1"),
        (0, "s2"),
        (0, "s2"),
    ]
    assert with_groups(frame.iloc[:0])[1].tolist() == []
    with pytest.raises(ValueError, match="missing group columns"):
        with_groups(frame.drop(columns="round"))


# ── NDCG, precision and intervals ───────────────────────────────────────────


def test_ndcg_known_values_ties_and_single_item() -> None:
    assert ndcg_at([3, 2, 1, 0], [4, 3, 2, 1], 10) == 1.0
    assert ndcg_at([0, 1], [1, 0], 10) == pytest.approx(1 / LOG3)
    assert ndcg_at([1, 3], [1, 0], 10) == pytest.approx((1 + 7 / LOG3) / (7 + 1 / LOG3)), (
        "gain is 2**label - 1"
    )
    # Ties keep the input order; a missing score ranks last.
    assert ndcg_at([0, 1], [0.5, 0.5], 10) == pytest.approx(1 / LOG3)
    assert ndcg_at([1, 0], [0.5, 0.5], 10) == 1.0
    assert ndcg_at([1, 0], [math.nan, 0.1], 10) == pytest.approx(1 / LOG3)
    # Only the first k count.
    assert ndcg_at([0, 0, 1], [3, 2, 1], 2) == 0.0
    assert ndcg_at([0, 0, 1], [3, 2, 1], 3) == pytest.approx(0.5)
    assert ndcg_at([2], [0.3], 10) == 1.0
    assert ndcg_at([0], [0.3], 10) == 0.0, "no relevant item scores 0, not LightGBM's 1"
    assert ndcg_at([], [], 10) == 0.0
    with pytest.raises(ValueError, match="one length"):
        ndcg_at([1, 0], [1.0], 10)
    with pytest.raises(ValueError, match="k must be at least 1"):
        ndcg_at([1, 0], [1, 0], 0)


def metric_frame() -> pandas.DataFrame:
    """Three groups over two rounds: perfect, the positive second, and a tie kept in target
    order."""
    rows = [
        (0, "s1", "a", 1, 0.9),
        (0, "s1", "b", 0, 0.1),
        (0, "s2", "a", 0, 0.9),
        (0, "s2", "b", 1, 0.1),
        (1, "s1", "b", 0, 0.5),
        (1, "s1", "a", 1, 0.5),
    ]
    return pandas.DataFrame(rows, columns=["round", "source_url", "target_url", "label", "score"])


def test_group_ndcg_precision_and_metrics_of_a_scorer() -> None:
    frame = metric_frame()

    assert group_ndcg(frame, "score", 10).tolist() == pytest.approx([1.0, 1 / LOG3, 1.0])
    assert precision_at(frame, "score", 5) == pytest.approx(0.2), "hits over k, not over pairs"
    assert precision_at(frame, "score", 1) == pytest.approx(2 / 3)

    found = ranking_metrics(frame, "score", ScorerName.BASELINE, all_groups=4, seed=1)

    assert found.scorer is ScorerName.BASELINE
    assert found.ndcg_at_10 == pytest.approx((2 + 1 / LOG3) / 3)
    assert found.ci_low <= found.ndcg_at_10 <= found.ci_high
    assert (found.groups, found.groups_with_positive_share, found.n_labelled_pairs) == (3, 0.75, 3)
    assert found.per_round == pytest.approx({0: (1 + 1 / LOG3) / 2, 1: 1.0})
    assert found.histogram == (0, 0, 0, 0, 0, 0, 1, 0, 0, 2)
    assert found.precision_at_5 == pytest.approx(0.2)


def test_metrics_refuse_groups_without_a_positive_and_too_few_groups_in_all() -> None:
    frame = metric_frame()
    blank = frame.assign(label=[1, 0, 0, 0, 0, 1])

    with pytest.raises(ValueError, match="without a positive"):
        ranking_metrics(blank, "score", ScorerName.LEARNED, all_groups=3, seed=1)
    with pytest.raises(ValueError, match="all_groups"):
        ranking_metrics(frame, "score", ScorerName.LEARNED, all_groups=2, seed=1)
    with pytest.raises(ValueError, match="no groups"):
        ranking_metrics(frame.iloc[:0], "score", ScorerName.LEARNED, all_groups=1, seed=1)


def test_bootstrap_ci_deterministic_by_seed() -> None:
    values = np.random.default_rng(3).normal(0.5, 0.1, 200)

    first = bootstrap_ci(values, seed=11)

    assert bootstrap_ci(values, seed=11) == first
    assert bootstrap_ci(values, seed=12) != first, "the seed does not drive the resampling"
    low, high = first
    assert low < values.mean() < high
    # The standard error is 0.1 / sqrt(200) = 0.007, so the 95% interval is about 0.028 wide.
    assert 0.02 < high - low < 0.036
    wide = bootstrap_ci(values, seed=11, level=0.99)
    assert wide[0] < low
    assert wide[1] > high
    assert bootstrap_ci([0.4] * 30, seed=1) == (0.4, 0.4)
    assert bootstrap_ci([0.7], seed=1) == (0.7, 0.7)
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([], seed=1)
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci(values, seed=1, level=1.0)


# ── training and prediction ─────────────────────────────────────────────────


def test_train_lambdarank_improves_over_first_iteration(
    planted: pandas.DataFrame, trained: Trained
) -> None:
    _, valid = split(planted)
    ordered, _ = with_groups(valid)
    matrix = ordered.loc[:, list(PLANTED_COLUMNS)].to_numpy(dtype=np.float32)

    first = ordered.assign(score=trained.booster.predict(matrix, num_iteration=1))
    best = ordered.assign(score=trained.booster.predict(matrix))
    first_ndcg = float(group_ndcg(first, "score", 10).mean())
    best_ndcg = float(group_ndcg(best, "score", 10).mean())

    assert trained.columns == PLANTED_COLUMNS
    assert trained.best_iteration > 1
    assert trained.booster.num_trees() == trained.best_iteration, "the model is cut to its best"
    assert best_ndcg > first_ndcg + 0.1, (
        f"validation NDCG@10 at the best iteration {best_ndcg:.3f} is not 0.1 above the "
        f"first iteration's {first_ndcg:.3f}"
    )
    shuffled = ordered.assign(score=np.random.default_rng(0).random(len(ordered)))
    assert best_ndcg > float(group_ndcg(shuffled, "score", 10).mean()) + 0.2


def test_training_is_deterministic(planted: pandas.DataFrame, trained: Trained) -> None:
    again = train(*split(planted), PLANTED_COLUMNS, PARAMS)

    assert again.best_iteration == trained.best_iteration
    assert again.booster.model_to_string() == trained.booster.model_to_string()


def test_training_refuses_bad_columns_and_empty_groups(planted: pandas.DataFrame) -> None:
    train_frame, valid = split(planted)

    with pytest.raises(ValueError, match="distinct"):
        train(train_frame, valid, ("same_hub", "same_hub"), PARAMS)
    with pytest.raises(ValueError, match="distinct"):
        train(train_frame, valid, (), PARAMS)
    with pytest.raises(ValueError, match="no query groups"):
        train(train_frame, valid.iloc[:0], PLANTED_COLUMNS, PARAMS)
    with pytest.raises(ValueError, match="missing feature columns"):
        train(train_frame, valid.drop(columns="same_hub"), PLANTED_COLUMNS, PARAMS)


def test_predict_chunked_matches_full_and_rejects_missing_column(
    planted: pandas.DataFrame, trained: Trained
) -> None:
    matrix = planted.loc[:, list(PLANTED_COLUMNS)].to_numpy(dtype=np.float32)
    full = trained.booster.predict(matrix)
    # Chunks carry extra columns in another order; the model reads its own by name.
    shuffled = planted.loc[:, ["label", *reversed(PLANTED_COLUMNS), "source_url"]]
    chunks = [shuffled.iloc[i : i + 1000] for i in range(0, len(planted), 1000)]

    parts = list(predict(trained, PLANTED_COLUMNS, [*chunks, shuffled.iloc[:0]]))

    assert [len(part) for part in parts] == [len(chunk) for chunk in chunks] + [0]
    assert np.concatenate(parts) == pytest.approx(full)
    assert all(part.dtype == np.float64 for part in parts)
    by_booster = np.concatenate(list(predict(trained.booster, PLANTED_COLUMNS, chunks)))
    assert by_booster == pytest.approx(full)

    with pytest.raises(ValueError, match="same_hub"):
        list(predict(trained, PLANTED_COLUMNS, [chunks[0], chunks[1].drop(columns="same_hub")]))
    with pytest.raises(ValueError, match="columns differ"):
        predict(trained, tuple(reversed(PLANTED_COLUMNS)), chunks)


def test_predict_1_25m_rows_bounded_memory(planted: pandas.DataFrame) -> None:
    """1.25M pairs at the width of the model, generated a chunk at a time and never as one
    frame. The chunks are float64, so each is converted to the float32 matrix with a copy: the
    peak stays within a few chunks, far below the float32 matrix of all rows, and a chunk is
    scored before the next is read."""
    model = train(*split(planted), MODEL_COLUMNS, RankerParams(max_rounds=40))
    rows, width = 1_250_000, len(MODEL_COLUMNS)
    chunk_bytes = PREDICT_CHUNK * width * 8
    full_matrix = rows * width * 4
    assert 4 * chunk_bytes < full_matrix / 3, "the bound would not catch a concatenation"
    pulled = 0

    def chunks() -> Iterator[pandas.DataFrame]:
        nonlocal pulled
        rng = np.random.default_rng(1)
        for _ in range(rows // PREDICT_CHUNK):
            pulled += 1
            yield pandas.DataFrame(
                rng.random((PREDICT_CHUNK, width)), columns=list(MODEL_COLUMNS), copy=False
            )

    tracemalloc.start()
    try:
        scored = predict(model, MODEL_COLUMNS, chunks())
        first = next(scored)
        assert pulled == 1, "chunks are read ahead of scoring"
        total, count = float(first.sum()), len(first)
        for part in scored:
            total += float(part.sum())
            count += len(part)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert count == rows
    assert math.isfinite(total)
    assert peak < 4 * chunk_bytes, (
        f"peak {peak / 1e6:.0f} MB above four chunks ({4 * chunk_bytes / 1e6:.0f} MB); the "
        f"float32 matrix of all rows is {full_matrix / 1e6:.0f} MB"
    )


# ── relevance-monotone constraints ──────────────────────────────────────────


def humped_frame(sources: int = 60, targets: int = 40) -> pandas.DataFrame:
    """Groups whose grades peak at mid values of each relevance column, like near-duplicates
    that are never linked: an unconstrained model learns to lower the score at the top."""
    rng = np.random.default_rng(94)
    rows = sources * targets
    frame = pandas.DataFrame(
        {
            "round": np.zeros(rows, dtype=np.int16),
            "source_url": [f"example.com/s{i // targets}" for i in range(rows)],
            "target_url": [f"example.com/t{i % targets}" for i in range(rows)],
            **{column: rng.random(rows) for column in RELEVANCE},
            "same_hub": rng.integers(0, 2, rows).astype(np.float64),
        }
    )
    hump = sum(4 * frame[c] * (1 - frame[c]) for c in RELEVANCE) + frame["same_hub"]
    frame["hump"] = hump + rng.normal(0, 0.2, rows)
    rank = frame.groupby("source_url")["hump"].rank(ascending=False, method="first")
    frame["label"] = np.select([rank <= 3, rank <= 8], [2, 1], 0)
    return frame.drop(columns="hump")


def largest_drop_and_rise(
    model: Trained, frame: pandas.DataFrame, column: str
) -> tuple[float, float]:
    """Each of 50 pairs scored with ``column`` swept over [0, 1], the rest fixed: the largest fall
    and the largest rise of a score from one step to the next."""
    grid = np.linspace(0.0, 1.0, 41)
    base = frame.iloc[:50].loc[:, list(model.columns)]
    swept = base.loc[base.index.repeat(len(grid))].assign(**{column: np.tile(grid, len(base))})
    scores = model.booster.predict(swept.to_numpy(dtype=np.float64)).reshape(len(base), len(grid))
    steps = np.diff(scores, axis=1)
    return float(-steps.min()), float(steps.max())


@pytest.fixture(scope="module")
def humped() -> tuple[pandas.DataFrame, pandas.DataFrame]:
    frame = humped_frame()
    valid = frame["source_url"].isin(split_sources(frame["source_url"], share=0.3, seed=7))
    return frame[~valid], frame[valid]


def test_monotone_columns_only_raise_a_score(
    humped: tuple[pandas.DataFrame, pandas.DataFrame],
) -> None:
    constrained = train(*humped, HUMP_COLUMNS, RankerParams(max_rounds=200))
    free = train(*humped, HUMP_COLUMNS, RankerParams(max_rounds=200, monotone_increasing=()))

    assert RankerParams().monotone_increasing == RELEVANCE
    free_drops = [largest_drop_and_rise(free, humped[1], c)[0] for c in RELEVANCE]
    assert max(free_drops) > 1e-3, "the fixture never makes a free model lower a score"
    for column in RELEVANCE:
        drop, rise = largest_drop_and_rise(constrained, humped[1], column)
        assert drop <= 1e-12, f"raising {column} lowered a score by {drop:.3g}"
        assert rise > 0, f"the constrained model ignores {column}"


def test_empty_monotone_turns_constraints_off(
    humped: tuple[pandas.DataFrame, pandas.DataFrame],
) -> None:
    constrained = train(*humped, HUMP_COLUMNS, RankerParams(max_rounds=20)).booster
    free = train(*humped, HUMP_COLUMNS, RankerParams(max_rounds=20, monotone_increasing=()))

    settings = constrained.model_to_string()
    assert "[monotone_constraints: 1,1,1,0]" in settings
    assert "[monotone_constraints_method: advanced]" in settings
    unset = free.booster.model_to_string()
    assert "[monotone_constraints: ]" in unset
    assert "monotone_constraints=" not in unset, "a constraint reached the trees"
    only_cosine = train(
        *humped, HUMP_COLUMNS, RankerParams(max_rounds=20, monotone_increasing=("content_cosine",))
    )
    assert "[monotone_constraints: 1,0,0,0]" in only_cosine.booster.model_to_string()


def test_unknown_monotone_column_raises(
    humped: tuple[pandas.DataFrame, pandas.DataFrame],
) -> None:
    with pytest.raises(ValueError, match="anchor_target_fit"):
        train(*humped, ("content_cosine", "same_hub"), RankerParams(max_rounds=5))
    with pytest.raises(ValueError, match="no_such_column"):
        train(
            *humped,
            HUMP_COLUMNS,
            RankerParams(max_rounds=5, monotone_increasing=("content_cosine", "no_such_column")),
        )


def test_reduced_models_keep_the_constraints_on_the_columns_they_have() -> None:
    reduced = params_for(RankerParams(), EXCL_PLACEMENT_COLUMNS)

    assert reduced.monotone_increasing == ("content_cosine",)
    assert params_for(RankerParams(), MODEL_COLUMNS).monotone_increasing == RELEVANCE
    assert params_for(RankerParams(monotone_increasing=()), MODEL_COLUMNS).monotone_increasing == ()
    with pytest.raises(ValueError, match="no_such_column"):
        params_for(RankerParams(monotone_increasing=("no_such_column",)), MODEL_COLUMNS)


# ── orphan targets, split seeds and seed results ────────────────────────────


def test_orphan_target_rows_are_left_out_and_nothing_else() -> None:
    frame = metric_frame().assign(target_url=["a", "b", "o", "b", "b", "a"])

    kept = without_orphan_targets(frame, {"o", "never-a-target"})

    assert kept["target_url"].tolist() == ["a", "b", "b", "b", "a"]
    assert without_orphan_targets(frame, set()).equals(frame)


def test_the_split_seed_gives_the_fixed_split_and_other_seeds_redraw_it() -> None:
    sources = [f"example.com/p{i}" for i in range(400)]
    settings = HeldOutSettings()

    test, valid = seed_split(sources, settings, settings.split_seed)

    assert test == split_sources(sources, share=0.2, seed=7)
    assert valid == split_sources(set(sources) - test, share=0.1, seed=8)
    for seed in settings.evaluation_seeds:
        other_test, other_valid = seed_split(sources, settings, seed)
        assert not other_test & other_valid, f"seed {seed}: a source on both sides"
        if seed != settings.split_seed:
            assert other_test != test, f"seed {seed} does not redraw the test sources"


def test_a_seed_result_pairs_learned_with_plain_over_the_seeds_test_groups() -> None:
    frame = metric_frame().assign(plain=[0.1, 0.9, 0.9, 0.1, 0.4, 0.6], baseline=0.5)

    found = seed_result(
        frame, seed=11, learned="score", plain="plain", baseline="baseline", bootstrap_seed=3
    )

    learned = [1.0, 1 / LOG3, 1.0]
    plain = [1 / LOG3, 1.0, 1 / LOG3]
    assert (found.seed, found.test_pages) == (11, 2)
    assert found.learned[0] == pytest.approx(np.mean(learned))
    assert found.plain[0] == pytest.approx(np.mean(plain))
    # Ties keep target order: "a" first, the positive in two of three groups.
    assert found.baseline[0] == pytest.approx((1 + 1 / LOG3 + 1) / 3)
    assert found.delta == pytest.approx(np.mean(np.subtract(learned, plain)))
    assert found.delta_ci_low <= found.delta <= found.delta_ci_high
    for triple in (found.learned, found.plain, found.baseline):
        assert triple[1] <= triple[0] <= triple[2]
    again = seed_result(
        frame, seed=11, learned="score", plain="plain", baseline="baseline", bootstrap_seed=3
    )
    assert again == found
    with pytest.raises(ValueError, match="without a positive"):
        seed_result(
            frame.assign(label=0),
            seed=11,
            learned="score",
            plain="plain",
            baseline="baseline",
            bootstrap_seed=3,
        )


# ── product measures ────────────────────────────────────────────────────────


def product_frame() -> pandas.DataFrame:
    """Three source pages. s2 ties "o2" with "a" (target order puts "a" first); s3 has a pair
    without a score (last) and an orphan without a cosine or a hub."""
    nan = math.nan
    rows = [
        ("s1", "a", 0.9, 0.8, 1.0),
        ("s1", "b", 0.5, 0.4, 0.0),
        ("s1", "o1", 0.7, 0.6, 0.0),
        ("s2", "c", 0.8, 0.9, 1.0),
        ("s2", "o2", 0.3, nan, 0.0),
        ("s2", "a", 0.3, 0.2, 1.0),
        ("s3", "b", nan, 0.3, 0.0),
        ("s3", "o1", 0.2, nan, nan),
    ]
    return pandas.DataFrame(
        rows, columns=["source_url", "target_url", "score", "content_cosine", "same_hub"]
    )


def test_product_measures_known_values() -> None:
    orphans = {"o1", "o2", "never-a-candidate"}

    two = product_measures(product_frame(), "score", ScorerName.LEARNED, orphans, pillars={}, k=2)
    one = product_measures(product_frame(), "score", ScorerName.PLAIN, orphans, pillars={}, k=1)

    # First two: s1 a, o1; s2 c, a; s3 o1, b.
    assert (two.scorer, two.k) == (ScorerName.LEARNED, 2)
    assert two.top_relevance == pytest.approx((0.8 + 0.6 + 0.9 + 0.2 + 0.3) / 5)
    assert two.same_hub_share == pytest.approx(3 / 6)
    assert two.orphan_slot_share == pytest.approx(2 / 6)
    assert two.orphan_page_share == pytest.approx(2 / 5), "orphans among the candidate targets"
    assert two.orphans_reached == pytest.approx(1 / 2), "o2 is never shown"
    # First one: s1 a, s2 c, s3 o1.
    assert one.top_relevance == pytest.approx((0.8 + 0.9) / 2)
    assert one.same_hub_share == pytest.approx(2 / 3)
    assert one.orphan_slot_share == pytest.approx(1 / 3)
    assert (
        product_measures(product_frame(), "score", ScorerName.LEARNED, orphans, pillars={}).k == 10
    )
    # A second frame, worked by hand independently: the tie at 2.0 goes to "o1", which sorts
    # before "t2"; a pair without a score comes last.
    nan = math.nan
    rows = [
        ("s1", "t1", 3.0, 0.9, 1.0),
        ("s1", "t2", 2.0, 0.5, 0.0),
        ("s1", "o1", 2.0, 0.4, 1.0),
        ("s1", "t3", 1.0, 0.1, nan),
        ("s2", "t1", nan, 0.8, 1.0),
        ("s2", "o1", 5.0, nan, 0.0),
        ("s2", "o2", 4.0, 0.2, 0.0),
    ]
    frame = pandas.DataFrame(rows, columns=["source_url", "target_url", "s", *PRODUCT_COLUMNS])
    found = product_measures(frame, "s", ScorerName.LEARNED, {"o1", "o2", "zz"}, pillars={}, k=2)
    assert (found.top_relevance, found.same_hub_share, found.orphan_slot_share) == (
        pytest.approx(0.5),
        0.5,
        0.75,
    )
    assert (found.orphan_page_share, found.orphans_reached) == (pytest.approx(0.4), 1.0)
    none = product_measures(product_frame(), "score", ScorerName.BASELINE, set(), pillars={}, k=2)
    assert (none.orphan_slot_share, none.orphan_page_share, none.orphans_reached) == (0, 0, None)


def test_orphan_sources_in_a_hub_are_measured_against_their_pillar() -> None:
    # First two: s1 a, o1; s2 c, a; s3 o1, b. s2 and s3 are orphans in a hub, s1 is not an
    # orphan, and s4 is an orphan in a hub that is no source.
    orphans = {"o1", "o2", "s2", "s3", "s4"}
    pillars = {"s1": "a", "s2": "o2", "s3": "o1", "s4": "a"}

    found = product_measures(
        product_frame(), "score", ScorerName.LEARNED, orphans, pillars=pillars, k=2
    )
    wider = product_measures(
        product_frame(), "score", ScorerName.LEARNED, orphans, pillars=pillars, k=3
    )
    hubless = product_measures(
        product_frame(), "score", ScorerName.LEARNED, orphans, pillars={"s1": "a"}, k=2
    )

    assert found.orphans_to_pillar == pytest.approx(1 / 2), "s3 shows o1, s2 misses o2"
    assert wider.orphans_to_pillar == 1.0, "o2 is s2's third pair"
    assert hubless.orphans_to_pillar is None
    assert (found.orphan_page_share, found.orphans_reached) == (pytest.approx(2 / 5), 0.5)


def test_the_inbound_gini_is_over_every_candidate_target() -> None:
    two = product_measures(product_frame(), "score", ScorerName.LEARNED, set(), pillars={}, k=2)
    one = product_measures(product_frame(), "score", ScorerName.LEARNED, set(), pillars={}, k=1)

    # First two into a, b, c, o1, o2: 2, 1, 1, 2, 0.
    assert two.inbound_gini == pytest.approx(1 / 3)
    # First one into a, b, c, o1, o2: 1, 0, 1, 1, 0.
    assert one.inbound_gini == pytest.approx(2 / 5)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([3, 3, 3], 0.0),
        ([0, 0, 0, 4], 0.75),
        ([5], 0.0),
        ([2, 1, 1, 2, 0], 1 / 3),
        ([], None),
        ([0, 0], None),
    ],
)
def test_gini_known_values(counts: list[int], expected: float | None) -> None:
    assert gini(counts) == (None if expected is None else pytest.approx(expected))


def test_gini_refuses_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gini([1, -1])


def test_product_measures_refuse_what_they_cannot_measure() -> None:
    with pytest.raises(ValueError, match="same_hub"):
        product_measures(
            product_frame().drop(columns="same_hub"),
            "score",
            ScorerName.LEARNED,
            set(),
            pillars={},
        )
    with pytest.raises(ValueError, match="no candidate pairs"):
        product_measures(product_frame().iloc[:0], "score", ScorerName.LEARNED, set(), pillars={})
    with pytest.raises(ValueError, match="k must be at least 1"):
        product_measures(product_frame(), "score", ScorerName.LEARNED, set(), pillars={}, k=0)


# ── importance and promotion ────────────────────────────────────────────────


def test_importance_gain_shares_sum_to_one_and_dominant_flag(planted: pandas.DataFrame) -> None:
    """A column that is the label itself takes nearly all the gain; the planted model's gain is
    spread."""
    leaky = planted.assign(leak=planted["label"].astype(float))
    columns = ("leak", *PLANTED_COLUMNS)

    entries = importance(train(*split(leaky), columns, PARAMS))

    assert {entry.column for entry in entries} == set(columns)
    assert [entry.gain for entry in entries] == sorted((e.gain for e in entries), reverse=True)
    assert sum(entry.gain_share for entry in entries) == pytest.approx(1.0)
    assert entries[0].column == "leak"
    assert entries[0].gain_share > DOMINANT_SHARE
    assert dominant_feature(entries) == "leak"


def test_no_column_dominates_at_or_below_the_share(trained: Trained) -> None:
    entries = importance(trained)

    assert sum(entry.gain_share for entry in entries) == pytest.approx(1.0)
    assert dominant_feature(entries) is None
    assert dominant_feature(make.importance(content_cosine=DOMINANT_SHARE, same_hub=0.3)) is None


def test_promotion_requires_ci_above_zero_and_allowed() -> None:
    rival = np.random.default_rng(5).uniform(0.2, 0.8, 60)
    sources = [f"example.com/p{i}" for i in range(60)]
    kwargs = {
        "sources": sources,
        "rival_name": ScorerName.BASELINE,
        "holder_version": None,
        "seed": 3,
    }

    tie = promotion(rival, rival, allowed=True, **kwargs)
    better = promotion(rival + 0.1, rival, allowed=False, **kwargs)
    allowed = promotion(rival + 0.1, rival, allowed=True, **kwargs)
    worse = promotion(rival - 0.1, rival, allowed=True, **kwargs)
    noisy = rival + np.random.default_rng(6).normal(0.01, 0.3, 60).clip(-rival, 1 - rival)
    unsure = promotion(noisy, rival, allowed=True, **kwargs)

    assert (tie.delta, tie.delta_ci_low, tie.would_promote, tie.promoted) == (0, 0, False, False)
    assert better.delta == pytest.approx(0.1)
    assert (better.would_promote, better.promoted) == (True, False)
    assert "not allowed" in better.reason
    assert (allowed.would_promote, allowed.promoted) == (True, True)
    assert (worse.delta, worse.would_promote, worse.promoted) == (pytest.approx(-0.1), False, False)
    assert unsure.delta > 0 > unsure.delta_ci_low, "the fixture's interval straddles zero"
    assert (unsure.would_promote, unsure.promoted) == (False, False)
    assert promotion(noisy, rival, allowed=True, **kwargs) == unsure
    for decision in (tie, better, allowed, worse, unsure):
        assert decision.rival is ScorerName.BASELINE
        assert decision.holder_version is None
        assert "baseline" in decision.reason

    holder = promotion(
        rival + 0.1,
        rival,
        sources=sources,
        rival_name=ScorerName.HOLDER,
        holder_version="3",
        allowed=True,
        seed=3,
    )
    assert (holder.rival, holder.holder_version, holder.promoted) == (ScorerName.HOLDER, "3", True)
    assert "version 3" in holder.reason
    with pytest.raises(ValueError, match="one length"):
        promotion(rival[:-1], rival, allowed=True, **kwargs)
    with pytest.raises(ValueError, match="non-empty"):
        promotion([], [], allowed=True, **kwargs)
    with pytest.raises(ValueError, match="one label per value"):
        promotion(rival, rival, allowed=True, **{**kwargs, "sources": sources[:-1]})


def test_the_promotion_interval_resamples_source_pages_not_groups() -> None:
    """Sixty groups from three source pages, twenty rounds each: two gain 0.1, one loses 0.05.
    Over groups the interval clears zero; over the three pages, the draw of the losing page
    three times (1 in 27) sits inside the lower 2.5%, so it does not."""
    difference = np.array([0.1] * 40 + [-0.05] * 20)
    by_page = ["example.com/a"] * 20 + ["example.com/b"] * 20 + ["example.com/c"] * 20
    by_group = [f"example.com/g{i}" for i in range(60)]
    kwargs = {"rival_name": ScorerName.BASELINE, "holder_version": None, "seed": 3}
    rival = np.full(60, 0.5)

    groups = promotion(rival + difference, rival, sources=by_group, allowed=True, **kwargs)
    pages = promotion(rival + difference, rival, sources=by_page, allowed=True, **kwargs)

    assert groups.delta == pages.delta == pytest.approx(0.05)
    assert groups.would_promote, "the fixture's per-group interval clears zero"
    assert pages.delta_ci_low == pytest.approx(-0.05)
    assert (pages.would_promote, pages.promoted) == (False, False)
    assert "from 3 source pages" in pages.reason


def test_bootstrap_resamples_whole_clusters_and_metrics_cluster_by_source_page() -> None:
    values = [1.0] * 5 + [0.0] * 5
    pages = ["a"] * 5 + ["b"] * 5

    by_value = bootstrap_ci(values, seed=1)
    by_page = bootstrap_ci(values, clusters=pages, seed=1)

    assert 0.1 < by_value[0] < by_value[1] < 0.9
    assert by_page == (0.0, 1.0), "two pages resample to 0, 0.5 or 1"
    with pytest.raises(ValueError, match="one label per value"):
        bootstrap_ci(values, clusters=pages[:-1], seed=1)
    # Two source pages over three rounds: one always ranks its positive first, one second.
    rows = [
        (r, source, target, label, score)
        for r in range(3)
        for source, first in (("s1", 1), ("s2", 0))
        for target, label, score in (("a", first, 0.9), ("b", 1 - first, 0.1))
    ]
    frame = pandas.DataFrame(rows, columns=["round", "source_url", "target_url", "label", "score"])

    found = ranking_metrics(frame, "score", ScorerName.LEARNED, all_groups=6, seed=1)

    assert found.groups == 6
    assert (found.ci_low, found.ci_high) == pytest.approx((1 / LOG3, 1.0)), (
        "the interval resamples groups, not source pages"
    )


# ── planted labels ──────────────────────────────────────────────────────────


def test_planted_labels_distribution_non_degenerate(planted: pandas.DataFrame) -> None:
    pairs = [(s, t) for s in URLS for t in URLS if s != t]

    grades = planted_grades(pairs)

    shares = {grade: n / len(pairs) for grade, n in Counter(grades.tolist()).items()}
    assert sorted(shares) == [0, 1, 2, 3]
    assert min(shares.values()) >= 0.01, f"a grade under 1% of pairs: {shares}"
    assert max(shares.values()) < 0.5, f"one grade holds half the pairs: {shares}"
    within = planted.groupby("source_url")["label"].nunique()
    assert (within >= 2).all(), "a source page whose candidates all share one grade"
    assert (
        planted["label"].tolist()
        == planted_grades(zip(planted["source_url"], planted["target_url"], strict=True)).tolist()
    ), "the frame's labels are the planted grades, not its features"
    assert planted_grades(pairs).tolist() == grades.tolist()


@pytest.mark.parametrize(
    ("source", "target", "grade"),
    [
        (page_url("trail", 1), page_url("tent", 5), 3),
        (page_url("tent", 5), page_url("trail", 1), 3),
        (page_url("kayak", 2), page_url("trail", ORPHAN_INDEX), 3),
        (page_url("trail", 2), page_url("kayak", ORPHAN_INDEX), 3),
        (page_url("trail", 1), page_url("trail", 9), 2),
        (page_url("kayak", 1), page_url("kayak", 20), 2),
        (page_url("trail", 1), noise_url("contact"), 1),
        (noise_url("press"), noise_url("contact"), 1),
        (page_url("kayak", 1), page_url("trail", 3), 0),
        (noise_url("press"), page_url("kayak", 3), 0),
        (page_url("trail", ORPHAN_INDEX), page_url("kayak", 3), 0),
    ],
)
def test_each_planted_rule_grades_its_pairs(source: str, target: str, grade: int) -> None:
    assert planted_grades([(source, target)]).tolist() == [grade]


def test_a_page_paired_with_itself_has_no_grade() -> None:
    [page] = sorted(NOISE_URLS)[:1]
    with pytest.raises(ValueError, match="two pages"):
        planted_grades([(page, page)])


# ── summary ─────────────────────────────────────────────────────────────────


def test_the_summary_states_what_ran_what_it_achieved_and_the_limitations() -> None:
    report = make.report(tenant_id="acme")

    text = summarise_ranker(report)

    assert "acme" in text
    for scorer in (ScorerName.LEARNED, ScorerName.LEARNED_EXCL_LINK_COUNTS, ScorerName.BASELINE):
        assert f"{scorer.value}: NDCG@10" in text
    assert "target_crawl_depth" in text
    assert "content_cosine holds more than 60% of the gain" in text
    assert "Promotion:" in text
    for limitation in (
        "existing links",
        "link-count",
        "learned_excl_placement",
        "mean average precision",
    ):
        assert limitation in text, f"the summary does not state the {limitation} limitation"
    assert (
        "The placement columns (context_relevance, anchor_target_fit) hold 0.0% of the gain."
        in text
    )
    placed = make.report(
        tenant_id="acme",
        columns=(*make.COLUMNS, "context_relevance"),
        importance=make.importance(context_relevance=0.7, content_cosine=0.1, same_hub=0.1),
        dominant_feature="context_relevance",
    )
    assert "rests mostly on whether a pair has an anchor" in summarise_ranker(placed)
    assert "50% of the body links hidden once, the rest never" in summarise_ranker(
        make.report(
            settings=HeldOutSettings(rounds=5, evaluation_seeds=(7, 11)), rounds=(), positives=0
        )
    )
    assert "every body link hidden in exactly one round" in summarise_ranker(
        make.report(settings=HeldOutSettings(evaluation_seeds=(7, 11)), rounds=(), positives=0)
    )
    # #94: what the orphan rule, the constraints and the seeds did, in plain words.
    assert (
        "Orphan targets: 6 candidate target pages have no inbound body link on the full graph"
        in text
    )
    assert "480 candidate rows are unlabelable" in text
    assert (
        "Monotone-increasing columns: content_cosine, anchor_target_fit, context_relevance" in text
    )
    assert "Across 2 evaluated split seeds (7, 11)" in text
    assert "the split seed (7) reuses the fixed-split models" in text
    assert "significantly worse in 1 of 2 evaluated seeds and better in 0" in text
    skipping = summarise_ranker(
        make.report(
            seed_results=(make.seed_result(7),),
            skipped_seeds={11: "24 test groups with a hidden link, fewer than 30"},
        )
    )
    assert "seed 11 skipped: 24 test groups with a hidden link, fewer than 30" in skipping
    assert "On the production candidates, the first 10 pairs of every source page" in text
    assert "50.0% of the orphans in a hub showing its main page, inbound Gini 0.420" in text
    assert "held-out NDCG says nothing about how orphans are ranked" in text
    unconstrained = summarise_ranker(
        make.report(
            params=RankerParams(monotone_increasing=()),
            product_measures=(),
            product_skipped_reason="no anchor choices for the tenant",
        )
    )
    assert "No monotone constraints." in unconstrained
    assert "Product measures skipped: no anchor choices for the tenant" in unconstrained
    skipped = summarise_ranker(make.skipped(tenant_id="acme"))
    assert "Skipped: 12 training groups" in skipped
    assert "mean average precision" in skipped
    assert "NDCG@10 0." not in skipped
