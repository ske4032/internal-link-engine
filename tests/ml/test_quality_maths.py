"""#74 maths: held-out links, recall against the random baseline, AUC, keyword matching, the
flat metrics of a report and soft alerts. Pure functions, no stores."""

from __future__ import annotations

import hashlib
import math
import random
import re

import numpy as np
import pytest
import quality_factories as make
from sklearn.metrics import roc_auc_score

from linking_engine.discovery.features import FEATURE_COLUMNS
from linking_engine.ml.quality import (
    _LINK_DERIVED_FIELDS,
    ALERT_BAND,
    ALERT_BANDS,
    HEADLINE_METRICS,
    HIDE_SEED,
    KEYWORD_ORIGINS,
    LENGTH_BINS,
    LINK_DERIVED_COLUMNS,
    Band,
    alerts,
    anchor_matches,
    auc,
    copy_words,
    feature_auc,
    has_signal,
    hide_links,
    keyword_origin,
    keyword_words,
    length_bin,
    quality_metrics,
    random_recall,
    recall_at,
    relevance_groups,
)
from linking_engine.models import FeatureAuc, KeywordRung, KeywordSource

NAN = math.nan

# ── held-out links ──────────────────────────────────────────────────────────

# 400 distinct body links: sources s0..s399 into four targets.
PAIRS = [(f"s{i}", f"t{i % 4}") for i in range(400)]


def pair_hash(seed: int, source: str, target: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\t{source}\t{target}".encode()).digest())


def test_a_link_is_hidden_when_its_seeded_hash_falls_under_the_share() -> None:
    hidden = hide_links(PAIRS)

    cut = int(0.1 * 2**256)
    assert hidden == {pair for pair in PAIRS if pair_hash(HIDE_SEED, *pair) < cut}
    # Binomial(400, 0.1): mean 40, sd 6; the count floats, the share holds.
    assert 22 <= len(hidden) <= 58


def test_the_hidden_set_ignores_order_and_duplicates_and_follows_the_seed() -> None:
    shuffled = PAIRS[::-1] + PAIRS[:10]
    random.Random(3).shuffle(shuffled)

    assert hide_links(shuffled) == hide_links(PAIRS)
    assert hide_links(PAIRS, seed=7) != hide_links(PAIRS), "the seed does not drive the draw"


def test_a_link_keeps_its_hidden_state_when_other_links_come_and_go() -> None:
    """A hash per pair, not a draw over the set: adding an unrelated link, or dropping
    others, never changes whether an existing link is hidden."""
    hidden = hide_links(PAIRS)
    grown = [*PAIRS, *((f"new{i}", "t9") for i in range(50))]
    shrunk = PAIRS[:200]

    assert hide_links(grown) & set(PAIRS) == hidden
    assert hide_links(shrunk) == hidden & set(shrunk)


def test_the_lowest_hash_is_hidden_when_the_share_would_hide_none() -> None:
    few = [("a", "b"), ("c", "d"), ("e", "f")]
    cut = int(0.1 * 2**256)
    assert all(pair_hash(HIDE_SEED, *pair) >= cut for pair in few), "the fixture hides one"

    assert hide_links(few) == {min(few, key=lambda pair: pair_hash(HIDE_SEED, *pair))}
    assert hide_links([]) == frozenset()


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.5])
def test_the_hidden_share_must_be_strictly_between_zero_and_one(share: float) -> None:
    with pytest.raises(ValueError, match="share"):
        hide_links(PAIRS, share=share)


def test_hide_links_folds_are_disjoint_and_fold0_matches_74() -> None:
    folds = [hide_links(PAIRS, fold=r) for r in range(10)]

    assert folds[0] == hide_links(PAIRS), "fold 0 is the quality evaluation's hidden set"
    for r, fold in enumerate(folds):
        low, high = int(r * 0.1 * 2**256), int((r + 1) * 0.1 * 2**256)
        assert fold == {pair for pair in PAIRS if low <= pair_hash(HIDE_SEED, *pair) < high}
    assert sum(len(fold) for fold in folds) == len(set().union(*folds)), "folds overlap"
    assert set().union(*folds) == set(PAIRS), "ten folds of 0.1 cover every link"
    shares = [hide_links(PAIRS, share=0.2, fold=r) for r in range(5)]
    assert set().union(*shares) == set(PAIRS)
    assert shares[0] == folds[0] | folds[1]


def test_a_fold_never_repeats_the_link_fold_0_took_for_want_of_one() -> None:
    few = [("a", "b"), ("c", "d"), ("e", "f")]
    lowest = min(few, key=lambda pair: pair_hash(HIDE_SEED, *pair))

    folds = [hide_links(few, fold=r) for r in range(10)]

    assert folds[0] == {lowest}
    assert all(lowest not in fold for fold in folds[1:])
    assert sum(len(fold) for fold in folds) == len(set().union(*folds))
    assert set().union(*folds) == set(few)


@pytest.mark.parametrize(("share", "fold"), [(0.1, 10), (0.1, -1), (0.2, 5), (0.3, 3)])
def test_hide_links_fold_out_of_range_raises(share: float, fold: int) -> None:
    with pytest.raises(ValueError, match="fold"):
        hide_links(PAIRS, share=share, fold=fold)


def test_recall_and_the_random_baseline_on_planted_ranks() -> None:
    """Four hidden links planted at ranks 1, 6, 11 and 16 of their target's candidates, whose
    targets have 20, 20, 40 and 40 eligible sources."""
    ranks = [1, 6, 11, 16]
    eligible = [20, 20, 40, 40]

    assert [recall_at(ranks, k) for k in (1, 5, 10, 11, 16, 50)] == [
        0.25,
        0.25,
        0.5,
        0.75,
        1.0,
        1.0,
    ]
    assert random_recall(eligible, 10) == pytest.approx((0.5 + 0.5 + 0.25 + 0.25) / 4)
    assert random_recall(eligible, 50) == 1.0, "a baseline never exceeds 1"


def test_a_hidden_link_not_retrieved_counts_as_missed_at_every_k() -> None:
    assert recall_at([1, None, 3, None], 50) == 0.5
    assert recall_at([None], 1) == 0.0


@pytest.mark.parametrize(
    ("ranks", "k", "message"), [([], 10, "no held-out links"), ([1], 0, "at least 1")]
)
def test_recall_needs_links_and_a_positive_k(ranks: list[int | None], k: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        recall_at(ranks, k)


@pytest.mark.parametrize(
    ("eligible", "k"), [([], 10), ([5, 0], 10), ([5], 0)], ids=["none", "zero", "k0"]
)
def test_the_random_baseline_needs_eligible_sources_and_a_positive_k(
    eligible: list[int], k: int
) -> None:
    with pytest.raises(ValueError, match="at least"):
        random_recall(eligible, k)


# ── AUC ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        pytest.param([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0, id="separable"),
        pytest.param([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.0, id="reversed"),
        pytest.param([1, 0, 1, 0], [1.0, 1.0, 2.0, 2.0], 0.5, id="inseparable"),
        pytest.param([1, 0], [3.0, 3.0], 0.5, id="tie-counts-half"),
        pytest.param([1, 0, 0], [-math.inf, 0.0, -math.inf], 0.25, id="infinite"),
    ],
)
def test_auc_of_known_fixtures(labels: list[int], scores: list[float], expected: float) -> None:
    assert auc(labels, scores) == pytest.approx(expected)


def test_auc_matches_scikit_learn_with_ties() -> None:
    rng = np.random.default_rng(0)
    labels = rng.random(300) < 0.3
    scores = rng.integers(0, 12, 300).astype(float)

    assert auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))


@pytest.mark.parametrize("labels", [[1, 1, 1], [0, 0, 0], []])
def test_auc_is_undefined_with_one_class(labels: list[int]) -> None:
    assert auc(labels, [0.5] * len(labels)) is None


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([1, 0], [0.5, NAN], "NaN"),
        ([1, 0, 1], [0.5, 0.2], "one length"),
        ([[1, 0]], [[0.5, 0.2]], "vectors"),
    ],
)
def test_auc_refuses_nan_and_mismatched_input(
    labels: list[object], scores: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        auc(labels, scores)


def test_a_feature_auc_drops_missing_rows_and_ranks_them_last_as_a_ranker() -> None:
    entry = feature_auc("content_cosine", [1, 1, 0, 0, 1, 0], [0.9, 0.8, 0.1, 0.2, NAN, NAN])

    assert entry.column == "content_cosine"
    assert entry.auc == 1.0, "the rows without a value are not part of the AUC"
    assert entry.coverage == pytest.approx(4 / 6)
    # Over all nine hidden/other pairs, missing last: 6 wins and one missing-missing tie.
    assert entry.ranker_auc == pytest.approx(6.5 / 9)
    assert has_signal(entry)


def test_a_feature_that_orders_hidden_links_low_still_has_signal() -> None:
    entry = feature_auc("target_saturation_ratio", [1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9])

    assert (entry.auc, entry.ranker_auc, entry.coverage) == (0.0, 1.0, 1.0)
    assert has_signal(entry)


def test_an_inseparable_feature_has_no_signal() -> None:
    entry = feature_auc("same_hub", [1, 0, 1, 0, 1, 0], [1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    assert entry.auc == pytest.approx(0.5)
    assert not has_signal(entry)


@pytest.mark.parametrize(
    ("values", "coverage"),
    [
        pytest.param([NAN, NAN, NAN, NAN], 0.0, id="all-null"),
        pytest.param([2.0, 2.0, 2.0, 2.0], 1.0, id="constant"),
    ],
)
def test_a_column_without_two_values_has_no_auc(values: list[float], coverage: float) -> None:
    entry = feature_auc("target_ctr_gap", [1, 0, 1, 0], values)

    assert (entry.auc, entry.coverage, entry.ranker_auc) == (None, coverage, 0.5)
    assert not has_signal(entry)


def test_a_column_whose_hidden_links_all_lack_a_value_is_reported_not_raised() -> None:
    """A GSC column where the hidden links' targets have no GSC data, the others some."""
    entry = feature_auc("target_ctr_gap", [1, 1, 0, 0, 0, 0], [NAN, NAN, 1.0, 2.0, NAN, 3.0])

    assert entry.auc is None, "the rows with a value hold one class only"
    assert entry.coverage == pytest.approx(0.5)
    assert 0.0 <= entry.ranker_auc <= 1.0


@pytest.mark.parametrize(
    ("value", "signal"),
    [(0.55, True), (0.45, True), (0.5499, False), (0.4501, False), (None, False)],
)
def test_signal_is_at_least_the_margin_from_half_in_either_direction(
    value: float | None, signal: bool
) -> None:
    entry = FeatureAuc(column="c", auc=value, coverage=1.0, ranker_auc=0.5)
    assert has_signal(entry) is signal


def test_the_signal_margin_can_be_widened() -> None:
    entry = FeatureAuc(column="c", auc=0.65, coverage=1.0, ranker_auc=0.65)
    assert has_signal(entry, margin=0.1)
    assert not has_signal(entry, margin=0.2)


def test_the_link_derived_columns_are_the_matrix_columns_hiding_a_link_moves() -> None:
    assert set(LINK_DERIVED_COLUMNS) <= set(FEATURE_COLUMNS)
    assert {
        "target_inbound_count",
        "source_outbound_count",
        "same_link_community",
        "target_crawl_depth",
    } <= set(LINK_DERIVED_COLUMNS)
    for field in _LINK_DERIVED_FIELDS:
        assert [c for c in LINK_DERIVED_COLUMNS if c == field or c.startswith(f"{field}_")], (
            f"{field} names no matrix column"
        )
    one_hot = [
        c for c in FEATURE_COLUMNS if c.startswith(("cluster_agreement_", "content_agreement_"))
    ]
    assert one_hot
    assert set(one_hot) <= set(LINK_DERIVED_COLUMNS), "a one-hot level escaped the exclusion"
    content = {"content_cosine", "same_hub", "target_impressions_log", "pair_kw_overlap"}
    assert not content & set(LINK_DERIVED_COLUMNS)


# ── keywords ────────────────────────────────────────────────────────────────


def test_keyword_words_are_normalised_and_drop_one_letter_words() -> None:
    assert keyword_words("The Trail-Running Shoes, a guide") == {
        "the",
        "trail",
        "running",
        "shoes",
        "guide",
    }
    assert keyword_words("\uff34\uff52\uff41\uff49\uff4c STRASSE") == keyword_words("trail Straße")
    assert keyword_words("a b") == frozenset()


@pytest.mark.parametrize(
    ("anchor", "keyword", "matches"),
    [
        pytest.param("Trail Running Shoes", "trail running shoes", True, id="same-words"),
        pytest.param("best trail running shoes", "trail running shoes", True, id="anchor-holds"),
        pytest.param("running shoes", "trail running shoes", True, id="keyword-holds"),
        pytest.param("red trail shoes", "blue trail shoes", True, id="jaccard-half"),
        pytest.param("red trail boots", "blue trail shoes", False, id="jaccard-fifth"),
        pytest.param("click here", "trail shoes", False, id="unrelated"),
        pytest.param("", "trail shoes", False, id="empty-anchor"),
        pytest.param("a", "trail shoes", False, id="one-letter-anchor"),
        pytest.param("trail shoes", "", False, id="empty-keyword"),
    ],
)
def test_an_anchor_matches_a_keyword_by_containment_or_half_its_words(
    anchor: str, keyword: str, matches: bool
) -> None:
    assert anchor_matches(keyword_words(anchor), keyword_words(keyword)) is matches


def test_copy_words_are_every_normalised_word_of_the_copy() -> None:
    copy = copy_words("Our guide to  Trail   SHOES, and the trail-running shoesmith.")

    assert copy == {
        "our",
        "guide",
        "to",
        "trail",
        "shoes",
        "and",
        "the",
        "running",
        "shoesmith",
    }
    assert keyword_words("shoes trail") <= copy
    assert not keyword_words("hiking boots") <= copy


@pytest.mark.parametrize(
    ("rank", "rung", "source", "origin"),
    [
        (1, KeywordRung.H1, KeywordSource.INFERRED, "primary_h1"),
        (1, KeywordRung.STRATEGIC, KeywordSource.CLIENT_STRATEGIC, "primary_strategic"),
        (2, None, KeywordSource.CLIENT_STRATEGIC, "secondary_client_strategic"),
        (3, None, KeywordSource.GSC_OBSERVED, "secondary_gsc_observed"),
        # Only rank 1 is primary, whatever the edge says.
        (2, KeywordRung.GSC, KeywordSource.GSC_OBSERVED, "secondary_gsc_observed"),
    ],
)
def test_a_keyword_origin_is_its_rung_at_rank_one_and_its_source_after(
    rank: int, rung: KeywordRung | None, source: KeywordSource, origin: str
) -> None:
    assert keyword_origin(rank, rung, source) == origin
    assert origin in KEYWORD_ORIGINS


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("", "1_2"),
        ("Tents", "1_2"),
        ("trail   shoes", "1_2"),
        ("trail running shoes", "3_4"),
        ("one two three four", "3_4"),
        ("one two three four five", "5_7"),
        ("one two three four five six seven", "5_7"),
        ("one two three four five six seven eight", "8plus"),
    ],
)
def test_a_keyword_falls_in_one_length_bin_by_its_normalised_word_count(
    text: str, label: str
) -> None:
    assert length_bin(text) == label
    assert label in [name for name, _, _ in LENGTH_BINS]


def test_relevance_groups_keep_the_given_order_and_skip_empty_groups() -> None:
    groups = relevance_groups({"b": [0.2, 0.4, 0.9], "a": [0.5], "c": []}, ["a", "b", "c", "d"])

    assert [(g.group, g.keywords) for g in groups] == [("a", 1), ("b", 3)]
    assert (groups[1].mean, groups[1].p50) == (pytest.approx(0.5), 0.4)
    with pytest.raises(ValueError, match="unknown relevance groups"):
        relevance_groups({"z": [0.1]}, ["a"])


# ── flat metrics ────────────────────────────────────────────────────────────

# MLflow metric names: alphanumerics, underscores, dashes, periods, spaces and slashes.
_MLFLOW_NAME = re.compile(r"^[\w\-. /]+$")


def test_every_number_of_a_full_report_has_a_stable_name() -> None:
    metrics = quality_metrics(make.report())

    assert metrics == {
        "seconds": 4.5,
        "not_applicable_checks": 0.0,
        "alerts": 0.0,
        "body_link_pairs": 40.0,
        "hidden_links": 4.0,
        "hidden_recoverable": 3.0,
        "held_out_candidates": 120.0,
        "recall_at_10": 0.5,
        "random_recall_at_10": 0.25,
        "recall_at_20": 0.75,
        "random_recall_at_20": 0.5,
        "recall_at_50": 1.0,
        "random_recall_at_50": 1.0,
        "auc_pairs": 120.0,
        "auc_positives": 3.0,
        "features_with_signal": 1.0,
        "auc_content_cosine": 0.9,
        "auc_same_hub": 0.52,
        "score_auc": 0.8,
        "best_feature_auc": 0.9,
        "score_auc_lift": pytest.approx(-0.1),
        "link_derived_weight_share": 0.5,
        "score_auc_excl_link_counts": 0.7,
        "best_feature_auc_excl_link_counts": 0.9,
        "score_auc_lift_excl_link_counts": pytest.approx(-0.2),
        "keyword_pages": 10.0,
        "keyword_resolved": 4.0,
        "keyword_rung_strategic": 1.0,
        "keyword_rung_h1": 3.0,
        "keyword_rejected_generic": 2.0,
        "keyword_rejected_repeated": 1.0,
        "keyword_unique_share": 0.75,
        "extract_pairs": 8.0,
        "extract_found_primary": 0.5,
        "extract_found_set": 0.75,
        "extract_exact_set": 0.375,
        "extract_stemmed_set": 0.25,
        "extract_stem_set_set": 0.125,
        "extract_words_primary": 0.5,
        "extract_words_set": 0.75,
        "extract_stem_set_threshold": 0.6,
        "anchor_match_anchors": 4.0,
        "anchor_match_primary": 0.25,
        "anchor_match_set": 0.5,
        "keyword_relevance_texts": 5.0,
        "keyword_vectors_embedded": 5.0,
        "keyword_vectors_cached": 0.0,
        "keyword_embed_tokens": 12.0,
        "keyword_relevance_rank_1_mean": 0.6,
        "keyword_relevance_rank_1_keywords": 3.0,
        "keyword_relevance_rank_2_mean": 0.4,
        "keyword_relevance_rank_2_keywords": 2.0,
        "keyword_relevance_primary_h1_mean": 0.6,
        "keyword_relevance_secondary_gsc_observed_mean": 0.4,
        "keyword_relevance_len_1_2_mean": 0.55,
        "keyword_relevance_len_3_4_mean": 0.3,
        "link_relevance_links": 30.0,
        "context_relevance_count": 25.0,
        "context_relevance_mean": 0.7,
        "context_relevance_p10": 0.55,
        "context_relevance_p25": 0.6,
        "context_relevance_p50": 0.7,
        "context_relevance_p75": 0.8,
        "context_relevance_p90": 0.85,
        "context_relevance_split": 0.62,
        "context_relevance_low_share": 0.3,
        "anchor_target_fit_count": 25.0,
        "anchor_target_fit_mean": 0.7,
        "anchor_target_fit_p10": 0.55,
        "anchor_target_fit_p25": 0.6,
        "anchor_target_fit_p50": 0.7,
        "anchor_target_fit_p75": 0.8,
        "anchor_target_fit_p90": 0.85,
        "coverage_pairs": 120.0,
        "gsc_pair_share": 0.25,
        "keyword_page_share": 0.4,
        "all_null_columns": 2.0,
        "constant_columns": 1.0,
    }
    assert all(isinstance(value, float) and math.isfinite(value) for value in metrics.values())
    assert [name for name in metrics if not _MLFLOW_NAME.match(name)] == []


def test_checks_that_do_not_apply_are_left_out_never_logged_as_zero() -> None:
    report = make.report(
        not_applicable=("retrieval", "feature_signal", "scorer", "keywords", "link_relevance"),
        coverage=make.coverage(pairs=0, gsc_pair_share=None, keyword_page_share=None),
    )

    assert quality_metrics(report) == {
        "seconds": 4.5,
        "not_applicable_checks": 5.0,
        "alerts": 0.0,
        "coverage_pairs": 0.0,
        "all_null_columns": 2.0,
        "constant_columns": 1.0,
    }


def test_a_keyword_sub_check_that_does_not_apply_leaves_the_rest_of_the_section() -> None:
    report = make.report(not_applicable=("keyword_relevance", "anchor_match"))

    metrics = quality_metrics(report)

    assert not [name for name in metrics if name.startswith(("keyword_relevance", "anchor_match"))]
    assert "keyword_vectors_embedded" not in metrics
    assert (metrics["keyword_resolved"], metrics["extract_pairs"]) == (4.0, 8.0)
    assert metrics["not_applicable_checks"] == 2.0


def test_an_excluded_score_that_cannot_be_computed_is_left_out() -> None:
    scorer = make.scorer(
        link_derived_weight_share=1.0,
        score_auc_excl_link_counts=None,
        score_auc_lift_excl_link_counts=None,
    )

    metrics = quality_metrics(make.report(scorer=scorer))

    assert "score_auc_excl_link_counts" not in metrics
    assert "score_auc_lift_excl_link_counts" not in metrics
    assert (metrics["link_derived_weight_share"], metrics["best_feature_auc_excl_link_counts"]) == (
        1.0,
        0.9,
    )


def test_every_headline_metric_is_a_name_a_full_report_produces() -> None:
    metrics = quality_metrics(make.report())

    assert [name for name in HEADLINE_METRICS if name not in metrics] == []
    assert set(ALERT_BANDS) <= set(HEADLINE_METRICS)


# ── soft alerts ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("previous", "current", "change"),
    [
        pytest.param(10.0, 12.0, None, id="up-exactly-the-band"),
        pytest.param(10.0, 8.0, None, id="down-exactly-the-band"),
        pytest.param(10.0, 12.5, 0.25, id="up-beyond"),
        pytest.param(10.0, 7.5, -0.25, id="down-beyond"),
        pytest.param(-10.0, -12.5, -0.25, id="negative-previous"),
    ],
)
def test_a_metric_alerts_when_it_moves_beyond_its_relative_band(
    previous: float, current: float, change: float | None
) -> None:
    found = alerts({"features_with_signal": current}, {"features_with_signal": previous})

    if change is None:
        assert found == ()
    else:
        [moved] = found
        assert (moved.metric, moved.previous, moved.current) == (
            "features_with_signal",
            previous,
            current,
        )
        assert moved.change == pytest.approx(change)
        assert (moved.band, moved.relative) == (ALERT_BAND, True)


def test_aucs_and_cosines_alert_on_an_absolute_band() -> None:
    assert alerts({"score_auc": 0.74}, {"score_auc": 0.70}) == ()
    [moved] = alerts({"score_auc": 0.60}, {"score_auc": 0.70})

    assert moved.change == pytest.approx(-0.1)
    assert (moved.band, moved.relative) == (0.05, False)


def test_a_check_becoming_not_applicable_alerts() -> None:
    [moved] = alerts({"not_applicable_checks": 1.0}, {"not_applicable_checks": 0.0})

    assert (moved.change, moved.band, moved.relative) == (1.0, 0.5, False)
    assert alerts({"not_applicable_checks": 2.0}, {"not_applicable_checks": 2.0}) == ()
    assert "not_applicable_checks" in HEADLINE_METRICS


def test_the_like_for_like_scorer_bar_alerts_on_an_absolute_band() -> None:
    assert alerts({"score_auc_excl_link_counts": 0.64}, {"score_auc_excl_link_counts": 0.6}) == ()
    [moved] = alerts({"score_auc_excl_link_counts": 0.5}, {"score_auc_excl_link_counts": 0.6})
    assert (moved.band, moved.relative) == (0.05, False)


def test_a_metric_leaving_zero_alerts_without_a_change() -> None:
    [moved] = alerts({"gsc_pair_share": 0.3}, {"gsc_pair_share": 0.0})

    assert (moved.previous, moved.current, moved.change) == (0.0, 0.3, None)
    assert alerts({"gsc_pair_share": 0.0}, {"gsc_pair_share": 0.0}) == ()


@pytest.mark.parametrize(
    ("current", "previous"),
    [
        pytest.param({}, {"recall_at_10": 0.5}, id="gone-now"),
        pytest.param({"recall_at_10": 0.5}, {}, id="new-now"),
        pytest.param({"recall_at_10": NAN}, {"recall_at_10": 0.5}, id="nan"),
        pytest.param({"recall_at_10": 0.5}, {"recall_at_10": math.inf}, id="inf"),
        pytest.param({"seconds": 500.0}, {"seconds": 5.0}, id="not-a-headline"),
    ],
)
def test_only_finite_headline_metrics_of_both_runs_are_compared(
    current: dict[str, float], previous: dict[str, float]
) -> None:
    assert alerts(current, previous) == ()


def test_the_bands_and_the_compared_metrics_can_be_configured() -> None:
    now, before = {"recall_at_10": 0.7, "seconds": 9.0}, {"recall_at_10": 0.5, "seconds": 3.0}

    assert [a.metric for a in alerts(now, before)] == ["recall_at_10"]
    assert alerts(now, before, bands={"recall_at_10": Band(0.5)}) == ()
    assert alerts(now, before, band=0.5) == ()
    assert [a.metric for a in alerts(now, before, metrics=("seconds",))] == ["seconds"]


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"band": 0.0}, id="zero-band"),
        pytest.param({"band": -0.2}, id="negative-band"),
        pytest.param({"bands": {"score_auc": Band(0.0, relative=False)}}, id="zero-override"),
    ],
)
def test_a_non_positive_band_is_refused(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive"):
        alerts({}, {}, **options)  # type: ignore[arg-type]
