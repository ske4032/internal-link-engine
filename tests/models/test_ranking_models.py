"""Validators of the #24-#27 ranker models: each rejection beside the boundary that passes."""

from __future__ import annotations

import pytest
import ranking_factories as make
from pydantic import BaseModel, ValidationError

from linking_engine.models import (
    HeldOutSettings,
    ImportanceEntry,
    ProductMeasures,
    PromotionDecision,
    RankerParams,
    RankerReport,
    RankingMetrics,
    RankReport,
    RoundSummary,
    ScorerName,
    SeedResult,
)

MODELS: tuple[type[BaseModel], ...] = (
    SeedResult,
    ProductMeasures,
    RankerParams,
    HeldOutSettings,
    RoundSummary,
    RankingMetrics,
    ImportanceEntry,
    PromotionDecision,
    RankerReport,
    RankReport,
)


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
def test_every_ranker_model_is_frozen_and_forbids_extras(model: type[BaseModel]) -> None:
    assert model.model_config.get("frozen") is True
    assert model.model_config.get("extra") == "forbid"


def test_a_report_round_trips_through_json() -> None:
    report = make.report()

    assert RankerReport.model_validate_json(report.model_dump_json()) == report


def test_the_contract_defaults() -> None:
    assert HeldOutSettings().model_dump() == {
        "rounds": 10,
        "share": 0.1,
        "seed": 42,
        "test_share": 0.2,
        "valid_share": 0.1,
        "split_seed": 7,
        "evaluation_seeds": (7, 11, 23, 42, 99),
    }
    assert RankerParams().model_dump() == {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "max_rounds": 1000,
        "early_stopping_rounds": 50,
        "eval_at": 10,
        "seed": 42,
        "monotone_increasing": ("content_cosine", "anchor_target_fit", "context_relevance"),
    }


@pytest.mark.parametrize(("rounds", "share"), [(10, 0.1), (5, 0.2), (3, 1 / 3), (1, 0.99)])
def test_rounds_may_cover_the_whole_hash_range(rounds: int, share: float) -> None:
    assert HeldOutSettings(rounds=rounds, share=share).rounds == rounds


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"rounds": 11, "share": 0.1}, "rounds \\* share"),
        ({"rounds": 0}, "rounds"),
        ({"share": 0.0}, "share"),
        ({"share": 1.0}, "share"),
        ({"test_share": 1.0}, "test_share"),
        ({"valid_share": 0.0}, "valid_share"),
    ],
)
def test_held_out_settings_outside_their_range_are_refused(
    fields: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        HeldOutSettings.model_validate(fields)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"positives": 37}, "positives <= recoverable <= hidden"),
        ({"recoverable": 41, "positives": 30}, "positives <= recoverable <= hidden"),
        ({"pairs": 20, "positives": 30, "recoverable": 30}, "more positives than pairs"),
        ({"groups_with_positive": 37}, "more groups with a positive"),
        ({"positive_placement_share": None}, "positive_placement_share is None exactly"),
        (
            {"positives": 0, "recoverable": 0, "groups_with_positive": 0},
            "positive_placement_share is None exactly",
        ),
        ({"negative_placement_share": None}, "negative_placement_share is None exactly"),
        (
            {"pairs": 36, "negative_placement_share": 0.5},
            "negative_placement_share is None exactly",
        ),
        ({"positive_placement_share": 1.1}, "less than or equal to 1"),
    ],
)
def test_a_round_summary_is_consistent(fields: dict[str, object], match: str) -> None:
    assert make.round_summary(positives=36, groups_with_positive=36).positives == 36
    empty = make.round_summary(
        positives=0, recoverable=0, groups_with_positive=0, positive_placement_share=None
    )
    assert empty.positive_placement_share is None
    only = make.round_summary(pairs=36, negative_placement_share=None)
    assert only.negative_placement_share is None
    with pytest.raises(ValidationError, match=match):
        make.round_summary(**fields)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"ci_low": 0.7}, "ci_low"),
        ({"n_labelled_pairs": 39}, "every evaluated group holds a positive"),
        ({"per_round": {}}, "per_round"),
        ({"per_round": {-1: 0.5}}, "per_round"),
        ({"histogram": (0,) * 9 + (39,)}, "add up to the groups"),
        ({"histogram": (1,) * 9}, "at least 10"),
    ],
)
def test_ranking_metrics_are_consistent(fields: dict[str, object], match: str) -> None:
    assert make.metrics(n_labelled_pairs=40).groups == 40
    with pytest.raises(ValidationError, match=match):
        make.metrics(**fields)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"rival": ScorerName.LEARNED}, "the holder or the baseline"),
        ({"holder_version": "2"}, "holder_version is set exactly when"),
        ({"rival": ScorerName.HOLDER}, "holder_version is set exactly when"),
        ({"delta_ci_low": 0.0}, "would_promote exactly when"),
        ({"delta_ci_low": -0.01, "would_promote": False, "promoted": True}, "only a model"),
        ({"delta_ci_low": 0.2}, "must not exceed"),
    ],
)
def test_a_promotion_decision_is_consistent(fields: dict[str, object], match: str) -> None:
    holder = make.promotion(rival=ScorerName.HOLDER, holder_version="2", promoted=True)
    assert holder.promoted
    with pytest.raises(ValidationError, match=match):
        make.promotion(**fields)


def test_a_skipped_run_carries_no_model_and_a_trained_run_carries_one() -> None:
    assert make.skipped().skipped_reason is not None
    for fields in (
        {"best_iteration": 3},
        {"model_version": "1"},
        {"promotion": make.promotion()},
        {"metrics": (make.metrics(),)},
    ):
        with pytest.raises(ValidationError, match="a skipped run has no model"):
            make.skipped(**fields)
    with pytest.raises(ValidationError, match="a trained run has a best iteration"):
        make.report(best_iteration=None)
    with pytest.raises(ValidationError, match="a trained run has a best iteration"):
        make.report(metrics=())


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"rounds": (make.round_summary(round=1), make.round_summary())}, "ascending"),
        ({"rounds": (make.round_summary(round=2),)}, "within the settings"),
        ({"body_links": 39}, "hides more links than there are"),
        ({"positives": 65}, "add up over the rounds"),
        ({"columns": ("same_hub", "same_hub")}, "duplicate columns"),
        ({"excluded_columns": {"same_hub": "why"}}, "an excluded column is a model column"),
        ({"metrics": (make.metrics(), make.metrics())}, "evaluated twice"),
        ({"importance": make.importance(unknown=0.0)}, "importance covers model columns"),
        ({"importance": make.importance(same_hub=0.3)}, "add up to 1"),
        ({"dominant_feature": "target_crawl_depth"}, "the dominant feature"),
    ],
)
def test_a_ranker_report_is_consistent(fields: dict[str, object], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        make.report(**fields)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"scorer": ScorerName.HOLDER}, "learned model or the baseline"),
        ({"model_version": None}, "model_version is set exactly when"),
        ({"fallback_reason": "no promoted model"}, "fallback_reason is set exactly when"),
        ({"scorer": ScorerName.BASELINE, "model_version": None}, "fallback_reason"),
    ],
)
def test_a_rank_report_names_its_model_or_why_the_baseline_ranked(
    fields: dict[str, object], match: str
) -> None:
    baseline = make.rank_report(
        scorer=ScorerName.BASELINE, model_version=None, fallback_reason="no promoted model"
    )
    assert baseline.fallback_reason == "no promoted model"
    with pytest.raises(ValidationError, match=match):
        make.rank_report(**fields)


# ── #94: monotone columns, evaluation seeds, seed results, product measures ─


def test_evaluation_seeds_must_include_the_split_seed() -> None:
    assert HeldOutSettings(split_seed=11, evaluation_seeds=(11,)).evaluation_seeds == (11,)
    with pytest.raises(ValidationError, match="must include the split seed"):
        HeldOutSettings(evaluation_seeds=(11, 23))
    with pytest.raises(ValidationError, match="must include the split seed"):
        HeldOutSettings(split_seed=5)
    with pytest.raises(ValidationError, match="distinct"):
        HeldOutSettings(evaluation_seeds=(7, 7, 11))


@pytest.mark.parametrize(
    "names", [("content_cosine", "content_cosine"), ("content_cosine", " ")], ids=["twice", "blank"]
)
def test_monotone_columns_are_distinct_names(names: tuple[str, ...]) -> None:
    assert RankerParams(monotone_increasing=()).monotone_increasing == ()
    with pytest.raises(ValidationError, match="distinct names"):
        RankerParams(monotone_increasing=names)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"learned": (0.5, 0.7, 0.6)}, "in \\[0, 1\\], low <= high"),
        ({"plain": (0.5, 0.4, 1.2)}, "in \\[0, 1\\], low <= high"),
        ({"delta_ci_low": 0.06}, "delta_ci_low must not exceed"),
        ({"test_pages": 0}, "greater than or equal to 1"),
    ],
)
def test_a_seed_result_is_consistent(fields: dict[str, object], match: str) -> None:
    assert make.seed_result().seed == 7
    with pytest.raises(ValidationError, match=match):
        make.seed_result(**fields)


def test_a_seed_is_significant_only_when_its_interval_leaves_zero() -> None:
    assert (make.seed_result().significantly_better, make.seed_result().significantly_worse) == (
        False,
        False,
    )
    better = make.seed_result(delta=0.05, delta_ci_low=0.01, delta_ci_high=0.09)
    worse = make.seed_result(delta=-0.05, delta_ci_low=-0.09, delta_ci_high=-0.01)
    edge = make.seed_result(delta=0.02, delta_ci_low=0.0, delta_ci_high=0.04)
    assert (better.significantly_better, better.significantly_worse) == (True, False)
    assert (worse.significantly_better, worse.significantly_worse) == (False, True)
    assert (edge.significantly_better, edge.significantly_worse) == (False, False)
    report = make.report(
        settings=HeldOutSettings(rounds=2, evaluation_seeds=(7, 11, 23, 42)),
        seed_results=(
            better,
            make.seed_result(11),
            make.seed_result(23, **{"delta": -0.05, "delta_ci_low": -0.09, "delta_ci_high": -0.01}),
        ),
        skipped_seeds={42: "24 test groups with a hidden link, fewer than 30"},
    )
    assert (report.seeds_better, report.seeds_worse) == (1, 1)


def test_product_measures_reach_orphans_exactly_when_there_are_some() -> None:
    none = make.product(orphan_slot_share=0.0, orphan_page_share=0.0, orphans_reached=None)
    assert none.orphans_reached is None
    with pytest.raises(ValidationError, match="orphans_reached is None exactly"):
        make.product(orphan_page_share=0.0, orphan_slot_share=0.0)
    with pytest.raises(ValidationError, match="orphans_reached is None exactly"):
        make.product(orphans_reached=None)
    assert make.product(top_relevance=None).top_relevance is None


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"seed_results": (make.seed_result(7), make.seed_result(7))}, "distinct evaluation"),
        ({"seed_results": (make.seed_result(8),)}, "distinct evaluation"),
        ({"product_measures": (make.product(), make.product())}, "measured twice"),
        ({"product_measures": ()}, "product measures or the reason"),
        ({"seed_results": (make.seed_result(7),)}, "a result or a skip reason for every seed"),
        ({"skipped_seeds": {11: "too few"}}, "skipped seeds are evaluation seeds without a"),
        ({"skipped_seeds": {23: "too few"}}, "skipped seeds are evaluation seeds without a"),
        (
            {"seed_results": (make.seed_result(7),), "skipped_seeds": {11: " "}},
            "a skipped seed names its reason",
        ),
        ({"product_skipped_reason": "no anchor choices"}, "product measures or the reason"),
        ({"unlabelable_rows": -1}, "greater than or equal to 0"),
    ],
)
def test_a_ranker_report_holds_its_seeds_and_product_measures_once(
    fields: dict[str, object], match: str
) -> None:
    skipped_products = make.report(product_measures=(), product_skipped_reason="no anchor choices")
    assert skipped_products.product_skipped_reason == "no anchor choices"
    with pytest.raises(ValidationError, match=match):
        make.report(**fields)
    with pytest.raises(ValidationError, match="no seed results, skipped seeds or product"):
        make.skipped(seed_results=(make.seed_result(),))
    with pytest.raises(ValidationError, match="no seed results, skipped seeds or product"):
        make.skipped(skipped_seeds={7: "too few"})
    one_skipped = make.report(
        seed_results=(make.seed_result(7),), skipped_seeds={11: "no validation group"}
    )
    assert one_skipped.skipped_seeds == {11: "no validation group"}
