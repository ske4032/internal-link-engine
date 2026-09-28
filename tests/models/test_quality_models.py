"""Validators of the #74 quality models: each rejection beside the boundary that passes."""

from __future__ import annotations

import math

import pytest
import quality_factories as make
from pydantic import BaseModel, ValidationError

from linking_engine.models import (
    FeatureAuc,
    KeywordRung,
    QualityBaseline,
    QualityReport,
    RecallAtK,
)
from linking_engine.models.quality import KEYWORD_CHECKS

MODELS: tuple[type[BaseModel], ...] = (
    type(make.versions()),
    type(make.retrieval()),
    type(make.feature_signal()),
    type(make.scorer()),
    type(make.extractability()),
    type(make.anchor_match()),
    type(make.rank()),
    type(make.relevance()),
    type(make.keywords()),
    type(make.link_relevance()),
    type(make.coverage()),
    type(make.alert()),
    QualityBaseline,
    QualityReport,
    FeatureAuc,
    RecallAtK,
)


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
def test_every_quality_model_is_frozen_and_forbids_extras(model: type[BaseModel]) -> None:
    assert model.model_config.get("frozen") is True
    assert model.model_config.get("extra") == "forbid"


def test_a_report_round_trips_through_json() -> None:
    report = make.report(alerts=(make.alert(),), baseline_run_id="run-1")

    assert QualityReport.model_validate_json(report.model_dump_json()) == report


# ── retrieval ───────────────────────────────────────────────────────────────


def test_hidden_links_are_bounded_by_the_body_links_and_recoverable_by_hidden() -> None:
    assert make.retrieval(body_link_pairs=4, hidden=4, recoverable=4).hidden == 4
    with pytest.raises(ValidationError, match="recoverable <= hidden <= body_link_pairs"):
        make.retrieval(body_link_pairs=3, hidden=4, recoverable=1)
    with pytest.raises(ValidationError, match="recoverable <= hidden <= body_link_pairs"):
        make.retrieval(hidden=4, recoverable=5)


@pytest.mark.parametrize(
    "ks",
    [pytest.param((20, 10), id="descending"), pytest.param((10, 10), id="repeated")],
)
def test_recall_ks_are_unique_and_ascending(ks: tuple[int, int]) -> None:
    recall = tuple(RecallAtK(k=k, recall=0.5, random=0.1) for k in ks)
    with pytest.raises(ValidationError, match="unique and ascending"):
        make.retrieval(recall=recall)


def test_recall_never_falls_as_k_grows() -> None:
    flat = (RecallAtK(k=10, recall=0.5, random=0.1), RecallAtK(k=20, recall=0.5, random=0.2))
    assert make.retrieval(recall=flat).recall == flat
    falling = (RecallAtK(k=10, recall=0.5, random=0.1), RecallAtK(k=20, recall=0.4, random=0.2))
    with pytest.raises(ValidationError, match="must not fall"):
        make.retrieval(recall=falling)


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_the_hidden_share_is_strictly_between_zero_and_one(share: float) -> None:
    with pytest.raises(ValidationError, match="hide_share"):
        make.retrieval(hide_share=share)


# ── feature signal and scorer ───────────────────────────────────────────────


def test_some_candidate_pairs_are_not_hidden_links() -> None:
    assert make.feature_signal(pairs=4, positives=3).positives == 3
    with pytest.raises(ValidationError, match="other than hidden links"):
        make.feature_signal(pairs=3, positives=3)


def test_features_with_signal_counts_the_columns_at_least_the_margin_from_half() -> None:
    columns = (
        FeatureAuc(column="up", auc=0.56, coverage=1.0, ranker_auc=0.56),
        FeatureAuc(column="down", auc=0.44, coverage=1.0, ranker_auc=0.56),
        FeatureAuc(column="flat", auc=0.54, coverage=1.0, ranker_auc=0.54),
        FeatureAuc(column="none", auc=None, coverage=0.0, ranker_auc=0.5),
    )
    assert make.feature_signal(columns=columns, features_with_signal=2).features_with_signal == 2
    for wrong in (1, 3):
        with pytest.raises(ValidationError, match="count the columns with signal"):
            make.feature_signal(columns=columns, features_with_signal=wrong)


def test_a_column_is_listed_once() -> None:
    twice = (
        FeatureAuc(column="same_hub", auc=0.5, coverage=1.0, ranker_auc=0.5),
        FeatureAuc(column="same_hub", auc=0.5, coverage=1.0, ranker_auc=0.5),
    )
    with pytest.raises(ValidationError, match="duplicate columns"):
        make.feature_signal(columns=twice, features_with_signal=0)


def lifted(
    score: float, best: float, excl: float | None, best_excl: float, **fields: object
) -> object:
    values: dict[str, object] = {
        "score_auc": score,
        "best_feature_auc": best,
        "score_auc_lift": score - best,
        "link_derived_weight_share": 0.5 if excl is not None else 1.0,
        "score_auc_excl_link_counts": excl,
        "best_feature_auc_excl_link_counts": best_excl,
        "score_auc_lift_excl_link_counts": None if excl is None else excl - best_excl,
        **fields,
    }
    return make.scorer(**values)


def test_each_lift_is_a_score_over_the_best_feature_of_the_same_columns() -> None:
    assert lifted(0.7, 0.6, 0.62, 0.55)
    with pytest.raises(ValidationError, match="score_auc_lift must be"):
        lifted(0.7, 0.6, 0.62, 0.55, score_auc_lift=0.2)
    # Like for like: the excluded score against the excluded best, never the full score.
    with pytest.raises(ValidationError, match="score_auc_excl_link_counts - best_feature"):
        lifted(0.7, 0.6, 0.62, 0.55, score_auc_lift_excl_link_counts=0.7 - 0.55)


def test_the_excluded_score_is_missing_exactly_when_every_weight_is_link_derived() -> None:
    assert lifted(0.7, 0.6, None, 0.55).score_auc_excl_link_counts is None
    with pytest.raises(ValidationError, match="None exactly when all the weight"):
        lifted(0.7, 0.6, None, 0.55, link_derived_weight_share=0.99)
    with pytest.raises(ValidationError, match="None exactly when all the weight"):
        lifted(0.7, 0.6, 0.62, 0.55, link_derived_weight_share=1.0)
    with pytest.raises(ValidationError, match="set together"):
        lifted(0.7, 0.6, 0.62, 0.55, score_auc_lift_excl_link_counts=None)


def test_the_best_of_fewer_columns_never_beats_the_best_of_all() -> None:
    assert lifted(0.7, 0.6, 0.62, 0.6)
    with pytest.raises(ValidationError, match="cannot beat the best of all"):
        lifted(0.7, 0.6, 0.62, 0.65)


@pytest.mark.parametrize("field", ["hidden_histogram", "other_histogram"])
def test_scorer_histograms_have_twenty_non_negative_bins(field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        make.scorer(**{field: (0,) * 19})
    with pytest.raises(ValidationError, match="non-negative"):
        make.scorer(**{field: (-1,) + (0,) * 19})


# ── keywords ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("primary", "whole_set"),
    [("found_primary", "found_set"), ("words_primary", "words_set")],
)
def test_the_primary_keyword_never_matches_more_pairs_than_its_set(
    primary: str, whole_set: str
) -> None:
    assert make.extractability(**{primary: 0.75, whole_set: 0.75})
    with pytest.raises(ValidationError, match="cannot match more pairs"):
        make.extractability(**{primary: 0.875, whole_set: 0.75})


def test_the_best_rungs_add_up_to_the_pairs_found() -> None:
    assert make.extractability(exact_set=0.5, stemmed_set=0.25, stem_set_set=0.0)
    with pytest.raises(ValidationError, match="add up to found_set"):
        make.extractability(stem_set_set=0.25)


@pytest.mark.parametrize("threshold", [0.0, 1.01])
def test_the_stem_set_threshold_is_in_the_open_closed_unit_interval(threshold: float) -> None:
    assert make.extractability(stem_set_threshold=1.0)
    with pytest.raises(ValidationError, match="stem_set_threshold"):
        make.extractability(stem_set_threshold=threshold)


def test_the_primary_keyword_never_matches_more_anchors_than_its_set() -> None:
    assert make.anchor_match(primary=0.5, any_rank=0.5)
    with pytest.raises(ValidationError, match="cannot match more anchors"):
        make.anchor_match(primary=0.6, any_rank=0.5)


def test_rank_percentiles_do_not_decrease_and_cosines_may_be_negative() -> None:
    assert make.rank(mean=-0.2, p10=-0.5, p50=-0.2, p90=0.1).p10 == -0.5
    with pytest.raises(ValidationError, match="must not decrease"):
        make.rank(p10=0.7, p50=0.6, p90=0.8)
    with pytest.raises(ValidationError, match="p10"):
        make.rank(p10=-1.01)


def test_every_keyword_text_is_embedded_or_cached() -> None:
    assert make.relevance(texts=5, embedded=2, cached=3).cached == 3
    with pytest.raises(ValidationError, match="either embedded or cached"):
        make.relevance(texts=5, embedded=2, cached=2)


def test_origins_and_lengths_group_the_same_keywords_as_the_ranks_once_each() -> None:
    first = make.relevance().by_origin[0]
    with pytest.raises(ValidationError, match="group the same keywords"):
        make.relevance(by_origin=(first,))
    with pytest.raises(ValidationError, match="duplicate relevance groups"):
        make.relevance(by_origin=(first, first.model_copy(update={"keywords": 2})))
    assert make.relevance(by_length=(first.model_copy(update={"group": "1_2", "keywords": 5}),))


def test_relevance_ranks_are_unique_and_ascending() -> None:
    with pytest.raises(ValidationError, match="unique and ascending"):
        make.relevance(ranks=(make.rank(rank=2), make.rank(rank=1)))


def test_a_relevance_reason_is_given_exactly_when_relevance_does_not_apply() -> None:
    assert make.keywords(relevance=None, relevance_reason="no Voyage API key").relevance is None
    with pytest.raises(ValidationError, match="relevance_reason is set exactly"):
        make.keywords(relevance=None)
    with pytest.raises(ValidationError, match="relevance_reason is set exactly"):
        make.keywords(relevance_reason="no Voyage API key")
    with pytest.raises(ValidationError, match="relevance_reason"):
        make.keywords(relevance=None, relevance_reason="")


def test_rung_counts_add_up_to_the_resolved_pages_which_never_exceed_the_pages() -> None:
    assert make.keywords(pages=4, resolved=4, by_rung={KeywordRung.H1: 4}).resolved == 4
    with pytest.raises(ValidationError, match="add up to the resolved pages"):
        make.keywords(resolved=4, by_rung={KeywordRung.H1: 3})
    with pytest.raises(ValidationError, match="more resolved pages than pages"):
        make.keywords(pages=3, resolved=4, by_rung={KeywordRung.H1: 4})
    with pytest.raises(ValidationError, match="non-negative"):
        make.keywords(fallbacks_rejected={"generic": -1})


# ── link relevance, coverage, alerts, baseline ──────────────────────────────


def test_link_relevance_has_no_more_scores_than_links() -> None:
    assert make.link_relevance(links=25).links == 25
    with pytest.raises(ValidationError, match="more scores than links"):
        make.link_relevance(links=24)


def test_the_gsc_share_is_set_exactly_when_there_are_pairs() -> None:
    assert make.coverage(pairs=0, gsc_pair_share=None).gsc_pair_share is None
    with pytest.raises(ValidationError, match="exactly when there are pairs"):
        make.coverage(pairs=0, gsc_pair_share=0.0)
    with pytest.raises(ValidationError, match="exactly when there are pairs"):
        make.coverage(pairs=3, gsc_pair_share=None)


def test_an_alert_has_no_change_only_when_a_relative_band_leaves_zero() -> None:
    assert make.alert(previous=0.0, current=0.3, change=None).change is None
    with pytest.raises(ValidationError, match="relative band from 0"):
        make.alert(previous=0.5, change=None)
    with pytest.raises(ValidationError, match="relative band from 0"):
        make.alert(previous=0.0, change=None, relative=False)
    with pytest.raises(ValidationError, match="band"):
        make.alert(band=0.0)
    with pytest.raises(ValidationError, match="current"):
        make.alert(current=math.nan)


def test_a_baseline_holds_only_finite_metrics() -> None:
    assert QualityBaseline(run_id="run-1", metrics={"recall_at_10": 0.5}).metrics
    for value in (math.nan, math.inf):
        with pytest.raises(ValidationError, match="finite"):
            QualityBaseline(run_id="run-1", metrics={"recall_at_10": value})


# ── the report ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "check", ["retrieval", "feature_signal", "scorer", "keywords", "link_relevance"]
)
def test_a_section_is_none_exactly_when_it_is_not_applicable(check: str) -> None:
    skipped = make.report(not_applicable=(check,))
    assert getattr(skipped, check) is None
    with pytest.raises(ValidationError, match=f"{check} is not applicable exactly"):
        make.report(**{check: None})
    with pytest.raises(ValidationError, match=f"{check} is not applicable exactly"):
        make.report(not_applicable=(check,), **{check: getattr(make.report(), check)})


@pytest.mark.parametrize("check", KEYWORD_CHECKS)
def test_a_keyword_sub_check_is_none_exactly_when_it_is_not_applicable(check: str) -> None:
    assert make.report(not_applicable=(check,)).keywords is not None
    with pytest.raises(ValidationError, match=f"{check} is not applicable exactly"):
        make.report(not_applicable=(check,), keywords=make.keywords())


def test_keyword_sub_checks_are_listed_only_when_keywords_apply() -> None:
    with pytest.raises(ValidationError, match="only when keywords apply"):
        make.report(not_applicable=("keywords", "keyword_relevance"))


def test_a_check_is_listed_once() -> None:
    with pytest.raises(ValidationError, match="twice"):
        make.report(not_applicable=("scorer", "scorer"))


def test_alerts_need_a_baseline_and_each_metric_alerts_once() -> None:
    first = make.alert()
    assert make.report(alerts=(first,), baseline_run_id="run-1").alerts == (first,)
    with pytest.raises(ValidationError, match="need a baseline run"):
        make.report(alerts=(first,))
    with pytest.raises(ValidationError, match="alerts twice"):
        make.report(alerts=(first, first), baseline_run_id="run-1")


def test_an_unknown_check_name_is_refused() -> None:
    with pytest.raises(ValidationError, match="not_applicable"):
        make.report(not_applicable=("ranker",))  # type: ignore[arg-type]
