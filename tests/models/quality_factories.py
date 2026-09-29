"""Valid #74 quality sections and reports, each overridable one field at a time."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from linking_engine.models import (
    AnchorMatchCheck,
    CoverageCheck,
    FeatureAuc,
    FeatureSignalCheck,
    KeywordCheck,
    KeywordExtractability,
    KeywordRelevance,
    KeywordRung,
    LinkRelevanceCheck,
    QualityAlert,
    QualityReport,
    QualityVersions,
    RankRelevance,
    RecallAtK,
    RelevanceGroup,
    RetrievalCheck,
    ScoreDistribution,
    ScorerCheck,
    SourceExtractability,
)

if TYPE_CHECKING:
    from linking_engine.models import CheckName

WHEN = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
# Low-entropy stand-ins for a commit sha, a code digest and a weights hash.
SHA = "a" * 40
DIGEST = "d" * 64
WEIGHTS_HASH = "f" * 64
EMPTY_BINS = (0,) * 20
CONTEXT_BINS = (0,) * 10 + (1, 2, 3, 4, 5, 4, 3, 2, 1, 0)


def versions(**fields: object) -> QualityVersions:
    values: dict[str, object] = {
        "git_sha": SHA,
        "feature_digest": DIGEST,
        "weights_version": "baseline-1",
        "weights_hash": WEIGHTS_HASH,
        **fields,
    }
    return QualityVersions.model_validate(values)


def retrieval(**fields: object) -> RetrievalCheck:
    values: dict[str, object] = {
        "hide_share": 0.1,
        "seed": 42,
        "body_link_pairs": 40,
        "hidden": 4,
        "recoverable": 3,
        "candidates": 120,
        "recall": (
            RecallAtK(k=10, recall=0.5, random=0.25),
            RecallAtK(k=20, recall=0.75, random=0.5),
            RecallAtK(k=50, recall=1.0, random=1.0),
        ),
        **fields,
    }
    return RetrievalCheck.model_validate(values)


def feature_signal(**fields: object) -> FeatureSignalCheck:
    values: dict[str, object] = {
        "pairs": 120,
        "positives": 3,
        "margin": 0.05,
        "columns": (
            FeatureAuc(column="content_cosine", auc=0.9, coverage=1.0, ranker_auc=0.9),
            FeatureAuc(column="same_hub", auc=0.52, coverage=1.0, ranker_auc=0.52),
            FeatureAuc(column="target_ctr_gap", auc=None, coverage=0.0, ranker_auc=0.5),
        ),
        "features_with_signal": 1,
        **fields,
    }
    return FeatureSignalCheck.model_validate(values)


def scorer(**fields: object) -> ScorerCheck:
    values: dict[str, object] = {
        "weights_version": "baseline-1",
        "score_auc": 0.8,
        "best_feature": "content_cosine",
        "best_feature_auc": 0.9,
        "score_auc_lift": 0.8 - 0.9,
        "link_derived_weight_share": 0.5,
        "score_auc_excl_link_counts": 0.7,
        "best_feature_excl_link_counts": "content_cosine",
        "best_feature_auc_excl_link_counts": 0.9,
        "score_auc_lift_excl_link_counts": 0.7 - 0.9,
        "hidden_histogram": (0,) * 19 + (3,),
        "other_histogram": (6,) * 19 + (3,),
        **fields,
    }
    return ScorerCheck.model_validate(values)


def source(**fields: object) -> SourceExtractability:
    values: dict[str, object] = {
        "pairs": 4,
        "found_primary": 0.25,
        "found_set": 0.5,
        "words_primary": 0.5,
        **fields,
    }
    return SourceExtractability.model_validate(values)


def extractability(**fields: object) -> KeywordExtractability:
    values: dict[str, object] = {
        "pairs": 8,
        "found_primary": 0.5,
        "found_set": 0.75,
        "exact_set": 0.375,
        "stemmed_set": 0.25,
        "stem_set_set": 0.125,
        "words_primary": 0.5,
        "words_set": 0.75,
        "stem_set_threshold": 0.6,
        # Two of the eight pairs have a target that resolved no keyword.
        "by_source": {
            KeywordRung.H1: source(),
            KeywordRung.STRATEGIC: source(
                pairs=2, found_primary=0.5, found_set=1.0, words_primary=1.0
            ),
        },
        **fields,
    }
    return KeywordExtractability.model_validate(values)


def anchor_match(**fields: object) -> AnchorMatchCheck:
    values: dict[str, object] = {"anchors": 4, "primary": 0.25, "any_rank": 0.5, **fields}
    return AnchorMatchCheck.model_validate(values)


def rank(**fields: object) -> RankRelevance:
    values: dict[str, object] = {
        "rank": 1,
        "keywords": 3,
        "mean": 0.6,
        "p10": 0.5,
        "p50": 0.6,
        "p90": 0.7,
        **fields,
    }
    return RankRelevance.model_validate(values)


def relevance(**fields: object) -> KeywordRelevance:
    values: dict[str, object] = {
        "ranks": (rank(), rank(rank=2, keywords=2, mean=0.4, p10=0.3, p50=0.4, p90=0.5)),
        "by_origin": (
            RelevanceGroup(group="primary_h1", keywords=3, mean=0.6, p50=0.6),
            RelevanceGroup(group="secondary_gsc_observed", keywords=2, mean=0.4, p50=0.4),
        ),
        "by_length": (
            RelevanceGroup(group="1_2", keywords=4, mean=0.55, p50=0.6),
            RelevanceGroup(group="3_4", keywords=1, mean=0.3, p50=0.3),
        ),
        "texts": 5,
        "embedded": 5,
        "cached": 0,
        "api_tokens": 12,
        **fields,
    }
    return KeywordRelevance.model_validate(values)


def keywords(**fields: object) -> KeywordCheck:
    values: dict[str, object] = {
        "pages": 10,
        "resolved": 4,
        "by_rung": {KeywordRung.STRATEGIC: 1, KeywordRung.H1: 3},
        "fallbacks_rejected": {"generic": 2, "repeated": 1},
        "unique_share": 0.75,
        "extractability": extractability(),
        "anchors": anchor_match(),
        "relevance": relevance(),
        **fields,
    }
    return KeywordCheck.model_validate(values)


def distribution(histogram: tuple[int, ...] = CONTEXT_BINS, **fields: object) -> ScoreDistribution:
    values: dict[str, object] = {
        "count": sum(histogram),
        "mean": 0.7,
        "p10": 0.55,
        "p25": 0.6,
        "p50": 0.7,
        "p75": 0.8,
        "p90": 0.85,
        "histogram": histogram,
        **fields,
    }
    return ScoreDistribution.model_validate(values)


def link_relevance(**fields: object) -> LinkRelevanceCheck:
    values: dict[str, object] = {
        "links": 30,
        "context": distribution(split=0.62, low_share=0.3),
        "anchor": distribution(),
        **fields,
    }
    return LinkRelevanceCheck.model_validate(values)


def coverage(**fields: object) -> CoverageCheck:
    values: dict[str, object] = {
        "pairs": 120,
        "gsc_pair_share": 0.25,
        "keyword_page_share": 0.4,
        "all_null_columns": ("context_relevance", "anchor_target_fit"),
        "constant_columns": ("has_gsc_data",),
        **fields,
    }
    return CoverageCheck.model_validate(values)


def alert(**fields: object) -> QualityAlert:
    values: dict[str, object] = {
        "metric": "recall_at_10",
        "previous": 0.5,
        "current": 0.25,
        "change": -0.5,
        "band": 0.2,
        "relative": True,
        **fields,
    }
    return QualityAlert.model_validate(values)


def report(
    tenant: str = "acme", *, not_applicable: tuple[CheckName, ...] = (), **fields: object
) -> QualityReport:
    """A full report; every check named in ``not_applicable`` has its section dropped."""
    sections: dict[str, object] = {
        "retrieval": retrieval(),
        "feature_signal": feature_signal(),
        "scorer": scorer(),
        "link_relevance": link_relevance(),
    }
    parts = {
        "keyword_uniqueness": "unique_share",
        "keyword_extractability": "extractability",
        "anchor_match": "anchors",
        "keyword_relevance": "relevance",
    }
    dropped: dict[str, object] = {
        field: None for name, field in parts.items() if name in not_applicable
    }
    if "keyword_relevance" in not_applicable:
        dropped["relevance_reason"] = "no Voyage API key"
    sections["keywords"] = None if "keywords" in not_applicable else keywords(**dropped)
    for name in not_applicable:
        if name in sections:
            sections[name] = None
    values: dict[str, object] = {
        "tenant_id": tenant,
        "versions": versions(),
        **sections,
        "coverage": coverage(),
        "not_applicable": not_applicable,
        "seconds": 4.5,
        "finished_at": WHEN,
        **fields,
    }
    return QualityReport.model_validate(values)
