"""Valid #24-#27 ranker models, each overridable one field at a time."""

from __future__ import annotations

from datetime import UTC, datetime

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

WHEN = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
# Low-entropy stand-ins for a commit sha and a code digest.
SHA = "a" * 40
DIGEST = "d" * 64
COLUMNS = ("content_cosine", "same_hub", "target_is_orphan")


def round_summary(**fields: object) -> RoundSummary:
    values: dict[str, object] = {
        "round": 0,
        "hidden": 40,
        "recoverable": 36,
        "pairs": 3000,
        "positives": 36,
        "groups_with_positive": 30,
        "positive_placement_share": 0.9,
        "negative_placement_share": 0.05,
        **fields,
    }
    return RoundSummary.model_validate(values)


def metrics(scorer: ScorerName = ScorerName.LEARNED, **fields: object) -> RankingMetrics:
    values: dict[str, object] = {
        "scorer": scorer,
        "ndcg_at_10": 0.62,
        "ci_low": 0.55,
        "ci_high": 0.69,
        "precision_at_5": 0.3,
        "groups": 40,
        "groups_with_positive_share": 0.5,
        "n_labelled_pairs": 52,
        "per_round": {0: 0.6, 1: 0.64},
        "histogram": (1, 2, 3, 4, 5, 6, 7, 6, 4, 2),
        **fields,
    }
    return RankingMetrics.model_validate(values)


def importance(**fields: object) -> tuple[ImportanceEntry, ...]:
    shares = {"content_cosine": 0.7, "same_hub": 0.2, "target_is_orphan": 0.1, **fields}
    return tuple(
        ImportanceEntry(column=column, gain=100 * share, gain_share=share)
        for column, share in shares.items()
    )


def promotion(**fields: object) -> PromotionDecision:
    values: dict[str, object] = {
        "rival": ScorerName.BASELINE,
        "holder_version": None,
        "delta": 0.08,
        "delta_ci_low": 0.03,
        "delta_ci_high": 0.12,
        "would_promote": True,
        "promoted": False,
        "reason": "better than the baseline; promotion is not allowed here",
        **fields,
    }
    return PromotionDecision.model_validate(values)


def seed_result(seed: int = 7, **fields: object) -> SeedResult:
    values: dict[str, object] = {
        "seed": seed,
        "test_pages": 12,
        "learned": (0.62, 0.55, 0.69),
        "plain": (0.6, 0.52, 0.68),
        "baseline": (0.4, 0.3, 0.5),
        "delta": 0.02,
        "delta_ci_low": -0.01,
        "delta_ci_high": 0.05,
        **fields,
    }
    return SeedResult.model_validate(values)


def product(scorer: ScorerName = ScorerName.LEARNED, **fields: object) -> ProductMeasures:
    values: dict[str, object] = {
        "scorer": scorer,
        "k": 10,
        "top_relevance": 0.71,
        "same_hub_share": 0.8,
        "orphan_slot_share": 0.09,
        "orphan_page_share": 0.08,
        "orphans_reached": 0.95,
        **fields,
    }
    return ProductMeasures.model_validate(values)


def report(tenant_id: str = "acme", **fields: object) -> RankerReport:
    values: dict[str, object] = {
        "tenant_id": tenant_id,
        "corpus_pages": 80,
        "body_links": 432,
        "feature_set_version": DIGEST,
        "git_sha": SHA,
        "settings": HeldOutSettings(rounds=2, evaluation_seeds=(7, 11)),
        "params": RankerParams(),
        "rounds": (round_summary(), round_summary(round=1, positives=30, recoverable=30)),
        "columns": COLUMNS,
        "excluded_columns": {"target_crawl_depth": "stored BFS over links"},
        "train_groups": 120,
        "valid_groups": 15,
        "test_groups": 40,
        "positives": 66,
        "best_iteration": 12,
        "metrics": (
            metrics(),
            metrics(ScorerName.LEARNED_EXCL_LINK_COUNTS, ndcg_at_10=0.5, ci_low=0.4),
            metrics(ScorerName.BASELINE, ndcg_at_10=0.4, ci_low=0.3, ci_high=0.5),
        ),
        "importance": importance(),
        "dominant_feature": "content_cosine",
        "promotion": promotion(),
        "model_version": None,
        "skipped_reason": None,
        "unlabelable_targets": 6,
        "unlabelable_rows": 480,
        "seed_results": (
            seed_result(7),
            seed_result(11, delta=-0.04, delta_ci_low=-0.07, delta_ci_high=-0.01),
        ),
        "product_measures": (
            product(),
            product(ScorerName.PLAIN, top_relevance=0.65, orphan_slot_share=0.03),
            product(ScorerName.BASELINE, top_relevance=0.6, same_hub_share=0.7),
        ),
        "product_skipped_reason": None,
        "seconds": 12.5,
        "finished_at": WHEN,
        **fields,
    }
    return RankerReport.model_validate(values)


def skipped(tenant_id: str = "acme", **fields: object) -> RankerReport:
    values: dict[str, object] = {
        "best_iteration": None,
        "metrics": (),
        "importance": (),
        "dominant_feature": None,
        "promotion": None,
        "seed_results": (),
        "product_measures": (),
        "skipped_reason": "12 training groups with a hidden link, 100 needed",
        **fields,
    }
    return report(tenant_id, **values)


def rank_report(**fields: object) -> RankReport:
    values: dict[str, object] = {
        "tenant_id": "acme",
        "pairs": 3000,
        "scorer": ScorerName.LEARNED,
        "model_version": "3",
        "fallback_reason": None,
        "seconds": 1.5,
        **fields,
    }
    return RankReport.model_validate(values)
