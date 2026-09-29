"""MLflow tracking for pipeline runs: one experiment per tenant, one run per execution."""

from __future__ import annotations

import csv
import io
import math
import time
from importlib.metadata import version
from typing import TYPE_CHECKING

import mlflow
from mlflow.entities import Metric

from linking_engine.anchor.extraction import (
    MAX_INNER_STOP_WORDS,
    MAX_SPAN,
    MIN_SHARED_STEMS,
    MIN_SPAN,
    SNOWBALL,
)
from linking_engine.anchor.keywords import (
    BRAND_SUFFIX_SHARE,
    MAX_KEYWORD_TOKENS,
    MAX_SECONDARY_QUERIES,
    MIN_QUERY_IMPRESSIONS,
    REPEATED_FALLBACK_PAGES,
)
from linking_engine.anchor.scoring import (
    AWKWARD_FLOOR,
    COSINE_SHARE,
    DIVERSITY_WEIGHT,
    KEYWORD_WEIGHT,
    LENGTH_WEIGHT,
    PROFILE_BONUS,
    SECONDARY_WEIGHT,
    SEMANTIC_WEIGHT,
    STEM_SHARE,
)
from linking_engine.anchor.semantic import (
    DEFAULT_SEMANTIC_THRESHOLD,
    MAX_PHRASES_PER_SENTENCE,
    MIN_NEGATIVES,
    NEGATIVE_SAMPLE,
    THRESHOLD_BOUNDS,
    TOP_SENTENCES,
)
from linking_engine.audit.links import (
    ALIGNMENT_COSINE_SHARE,
    ALIGNMENT_JACCARD_SHARE,
    ALIGNMENT_WEIGHT,
    CONTEXT_WEIGHT,
    FENCE_IQRS,
    FIT_WEIGHT,
    GENERIC_CAP,
    MIN_IQR,
    OVER_OPTIMISED_MIN,
    OVER_OPTIMISED_SHARE,
)
from linking_engine.audit.relevance import MIN_MODE_GAP, MIN_SPLIT_SCORES, SPLIT_SEED
from linking_engine.discovery.bridges import (
    ALTERNATIVES,
    COSINE_WEIGHT,
    DENSITY_WEIGHT,
    FLOOR_SHARE,
    JACCARD_WEIGHT,
    NEAREST_HUBS,
    RELEVANCE_DECIMALS,
    SHARED_QUERIES,
    TOP_GAP_PAIRS,
)
from linking_engine.discovery.candidates import PER_TARGET
from linking_engine.discovery.features import POSITION_BANDS
from linking_engine.graph.algorithms import (
    DAMPING,
    HUB_MATCH_SIMILARITY,
    HUB_MIN_CLUSTER_SIZE,
    HUB_MIN_SAMPLES,
    KNN_NEIGHBOURS,
    MIN_PILLAR_COMMUNITY,
    PROJECTION_EDGE_BUDGET,
    RESOLUTION,
    SEED,
    STABILITY_SEEDS,
)
from linking_engine.gsc import MIN_CURVE_IMPRESSIONS, MIN_CURVE_ROWS
from linking_engine.ml.quality import (
    ALERT_BAND,
    ANCHOR_JACCARD,
    HIDE_SEED,
    HIDE_SHARE,
    LINK_DERIVED_COLUMNS,
    QUALITY_STAGE,
    RECALL_KS,
    SIGNAL_MARGIN,
    has_signal,
    quality_metrics,
)
from linking_engine.models import IssueFlag, QualityBaseline
from linking_engine.models.anchors import SCORE_HISTOGRAM_BINS as ANCHOR_SCORE_BINS
from linking_engine.models.anchors import SENTENCE_INDEX_BINS, UNANCHORED_ADVICE
from linking_engine.models.audit import AUDIT_VERDICTS
from linking_engine.models.relevance import HISTOGRAM_BINS
from linking_engine.models.scoring import SCORE_HISTOGRAM_BINS
from linking_engine.urls import PAGINATION_PARAMS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.models import (
        AnchorReport,
        AnchorSelectionReport,
        BridgeReport,
        CandidateReport,
        CandidateSet,
        CentralityReport,
        CommunityReport,
        DuplicateReport,
        FeatureReport,
        HubPair,
        HubReport,
        KeywordReport,
        LinkAuditReport,
        LinkRelevanceReport,
        QualityReport,
        ScoreDistribution,
        ScoreReport,
    )


# MLflow 3's UI opens an experiment without a kind in its GenAI (tracing) view, which hides the
# metrics, charts and tables pipeline runs log.
EXPERIMENT_KIND_TAG = "mlflow.experimentKind"
EXPERIMENT_KIND = "custom_model_development"


def analytics_experiment(tenant_id: str) -> str:
    return f"analytics-{tenant_id}"


def use_analytics_experiment(tenant_id: str) -> None:
    """Make the tenant's experiment the active one, tagged to open in the model-training view."""
    experiment = mlflow.set_experiment(analytics_experiment(tenant_id))
    if experiment.tags.get(EXPERIMENT_KIND_TAG) != EXPERIMENT_KIND:
        mlflow.set_experiment_tag(EXPERIMENT_KIND_TAG, EXPERIMENT_KIND)


def analytics_metrics(
    centrality: CentralityReport, communities: CommunityReport, hubs: HubReport
) -> dict[str, float]:
    """Every numeric result of the run, flat; values that could not be computed are left out."""
    metrics: dict[str, float] = {
        f"centrality_{name}": float(value)
        for name, value in centrality.model_dump(exclude={"tenant_id"}).items()
    }
    for name, found in (
        ("link", communities.link),
        ("keyword", communities.keyword),
        ("content", communities.content),
    ):
        metrics.update(
            {
                f"{name}_{field}": float(value)
                for field, value in found.model_dump().items()
                if value is not None
            }
        )
    summary = communities.model_dump(
        exclude={"tenant_id", "link", "keyword", "content", "orphan_labels"}
    )
    metrics.update({name: float(value) for name, value in summary.items() if value is not None})
    metrics.update(
        {
            f"orphans_{label.value.lower()}": float(count)
            for label, count in communities.orphan_labels.items()
        }
    )
    metrics.update(
        {
            f"hub_{name}": float(value)
            for name, value in hubs.model_dump(
                exclude={"tenant_id", "section_pages", "section_noise"}
            ).items()
            if value is not None
        }
    )
    return metrics


def log_analytics(
    centrality: CentralityReport, communities: CommunityReport, hubs: HubReport, summary: str
) -> str:
    """Log one graph analytics run with its description; returns the MLflow run id."""
    tenant_id = communities.tenant_id
    if {centrality.tenant_id, hubs.tenant_id} != {tenant_id}:
        raise ValueError("all reports must come from the same tenant")
    use_analytics_experiment(tenant_id)
    with mlflow.start_run(
        run_name="graph analytics",
        tags={
            "tenant_id": tenant_id,
            "kind": "pipeline",
            "stage": "graph-analytics",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_params(
            {
                "damping": DAMPING,
                "resolution": RESOLUTION,
                "seed": SEED,
                "stability_seeds": ",".join(map(str, STABILITY_SEEDS)),
                "knn_neighbours": KNN_NEIGHBOURS,
                "projection_edge_budget": PROJECTION_EDGE_BUDGET,
                "min_pillar_community": MIN_PILLAR_COMMUNITY,
                "hub_min_cluster_size": HUB_MIN_CLUSTER_SIZE,
                "hub_min_samples": HUB_MIN_SAMPLES,
                "hub_match_similarity": HUB_MATCH_SIMILARITY,
                "igraph": version("igraph"),
                "leidenalg": version("leidenalg"),
                "hdbscan": version("hdbscan"),
            }
        )
        mlflow.log_metrics(analytics_metrics(centrality, communities, hubs))
        mlflow.log_dict(
            {
                "centrality": centrality.model_dump(mode="json"),
                "communities": communities.model_dump(mode="json"),
                "hubs": hubs.model_dump(mode="json"),
            },
            "report.json",
        )
        mlflow.log_text(summary, "summary.md")
        return str(run.info.run_id)


_CANDIDATE_PARAMS = ("index", "per_target", "chunk_size")
# Finished quality-eval runs looked at for the tenant's latest.
_BASELINE_CANDIDATES = 10


def candidate_metrics(report: CandidateReport) -> dict[str, float]:
    """Every numeric result of the run, flat; values that could not be computed are left out."""
    return {
        name: float(value)
        for name, value in report.model_dump(
            exclude={"tenant_id", "finished_at", *_CANDIDATE_PARAMS}
        ).items()
        if value is not None
    }


def candidate_table(found: CandidateSet) -> str:
    """One CSV row per target: its candidates, the eligible and linked pages, and the first
    and last kept similarity (empty without candidates)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "target_url",
            "candidates",
            "eligible",
            "linked",
            "linked_nearer",
            "best_similarity",
            "last_similarity",
        )
    )
    writer.writerows(
        (
            target.target_url,
            len(target.sources),
            target.eligible,
            target.linked,
            target.linked_nearer,
            target.similarities[0] if target.similarities else "",
            target.similarities[-1] if target.similarities else "",
        )
        for target in found.targets
    )
    return buffer.getvalue()


def log_candidates(found: CandidateSet, summary: str) -> str:
    """Log one candidate retrieval run with its description; returns the MLflow run id."""
    report = found.report
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="candidate retrieval",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "candidate-retrieval",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_params(report.model_dump(include=set(_CANDIDATE_PARAMS)))
        mlflow.log_metrics(candidate_metrics(report))
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        mlflow.log_text(candidate_table(found), "targets.csv")
        return str(run.info.run_id)


def keyword_metrics(report: KeywordReport) -> dict[str, float]:
    """Counts of the run, flat: per rung, per edge source, per page language and per
    rejected fallback reason."""
    metrics: dict[str, float] = {
        "pages": float(report.pages),
        "resolved": float(report.resolved),
        "gsc_enabled": float(report.gsc_enabled),
        "gsc_rows": float(report.gsc_rows),
        "gsc_rejected": float(report.gsc_rejected),
        "skipped_rows": float(report.skipped_rows),
        "seconds": report.seconds,
    }
    metrics.update({f"rung_{rung.value.lower()}": float(n) for rung, n in report.by_rung.items()})
    metrics.update(
        {f"edges_{source.value.lower()}": float(n) for source, n in report.edges_written.items()}
    )
    metrics.update(
        {
            f"stale_{source.value.lower()}": float(n)
            for source, n in report.stale_edges_deleted.items()
        }
    )
    metrics.update({f"pages_{language}": float(n) for language, n in report.by_language.items()})
    metrics.update(
        {f"rejected_{reason}": float(n) for reason, n in report.fallbacks_rejected.items()}
    )
    metrics["long_fallbacks"] = float(report.long_fallbacks)
    metrics["secondary_keywords"] = float(report.secondary_keywords)
    metrics["pages_with_secondaries"] = float(report.pages_with_secondaries)
    return metrics


def log_keywords(report: KeywordReport, summary: str) -> str:
    """Log one keyword resolution run with its description; returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="keyword resolution",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "resolve-keywords",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_params(
            {
                "min_curve_rows": MIN_CURVE_ROWS,
                "min_curve_impressions": MIN_CURVE_IMPRESSIONS,
                "min_query_impressions": MIN_QUERY_IMPRESSIONS,
                "brand_suffix_share": BRAND_SUFFIX_SHARE,
                "repeated_fallback_pages": REPEATED_FALLBACK_PAGES,
                "max_keyword_tokens": MAX_KEYWORD_TOKENS,
                "max_secondary_queries": MAX_SECONDARY_QUERIES,
            }
        )
        mlflow.log_metrics(keyword_metrics(report))
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        return str(run.info.run_id)


def feature_metrics(report: FeatureReport) -> dict[str, float]:
    """Counts of the run and every column's null share, flat; the GSC share is left out
    without pairs."""
    metrics: dict[str, float] = {
        "pairs": float(report.pairs),
        "chunks": float(report.chunks),
        "columns": float(len(report.columns)),
        "all_null_columns": float(len(report.all_null_columns)),
        "constant_columns": float(len(report.constant_columns)),
        "seconds": report.seconds,
    }
    if report.has_gsc_data_share is not None:
        metrics["has_gsc_data_share"] = report.has_gsc_data_share
    metrics.update({f"null_share_{name}": share for name, share in report.null_share.items()})
    return metrics


def log_features(report: FeatureReport, summary: str) -> str:
    """Log one feature assembly run with its description and column order, never the matrix;
    returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="feature assembly",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "feature-assembly",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_params(
            {
                "cache_key": report.cache_key,
                "cache_hit": report.cache_hit,
                "position_bands": ",".join(map(str, POSITION_BANDS)),
            }
        )
        mlflow.log_metrics(feature_metrics(report))
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        mlflow.log_dict({"feature_columns": list(report.columns)}, "columns.json")
        return str(run.info.run_id)


def duplicate_metrics(report: DuplicateReport) -> dict[str, float]:
    """Counts of the run, flat."""
    return {
        "groups": float(len(report.groups)),
        "pages_in_groups": float(report.pages_in_groups),
        "non_canonical": float(report.non_canonical),
        "largest_group": float(report.largest_group),
        "seconds": report.seconds,
    }


def log_duplicates(report: DuplicateReport, summary: str) -> str:
    """Log one duplicate grouping run with its description and every group's canonical and
    copies; returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="duplicate pages",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "duplicates",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_metrics(duplicate_metrics(report))
        mlflow.log_dict(
            {**report.model_dump(mode="json", exclude={"groups"}), "groups": len(report.groups)},
            "report.json",
        )
        mlflow.log_text(summary, "summary.md")
        mlflow.log_dict(
            {"groups": [group.model_dump(mode="json") for group in report.groups]}, "groups.json"
        )
        return str(run.info.run_id)


def score_metrics(report: ScoreReport) -> dict[str, float]:
    """Counts, score percentiles, missing shares and top contributors of the run, flat;
    percentiles are left out without pairs."""
    metrics: dict[str, float] = {"pairs": float(report.pairs), "seconds": report.seconds}
    metrics.update({f"tier_{tier}": float(count) for tier, count in report.tiers.items()})
    for name in ("score_p10", "score_p50", "score_p90"):
        value = getattr(report, name)
        if value is not None:
            metrics[name] = float(value)
    metrics.update({f"missing_share_{name}": share for name, share in report.missing_share.items()})
    metrics.update(
        {f"top_contributor_{name}": float(count) for name, count in report.top_contributors.items()}
    )
    return metrics


def score_tables(report: ScoreReport) -> dict[str, dict[str, list[object]]]:
    """The run's histogram, tiers and weights as MLflow tables, keyed by artifact file."""
    width = 100 / SCORE_HISTOGRAM_BINS
    first, second = report.weights.tier_shares
    targets = (first, second, 1 - first - second)
    return {
        "score_histogram.json": {
            "bin": list(range(SCORE_HISTOGRAM_BINS)),
            "low": [round(i * width, 6) for i in range(SCORE_HISTOGRAM_BINS)],
            "high": [round((i + 1) * width, 6) for i in range(SCORE_HISTOGRAM_BINS)],
            "count": list(report.score_histogram),
        },
        "tiers.json": {
            "tier": [1, 2, 3],
            "pairs": [report.tiers.get(tier, 0) for tier in (1, 2, 3)],
            "share": [
                report.tiers.get(tier, 0) / report.pairs if report.pairs else 0.0
                for tier in (1, 2, 3)
            ],
            "target_share": list(targets),
        },
        "weights.json": {
            "column": [f.column for f in report.weights.features],
            "weight": [f.weight for f in report.weights.features],
            "direction": [f.direction for f in report.weights.features],
            "normalisation": [f.normalisation for f in report.weights.features],
            "missing_share": [report.missing_share.get(f.column) for f in report.weights.features],
            "top_contributor": [
                report.top_contributors.get(f.column, 0) for f in report.weights.features
            ],
        },
    }


def log_scores(report: ScoreReport, summary: str) -> str:
    """Log one scoring run from its report: summary metrics, the score histogram as the
    step-indexed metric ``score_hist`` (step = bin), and tables; returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="baseline scoring",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "score-pairs",
            "mlflow.note.content": summary,
        },
    ) as run:
        run_id = str(run.info.run_id)
        mlflow.log_params(
            {
                "weights_version": report.weights.version,
                "weights_hash": report.weights_hash,
                "feature_cache_key": report.feature_cache_key,
                "tier_shares": ",".join(map(str, report.weights.tier_shares)),
            }
        )
        mlflow.log_metrics(score_metrics(report))
        now = int(time.time() * 1000)
        mlflow.MlflowClient().log_batch(
            run_id,
            metrics=[
                Metric("score_hist", float(count), now, step)
                for step, count in enumerate(report.score_histogram)
            ],
        )
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        for artifact, table in score_tables(report).items():
            mlflow.log_table(table, artifact)
        return run_id


_RELEVANCE_COUNTS = ("links", "scored", "generic_anchors", "without_anchor_vector", "seconds")


def _relevance_scores(report: LinkRelevanceReport) -> dict[str, ScoreDistribution]:
    return {
        name: found
        for name, found in (
            ("context_relevance", report.context),
            ("anchor_target_fit", report.anchor),
        )
        if found is not None
    }


def link_relevance_metrics(report: LinkRelevanceReport) -> dict[str, float]:
    """Counts, runtime and each score's statistics and split, flat; absent values left out."""
    metrics = {name: float(getattr(report, name)) for name in _RELEVANCE_COUNTS}
    for name, found in _relevance_scores(report).items():
        metrics.update(
            {
                f"{name}_{field}": float(value)
                for field, value in found.model_dump(exclude={"histogram"}).items()
                if value is not None
            }
        )
    return metrics


def relevance_table(report: LinkRelevanceReport) -> dict[str, list[object]]:
    """Both scores' histograms as one MLflow table, a row per score and bin over [0, 1]."""
    width = 1 / HISTOGRAM_BINS
    scores = _relevance_scores(report)
    return {
        "score": [name for name in scores for _ in range(HISTOGRAM_BINS)],
        "bin": [step for _ in scores for step in range(HISTOGRAM_BINS)],
        "low": [round(step * width, 6) for _ in scores for step in range(HISTOGRAM_BINS)],
        "high": [round((step + 1) * width, 6) for _ in scores for step in range(HISTOGRAM_BINS)],
        "count": [count for found in scores.values() for count in found.histogram],
    }


def log_link_relevance(report: LinkRelevanceReport, summary: str) -> str:
    """Log one link relevance run from its report: statistics and splits as metrics, each
    score's histogram as the step-indexed metric ``<score>_hist`` (step = bin) and as a table;
    returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="link relevance",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "score-links",
            "mlflow.note.content": summary,
        },
    ) as run:
        run_id = str(run.info.run_id)
        mlflow.log_params(
            {
                "min_split_scores": MIN_SPLIT_SCORES,
                "min_mode_gap": MIN_MODE_GAP,
                "split_seed": SPLIT_SEED,
                "histogram_bins": HISTOGRAM_BINS,
            }
        )
        mlflow.log_metrics(link_relevance_metrics(report))
        scores = _relevance_scores(report)
        if scores:
            now = int(time.time() * 1000)
            mlflow.MlflowClient().log_batch(
                run_id,
                metrics=[
                    Metric(f"{name}_hist", float(count), now, step)
                    for name, found in scores.items()
                    for step, count in enumerate(found.histogram)
                ],
            )
            mlflow.log_table(relevance_table(report), "relevance_histogram.json")
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        return run_id


_AUDIT_COUNTS = (
    "links",
    "unverified",
    "healthy",
    "source_pages",
    "index_like_pages",
    "listing_pages",
    "sitemap_pages",
    "sitemap_links",
    "paginated_pages",
    "paginated_links",
    "ladder_pairs",
    "proposals",
    "keyword_cosines",
    "seconds",
)


def _audit_scores(report: LinkAuditReport) -> dict[str, ScoreDistribution]:
    return {
        name: found
        for name, found in (
            ("keyword_alignment", report.keyword_alignment),
            ("context_relevance", report.context_relevance),
            ("anchor_target_fit", report.anchor_target_fit),
            ("equity_efficiency", report.equity_efficiency),
            ("anchor_quality", report.anchor_quality),
        )
        if found is not None
    }


def link_audit_metrics(report: LinkAuditReport) -> dict[str, float]:
    """Counts by flag and verdict, the cut-offs the tenant's links gave and each score's
    statistics and split, flat; absent values left out."""
    metrics = {name: float(getattr(report, name)) for name in _AUDIT_COUNTS}
    metrics["embeddings"] = float(report.embeddings)
    metrics.update(
        {f"flag_{flag.value.lower()}": float(report.by_flag.get(flag, 0)) for flag in IssueFlag}
    )
    metrics.update(
        {
            f"verdict_{verdict.value.lower()}": float(report.by_verdict.get(verdict, 0))
            for verdict in sorted(AUDIT_VERDICTS)
        }
    )
    metrics.update(
        {
            f"cutoff_{cutoff.name}": cutoff.value
            for cutoff in report.cutoffs
            if cutoff.value is not None
        }
    )
    for name, found in _audit_scores(report).items():
        metrics.update(
            {
                f"{name}_{field}": float(value)
                for field, value in found.model_dump(exclude={"histogram"}).items()
                if value is not None
            }
        )
    return metrics


def link_audit_tables(report: LinkAuditReport) -> dict[str, dict[str, list[object]]]:
    """The reasons given and how many links each, the cut-offs with how they were derived, and
    every score's histogram, a row per score and bin over [0, 1]; no urls."""
    width = 1 / HISTOGRAM_BINS
    scores = _audit_scores(report)
    reasons = sorted(report.by_reason.items())
    return {
        "audit_reasons.json": {
            "reason": [reason.value for reason, _ in reasons],
            "links": [count for _, count in reasons],
        },
        "audit_cutoffs.json": {
            "cutoff": [cutoff.name for cutoff in report.cutoffs],
            "value": [cutoff.value for cutoff in report.cutoffs],
            "reason": [cutoff.reason for cutoff in report.cutoffs],
        },
        "audit_histogram.json": {
            "score": [name for name in scores for _ in range(HISTOGRAM_BINS)],
            "bin": [step for _ in scores for step in range(HISTOGRAM_BINS)],
            "low": [round(step * width, 6) for _ in scores for step in range(HISTOGRAM_BINS)],
            "high": [
                round((step + 1) * width, 6) for _ in scores for step in range(HISTOGRAM_BINS)
            ],
            "count": [count for found in scores.values() for count in found.histogram],
        },
    }


def log_link_audit(report: LinkAuditReport, summary: str) -> str:
    """Log one link audit run from its report in the tenant's analytics experiment: counts and
    cut-offs as metrics, each score's histogram as the step-indexed metric ``<score>_hist``
    (step = bin), reasons, cut-offs and histograms as tables; returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="link audit",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "link-audit",
            "audit_run_id": report.run_id,
            "mlflow.note.content": summary,
        },
    ) as run:
        run_id = str(run.info.run_id)
        mlflow.log_params(
            {
                "alignment_jaccard_share": ALIGNMENT_JACCARD_SHARE,
                "alignment_cosine_share": ALIGNMENT_COSINE_SHARE,
                "alignment_weight": ALIGNMENT_WEIGHT,
                "fit_weight": FIT_WEIGHT,
                "context_weight": CONTEXT_WEIGHT,
                "generic_cap": GENERIC_CAP,
                "over_optimised_min": OVER_OPTIMISED_MIN,
                "over_optimised_share": OVER_OPTIMISED_SHARE,
                "fence_iqrs": FENCE_IQRS,
                "min_iqr": MIN_IQR,
                "pagination_params": ",".join(sorted(PAGINATION_PARAMS)),
                "min_split_scores": MIN_SPLIT_SCORES,
                "min_mode_gap": MIN_MODE_GAP,
                "split_seed": SPLIT_SEED,
                "histogram_bins": HISTOGRAM_BINS,
                "embeddings_skipped_reason": report.embeddings_skipped_reason or "none",
                "vectors_skipped_reason": report.vectors_skipped_reason or "none",
            }
        )
        mlflow.log_metrics(link_audit_metrics(report))
        scores = _audit_scores(report)
        if scores:
            now = int(time.time() * 1000)
            mlflow.MlflowClient().log_batch(
                run_id,
                metrics=[
                    Metric(f"{name}_hist", float(count), now, step)
                    for name, found in scores.items()
                    for step, count in enumerate(found.histogram)
                ],
            )
        for artifact, table in link_audit_tables(report).items():
            mlflow.log_table(table, artifact)
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        return run_id


def bridge_metrics(report: BridgeReport) -> dict[str, float]:
    """Every count of the run, flat, with the covered pairs per reason as ``pairs_<reason>``."""
    metrics: dict[str, float] = {
        name: float(value)
        for name, value in report.model_dump(
            exclude={"tenant_id", "floor_share", "by_reason", "finished_at"}
        ).items()
    }
    metrics.update(
        {
            f"pairs_{reason.value.lower()}": float(count)
            for reason, count in report.by_reason.items()
        }
    )
    return metrics


def hub_pair_table(pairs: Sequence[HubPair]) -> dict[str, list[object]]:
    """Every scored hub pair with its three bridge-gap terms, how many queries the hubs share
    and its reasons; hub ids and counts, never urls or queries."""
    columns = (
        "language",
        "hub_a",
        "hub_b",
        "size_a",
        "size_b",
        "pages_ab",
        "pages_ba",
        "link_density",
        "centroid_cosine",
        "query_jaccard",
        "bridge_gap",
    )
    table: dict[str, list[object]] = {
        name: [getattr(pair, name) for pair in pairs] for name in columns
    }
    table["shared_query_count"] = [len(pair.shared_queries) for pair in pairs]
    table["reasons"] = [",".join(reason.value for reason in pair.reasons) for pair in pairs]
    return table


def log_bridges(report: BridgeReport, pairs: Sequence[HubPair], summary: str) -> str:
    """Log one hub-bridge run: its counts, the hub pair table and its description; no page urls
    reach the run. Returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="hub bridges",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "hub-bridges",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_params(
            {
                "floor_share": FLOOR_SHARE,
                "nearest_hubs": NEAREST_HUBS,
                "top_gap_pairs": TOP_GAP_PAIRS,
                "alternatives": ALTERNATIVES,
                "density_weight": DENSITY_WEIGHT,
                "cosine_weight": COSINE_WEIGHT,
                "jaccard_weight": JACCARD_WEIGHT,
                "shared_queries": SHARED_QUERIES,
                "relevance_decimals": RELEVANCE_DECIMALS,
            }
        )
        mlflow.log_metrics(bridge_metrics(report))
        mlflow.log_table(hub_pair_table(pairs), "hub_pairs.json")
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        return str(run.info.run_id)


def anchor_metrics(report: AnchorReport) -> dict[str, float]:
    """Every count of the run, flat: per rung (``rung_<rung>``), per best rung
    (``best_rung_<rung>``) and per keyword rank (``keyword_rank_<n>``), plus the match shares
    of the pairs with a ranked keyword set."""
    metrics: dict[str, float] = {
        name: float(value)
        for name, value in report.model_dump(
            exclude={
                "tenant_id",
                "stem_set_threshold",
                "by_rung",
                "best_rung",
                "by_keyword_rank",
                "by_rung_and_rank",
                "stem_jaccard_histogram",
                "sentence_index_histogram",
                "stemmed_languages",
                "unstemmed_languages",
                "finished_at",
            }
        ).items()
    }
    metrics.update({f"rung_{rung.value.lower()}": float(n) for rung, n in report.by_rung.items()})
    metrics.update(
        {f"best_rung_{rung.value.lower()}": float(n) for rung, n in report.best_rung.items()}
    )
    metrics.update({f"keyword_rank_{rank}": float(n) for rank, n in report.by_keyword_rank.items()})
    if report.pairs_with_keywords:
        metrics["matched_share"] = report.pairs_matched / report.pairs_with_keywords
        metrics["primary_matched_share"] = report.primary_matched / report.pairs_with_keywords
    return metrics


def anchor_language_table(report: AnchorReport) -> dict[str, list[object]]:
    """Source pages per language, and whether a stemmer covered it."""
    rows = sorted(
        [(language, True, n) for language, n in report.stemmed_languages.items()]
        + [(language, False, n) for language, n in report.unstemmed_languages.items()]
    )
    return {
        "language": [language for language, _, _ in rows],
        "stemmed": [stemmed for _, stemmed, _ in rows],
        "source_pages": [n for _, _, n in rows],
    }


def anchor_rank_table(report: AnchorReport) -> dict[str, list[object]]:
    """Matches per rung and keyword rank, one row each."""
    rows = [
        (rung.value, rank, count)
        for rung, ranks in report.by_rung_and_rank.items()
        for rank, count in ranks.items()
    ]
    return {
        "rung": [rung for rung, _, _ in rows],
        "keyword_rank": [rank for _, rank, _ in rows],
        "matches": [count for _, _, count in rows],
    }


def log_anchors(report: AnchorReport, summary: str) -> str:
    """Log one anchor extraction run from its report: counts, shares, the stem set Jaccard and
    sentence position histograms as step-indexed metrics (step = bin), and tables; never urls,
    phrases or sentences. Returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="anchor extraction",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "anchor-extraction",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_params(
            {
                "stem_set_threshold": report.stem_set_threshold,
                "min_span": MIN_SPAN,
                "max_span": MAX_SPAN,
                "min_shared_stems": MIN_SHARED_STEMS,
                "max_inner_stop_words": MAX_INNER_STOP_WORDS,
                "stemmer_languages": ",".join(sorted(SNOWBALL)),
                "sentence_index_bins": ",".join(map(str, SENTENCE_INDEX_BINS)),
            }
        )
        mlflow.log_metrics(anchor_metrics(report))
        now = int(time.time() * 1000)
        mlflow.MlflowClient().log_batch(
            run.info.run_id,
            metrics=[
                Metric(name, float(count), now, step)
                for name, histogram in (
                    ("stem_jaccard_hist", report.stem_jaccard_histogram),
                    ("sentence_index_hist", report.sentence_index_histogram),
                )
                for step, count in enumerate(histogram)
            ],
        )
        mlflow.log_table(anchor_rank_table(report), "rung_by_rank.json")
        mlflow.log_table(anchor_language_table(report), "languages.json")
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        return str(run.info.run_id)


def selection_metrics(report: AnchorSelectionReport) -> dict[str, float]:
    """Every count of the run, flat, with the threshold's values when the rung used one; per
    unanchored reason
    (``unanchored_<reason>``), chosen type (``type_<type>``) and chosen keyword rank
    (``keyword_rank_<n>``)."""
    threshold = report.threshold
    metrics: dict[str, float] = {
        name: float(value)
        for name, value in report.model_dump(
            exclude={
                "tenant_id",
                "threshold",
                "semantic_skipped_reason",
                "embedding_skipped_reason",
                "unanchored",
                "chosen_types",
                "chosen_ranks",
                "profile",
                "score_histogram",
                "semantic_histogram",
                "finished_at",
            }
        ).items()
    }
    metrics.update(
        {
            "semantic_skipped": float(report.semantic_skipped_reason is not None),
            "embedding_skipped": float(report.embedding_skipped_reason is not None),
        }
    )
    # A threshold is only reported as used when the rung ran, and a derivation only when the
    # threshold was derived rather than configured.
    if report.semantic_skipped_reason is None:
        metrics.update(
            {
                "threshold_value": threshold.value,
                "threshold_overridden": float(threshold.overridden),
                "threshold_positives": float(threshold.positives),
            }
        )
        if threshold.positive_recall is not None:
            metrics["threshold_positive_recall"] = threshold.positive_recall
        if not threshold.overridden:
            metrics.update(
                {
                    "threshold_negatives": float(threshold.negatives),
                    "threshold_bounded": float(threshold.bounded),
                    "threshold_fallback": float(threshold.fallback),
                }
            )
    if report.targets:
        metrics["targets_with_anchor_share"] = report.targets_with_anchor / report.targets
    metrics.update(
        {f"unanchored_{reason.value.lower()}": float(n) for reason, n in report.unanchored.items()}
    )
    metrics.update(
        {f"type_{kind.value.lower()}": float(n) for kind, n in report.chosen_types.items()}
    )
    metrics.update({f"keyword_rank_{rank}": float(n) for rank, n in report.chosen_ranks.items()})
    return metrics


def selection_tables(report: AnchorSelectionReport) -> dict[str, dict[str, list[object]]]:
    """The chosen types against the profile, the chosen keyword ranks and the unanchored pairs
    by reason with their advice, as MLflow tables keyed by artifact file."""
    wanted = report.profile.model_dump()
    types = list(report.chosen_types)
    return {
        "types.json": {
            "type": [kind.value for kind in types],
            "chosen": [report.chosen_types[kind] for kind in types],
            "share": [
                report.chosen_types[kind] / report.chosen if report.chosen else 0.0
                for kind in types
            ],
            "profile": [wanted[kind.value.lower()] for kind in types],
        },
        "keyword_ranks.json": {
            "keyword_rank": list(report.chosen_ranks),
            "chosen": list(report.chosen_ranks.values()),
        },
        "unanchored.json": {
            "reason": [reason.value for reason in report.unanchored],
            "pairs": list(report.unanchored.values()),
            "advice": [UNANCHORED_ADVICE[reason] for reason in report.unanchored],
        },
    }


def selection_params(report: AnchorSelectionReport) -> dict[str, object]:
    """The scoring constants; the semantic rung's when it ran, and the threshold derivation's
    only when the threshold was derived rather than configured."""
    params: dict[str, object] = {
        "semantic_weight": SEMANTIC_WEIGHT,
        "keyword_weight": KEYWORD_WEIGHT,
        "diversity_weight": DIVERSITY_WEIGHT,
        "length_weight": LENGTH_WEIGHT,
        "stem_share": STEM_SHARE,
        "cosine_share": COSINE_SHARE,
        "secondary_weight": SECONDARY_WEIGHT,
        "profile_bonus": PROFILE_BONUS,
        "awkward_floor": AWKWARD_FLOOR,
        "score_bins": ANCHOR_SCORE_BINS,
    }
    if report.semantic_skipped_reason is None:
        params.update(
            {
                "top_sentences": TOP_SENTENCES,
                "max_phrases_per_sentence": MAX_PHRASES_PER_SENTENCE,
                "threshold_overridden": report.threshold.overridden,
            }
        )
        if not report.threshold.overridden:
            params.update(
                {
                    "threshold_quantile": report.threshold.quantile,
                    "threshold_bounds": ",".join(map(str, THRESHOLD_BOUNDS)),
                    "min_negatives": MIN_NEGATIVES,
                    "negative_sample": NEGATIVE_SAMPLE,
                    "default_semantic_threshold": DEFAULT_SEMANTIC_THRESHOLD,
                }
            )
    return params


def log_anchor_selection(report: AnchorSelectionReport, summary: str) -> str:
    """Log one anchor selection run from its report: counts, the semantic threshold, the chosen
    anchors' totals and the semantic matches' similarities as step-indexed histograms (step =
    bin), and tables; never urls, phrases, sentences or keywords. Returns the MLflow run id."""
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="anchor selection",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "pipeline",
            "stage": "anchor-selection",
            "embedding_skipped": report.embedding_skipped_reason or "none",
            "mlflow.note.content": summary,
        },
    ) as run:
        mlflow.log_params(selection_params(report))
        mlflow.log_metrics(selection_metrics(report))
        now = int(time.time() * 1000)
        mlflow.MlflowClient().log_batch(
            run.info.run_id,
            metrics=[
                Metric(name, float(count), now, step)
                for name, histogram in (
                    ("anchor_score_hist", report.score_histogram),
                    ("semantic_similarity_hist", report.semantic_histogram),
                )
                for step, count in enumerate(histogram)
            ],
        )
        for artifact, table in selection_tables(report).items():
            mlflow.log_table(table, artifact)
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_text(summary, "summary.md")
        return str(run.info.run_id)


def _joined(names: Sequence[str]) -> str:
    return ",".join(names) or "none"


def quality_step_metrics(report: QualityReport) -> dict[str, tuple[int, ...]]:
    """The report's distributions, each logged as a metric whose step is the bin and whose
    value is the count."""
    steps: dict[str, tuple[int, ...]] = {}
    if report.scorer is not None:
        steps["score_hist_hidden"] = report.scorer.hidden_histogram
        steps["score_hist_other"] = report.scorer.other_histogram
    if report.link_relevance is not None:
        steps["context_relevance_hist"] = report.link_relevance.context.histogram
        if report.link_relevance.anchor is not None:
            steps["anchor_target_fit_hist"] = report.link_relevance.anchor.histogram
    return steps


def quality_tables(report: QualityReport) -> dict[str, dict[str, list[object]]]:
    """The report's per-item results as MLflow tables, keyed by artifact file; column names,
    ranks and counts only, never urls or keyword texts."""
    tables: dict[str, dict[str, list[object]]] = {}
    if (retrieval := report.retrieval) is not None:
        tables["recall.json"] = {
            "k": [entry.k for entry in retrieval.recall],
            "recall": [entry.recall for entry in retrieval.recall],
            "random": [entry.random for entry in retrieval.recall],
        }
    if (signal := report.feature_signal) is not None:
        tables["feature_auc.json"] = {
            "column": [entry.column for entry in signal.columns],
            "auc": [entry.auc for entry in signal.columns],
            "coverage": [entry.coverage for entry in signal.columns],
            "ranker_auc": [entry.ranker_auc for entry in signal.columns],
            "signal": [has_signal(entry, signal.margin) for entry in signal.columns],
            "link_derived": [entry.column in LINK_DERIVED_COLUMNS for entry in signal.columns],
        }
    if (scorer := report.scorer) is not None:
        width = 100 / SCORE_HISTOGRAM_BINS
        tables["score_histogram.json"] = {
            "bin": list(range(SCORE_HISTOGRAM_BINS)),
            "low": [round(i * width, 6) for i in range(SCORE_HISTOGRAM_BINS)],
            "high": [round((i + 1) * width, 6) for i in range(SCORE_HISTOGRAM_BINS)],
            "hidden": list(scorer.hidden_histogram),
            "other": list(scorer.other_histogram),
        }
    if (keywords := report.keywords) is not None:
        tables["keyword_rungs.json"] = {
            "rung": [rung.value for rung in keywords.by_rung],
            "pages": list(keywords.by_rung.values()),
        }
        if keywords.fallbacks_rejected:
            tables["keyword_rejections.json"] = {
                "reason": list(keywords.fallbacks_rejected),
                "fallbacks": list(keywords.fallbacks_rejected.values()),
            }
        if (relevance := keywords.relevance) is not None:
            tables["keyword_relevance.json"] = {
                "rank": [entry.rank for entry in relevance.ranks],
                "keywords": [entry.keywords for entry in relevance.ranks],
                "mean": [entry.mean for entry in relevance.ranks],
                "p10": [entry.p10 for entry in relevance.ranks],
                "p50": [entry.p50 for entry in relevance.ranks],
                "p90": [entry.p90 for entry in relevance.ranks],
            }
            for name, column, groups in (
                ("keyword_relevance_by_origin.json", "origin", relevance.by_origin),
                ("keyword_relevance_by_length.json", "words", relevance.by_length),
            ):
                tables[name] = {
                    column: [entry.group for entry in groups],
                    "keywords": [entry.keywords for entry in groups],
                    "mean": [entry.mean for entry in groups],
                    "median": [entry.p50 for entry in groups],
                }
    if (links := report.link_relevance) is not None:
        width = 1 / HISTOGRAM_BINS
        scores = {"context_relevance": links.context}
        if links.anchor is not None:
            scores["anchor_target_fit"] = links.anchor
        tables["relevance_histogram.json"] = {
            "score": [name for name in scores for _ in range(HISTOGRAM_BINS)],
            "bin": [step for _ in scores for step in range(HISTOGRAM_BINS)],
            "low": [round(step * width, 6) for _ in scores for step in range(HISTOGRAM_BINS)],
            "high": [
                round((step + 1) * width, 6) for _ in scores for step in range(HISTOGRAM_BINS)
            ],
            "count": [count for found in scores.values() for count in found.histogram],
        }
    if report.alerts:
        tables["alerts.json"] = {
            "metric": [alert.metric for alert in report.alerts],
            "previous": [alert.previous for alert in report.alerts],
            "current": [alert.current for alert in report.alerts],
            "change": [alert.change for alert in report.alerts],
            "band": [alert.band for alert in report.alerts],
            "relative": [alert.relative for alert in report.alerts],
        }
    return tables


def previous_quality_run(tenant_id: str) -> QualityBaseline | None:
    """The tenant's latest finished quality-eval run by end time, from its own experiment
    only; None before the first. Read-only: a missing experiment is not created."""
    experiment = mlflow.get_experiment_by_name(analytics_experiment(tenant_id))
    if experiment is None:
        return None
    runs = mlflow.MlflowClient().search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.stage = '{QUALITY_STAGE}' and attributes.status = 'FINISHED'",
        order_by=["attributes.end_time DESC"],
        max_results=_BASELINE_CANDIDATES,
    )
    for run in runs:
        if run.data.tags.get("tenant_id") == tenant_id:
            return QualityBaseline(
                run_id=run.info.run_id,
                metrics={
                    name: float(value)
                    for name, value in run.data.metrics.items()
                    if math.isfinite(value)
                },
            )
    return None


def log_quality(report: QualityReport, summary: str) -> str:
    """Log one quality evaluation: the flat metrics, the distributions as step-indexed
    metrics (step = bin), tables, the report, the metrics and the description; not
    applicable checks and alerts as comma-joined tags. Returns the MLflow run id."""
    versions = report.versions
    use_analytics_experiment(report.tenant_id)
    with mlflow.start_run(
        run_name="quality eval",
        tags={
            "tenant_id": report.tenant_id,
            "kind": "eval",
            "stage": QUALITY_STAGE,
            "git_sha": versions.git_sha,
            "feature_digest": versions.feature_digest,
            "weights_version": versions.weights_version,
            "weights_hash": versions.weights_hash,
            "not_applicable": _joined(report.not_applicable),
            "alerts": _joined([alert.metric for alert in report.alerts]),
            "baseline_run": report.baseline_run_id or "none",
            "mlflow.note.content": summary,
        },
    ) as run:
        run_id = str(run.info.run_id)
        mlflow.log_params(
            {
                "hide_share": HIDE_SHARE,
                "hide_seed": HIDE_SEED,
                "recall_ks": ",".join(map(str, RECALL_KS)),
                "per_target": PER_TARGET,
                "signal_margin": SIGNAL_MARGIN,
                "anchor_jaccard": ANCHOR_JACCARD,
                "alert_band": ALERT_BAND,
                "link_derived_columns": ",".join(LINK_DERIVED_COLUMNS),
            }
        )
        metrics = quality_metrics(report)
        mlflow.log_metrics(metrics)
        steps = quality_step_metrics(report)
        if steps:
            now = int(time.time() * 1000)
            mlflow.MlflowClient().log_batch(
                run_id,
                metrics=[
                    Metric(name, float(count), now, step)
                    for name, counts in steps.items()
                    for step, count in enumerate(counts)
                ],
            )
        mlflow.log_dict(report.model_dump(mode="json"), "report.json")
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_text(summary, "summary.md")
        for artifact, table in quality_tables(report).items():
            mlflow.log_table(table, artifact)
        return run_id
