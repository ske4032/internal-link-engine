"""MLflow tracking for pipeline runs: one experiment per tenant, one run per execution."""

from __future__ import annotations

import csv
import io
import time
from importlib.metadata import version
from typing import TYPE_CHECKING

import mlflow
from mlflow.entities import Metric

from linking_engine.anchor.keywords import (
    BRAND_SUFFIX_SHARE,
    MAX_KEYWORD_TOKENS,
    MAX_SECONDARY_QUERIES,
    MIN_QUERY_IMPRESSIONS,
    REPEATED_FALLBACK_PAGES,
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
from linking_engine.models.relevance import HISTOGRAM_BINS
from linking_engine.models.scoring import SCORE_HISTOGRAM_BINS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.models import (
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
        LinkRelevanceReport,
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
