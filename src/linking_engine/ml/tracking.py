"""MLflow tracking for pipeline runs: one experiment per tenant, one run per execution."""

from __future__ import annotations

import csv
import io
from importlib.metadata import version
from typing import TYPE_CHECKING

import mlflow

from linking_engine.anchor.keywords import (
    BRAND_SUFFIX_SHARE,
    MAX_KEYWORD_TOKENS,
    MAX_SECONDARY_QUERIES,
    MIN_QUERY_IMPRESSIONS,
    REPEATED_FALLBACK_PAGES,
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

if TYPE_CHECKING:
    from linking_engine.models import (
        CandidateReport,
        CandidateSet,
        CentralityReport,
        CommunityReport,
        FeatureReport,
        HubReport,
        KeywordReport,
    )


def analytics_experiment(tenant_id: str) -> str:
    return f"analytics-{tenant_id}"


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
    mlflow.set_experiment(analytics_experiment(tenant_id))
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
    mlflow.set_experiment(analytics_experiment(report.tenant_id))
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
    mlflow.set_experiment(analytics_experiment(report.tenant_id))
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
    mlflow.set_experiment(analytics_experiment(report.tenant_id))
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
