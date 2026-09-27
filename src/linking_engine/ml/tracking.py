"""MLflow tracking for pipeline runs: one experiment per tenant, one run per execution."""

from __future__ import annotations

import csv
import io
from importlib.metadata import version
from typing import TYPE_CHECKING

import mlflow

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

if TYPE_CHECKING:
    from linking_engine.models import (
        CandidateReport,
        CandidateSet,
        CentralityReport,
        CommunityReport,
        HubReport,
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
