"""MLflow tracking for pipeline runs: one experiment per tenant, one run per execution."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

import mlflow

from linking_engine.graph.algorithms import (
    DAMPING,
    KNN_NEIGHBOURS,
    MIN_PILLAR_COMMUNITY,
    PROJECTION_EDGE_BUDGET,
    RESOLUTION,
    SEED,
    STABILITY_SEEDS,
)

if TYPE_CHECKING:
    from linking_engine.models import CentralityReport, CommunityReport


def analytics_experiment(tenant_id: str) -> str:
    return f"analytics-{tenant_id}"


def analytics_metrics(
    centrality: CentralityReport, communities: CommunityReport
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
    return metrics


def log_analytics(centrality: CentralityReport, communities: CommunityReport, summary: str) -> str:
    """Log one graph analytics run with its description; returns the MLflow run id."""
    tenant_id = communities.tenant_id
    if centrality.tenant_id != tenant_id:
        raise ValueError("both reports must come from the same tenant")
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
                "igraph": version("igraph"),
                "leidenalg": version("leidenalg"),
            }
        )
        mlflow.log_metrics(analytics_metrics(centrality, communities))
        mlflow.log_dict(
            {
                "centrality": centrality.model_dump(mode="json"),
                "communities": communities.model_dump(mode="json"),
            },
            "report.json",
        )
        mlflow.log_text(summary, "summary.md")
        return str(run.info.run_id)
