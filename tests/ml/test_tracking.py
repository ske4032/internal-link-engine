from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text

from linking_engine.ml.tracking import (
    analytics_experiment,
    analytics_metrics,
    candidate_metrics,
    candidate_table,
    log_analytics,
    log_candidates,
)
from linking_engine.models import (
    CandidateReport,
    CandidateSet,
    CentralityReport,
    CommunityReport,
    HubReport,
    OrphanLabel,
    PassReport,
    TargetCandidates,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs go to a throwaway local store, never the remote server. Only the environment is
    set: an explicit set_tracking_uri would outlive the test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def found(pages: int, drift: float | None = None) -> PassReport:
    return PassReport(
        pages=pages,
        edges=pages * 2,
        communities=3 if pages else 0,
        singletons=0,
        largest_community_pct=0.4 if pages else 0.0,
        median_community_size=10.0 if pages else 0.0,
        modularity=0.5 if pages else 0.0,
        disconnected_communities=0,
        seed_stability_ari_mean=0.9 if pages else None,
        seed_stability_ari_min=0.8 if pages else None,
        drift_ari=drift,
        pillars=3 if pages else 0,
        runtime_s=0.1,
        stability_s=0.2,
    )


def hub_report(tenant: str = "acme") -> HubReport:
    return HubReport(
        tenant_id=tenant,
        pages=30,
        hubs=3,
        noise=6,
        noise_pct=0.2,
        largest_hub_pct=0.4,
        median_hub_size=8.0,
        relative_validity=0.15,
        persistence_mean=0.2,
        persistence_min=0.1,
        persistence_weighted=0.25,
        matched_hubs=2,
        new_hubs=1,
        retired_hubs=0,
        section_pages={"/blog": 20, "/legal": 10},
        section_noise={"/legal": 6},
        runtime_s=0.5,
        write_s=0.1,
    )


def reports(tenant: str = "acme") -> tuple[CentralityReport, CommunityReport, HubReport]:
    centrality = CentralityReport(
        tenant_id=tenant, pages=30, placeholders=2, pagerank_s=0.01, betweenness_s=0.02, write_s=0.1
    )
    communities = CommunityReport(
        tenant_id=tenant,
        crawled_pages=30,
        seen_not_crawled=2,
        link=found(28, drift=0.95),
        keyword=found(0),
        content=found(30),
        keywords=0,
        keywords_dropped=0,
        pages_with_keywords=0,
        pages_with_embeddings=30,
        agreement_link_content=0.3,
        orphans=4,
        dead_ends=5,
        orphan_labels={OrphanLabel.MENUS_ONLY: 3, OrphanLabel.NOT_LINKED: 1},
        write_s=0.2,
    )
    return centrality, communities, hub_report(tenant)


def test_metrics_are_flat_and_leave_out_what_was_not_computed() -> None:
    metrics = analytics_metrics(*reports())
    assert metrics["link_modularity"] == 0.5
    assert metrics["link_drift_ari"] == 0.95
    assert metrics["centrality_pages"] == 30
    assert metrics["orphans_menus_only"] == 3
    assert metrics["agreement_link_content"] == 0.3
    assert "keyword_seed_stability_ari_mean" not in metrics
    assert "agreement_link_keyword" not in metrics
    assert (metrics["hub_hubs"], metrics["hub_noise_pct"], metrics["hub_new_hubs"]) == (3, 0.2, 1)
    assert "hub_drift_ari" not in metrics
    assert not any("section" in name for name in metrics)


def test_a_run_is_logged_to_the_tenants_experiment_with_its_description(local_mlflow: str) -> None:
    run_id = log_analytics(*reports(), "Graph analytics for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == analytics_experiment("acme") == "analytics-acme"
    assert run.data.tags["mlflow.note.content"] == "Graph analytics for tenant acme."
    assert run.data.tags["tenant_id"] == "acme"
    assert run.data.params["resolution"] == "1.0"
    assert run.data.metrics["content_pages"] == 30
    assert run.data.params["hub_min_samples"] == "1"
    assert {a.path for a in client.list_artifacts(run_id)} == {"report.json", "summary.md"}


def test_reports_from_two_tenants_are_refused(local_mlflow: str) -> None:
    centrality, communities, _ = reports("acme")
    with pytest.raises(ValueError, match="same tenant"):
        log_analytics(centrality, communities, hub_report("globex"), "x")


# ── candidate retrieval ─────────────────────────────────────────────────────

COUNTS: dict[str, int | float | None] = {
    "targets": 2,
    "indexable_assumed": 1,
    "source_pages": 3,
    "candidates": 3,
    "full_targets": 1,
    "short_targets": 1,
    "empty_targets": 0,
    "min_per_target": 1,
    "median_per_target": 1.5,
    "max_per_target": 2,
    "linked_pairs": 4,
    "linked_nearer": 1,
    "drop_rate": 0.25,
}
NO_TARGETS: dict[str, int | float | None] = {
    "targets": 0,
    "indexable_assumed": 0,
    "source_pages": 0,
    "candidates": 0,
    "full_targets": 0,
    "short_targets": 0,
    "empty_targets": 0,
    "linked_pairs": 0,
    "linked_nearer": 0,
}


def candidate_set(tenant: str = "acme", *, empty: bool = False) -> CandidateSet:
    targets = (
        ()
        if empty
        else (
            TargetCandidates(
                target_url="example.com/a",
                sources=("example.com/b", "example.com/c"),
                similarities=(0.875, 0.5),
                eligible=5,
                linked=3,
                linked_nearer=1,
            ),
            TargetCandidates(
                target_url="example.com/new",
                sources=("example.com/a",),
                similarities=(0.625,),
                eligible=1,
                linked=1,
                linked_nearer=0,
            ),
        )
    )
    report = CandidateReport.model_validate(
        {
            "tenant_id": tenant,
            "index": "page_content",
            "per_target": 2,
            "chunk_size": 512,
            "crawled_pages": 4,
            "not_indexable": 1,
            "without_vector": 1,
            "load_seconds": 0.5,
            "search_seconds": 0.125,
            "seconds": 0.75,
            "finished_at": datetime(2026, 9, 27, tzinfo=UTC),
            **(NO_TARGETS if empty else COUNTS),
        }
    )
    return CandidateSet(report=report, targets=targets)


def test_candidate_metrics_are_the_runs_results_without_settings_or_missing_values() -> None:
    assert candidate_metrics(candidate_set().report) == {
        "crawled_pages": 4,
        "not_indexable": 1,
        "without_vector": 1,
        **COUNTS,
        "load_seconds": 0.5,
        "search_seconds": 0.125,
        "seconds": 0.75,
    }
    empty = candidate_metrics(candidate_set(empty=True).report)
    assert (empty["targets"], empty["candidates"], empty["source_pages"]) == (0, 0, 0)
    for missing in ("min_per_target", "median_per_target", "max_per_target", "drop_rate"):
        assert missing not in empty, f"{missing} was not computed and must be left out"


def test_the_target_table_leaves_similarities_blank_without_candidates() -> None:
    found = candidate_set()
    lonely = TargetCandidates(
        target_url="example.com/lonely",
        sources=(),
        similarities=(),
        eligible=0,
        linked=2,
        linked_nearer=0,
    )
    with_lonely = CandidateSet(
        report=found.report.model_copy(
            update={
                "targets": 3,
                "empty_targets": 1,
                "source_pages": 4,
                "linked_pairs": found.report.linked_pairs + lonely.linked,
            }
        ),
        targets=(*found.targets, lonely),
    )

    rows = list(csv.reader(io.StringIO(candidate_table(with_lonely))))

    assert rows == [
        [
            "target_url",
            "candidates",
            "eligible",
            "linked",
            "linked_nearer",
            "best_similarity",
            "last_similarity",
        ],
        ["example.com/a", "2", "5", "3", "1", "0.875", "0.5"],
        ["example.com/new", "1", "1", "1", "0", "0.625", "0.625"],
        ["example.com/lonely", "0", "0", "2", "0", "", ""],
    ]


def test_a_candidate_run_is_logged_with_its_settings_report_and_target_table(
    local_mlflow: str,
) -> None:
    found = candidate_set()

    run_id = log_candidates(found, "Candidate retrieval for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert run.info.run_name == "candidate retrieval"
    assert {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")} == {
        "tenant_id": "acme",
        "kind": "pipeline",
        "stage": "candidate-retrieval",
    }
    assert run.data.tags["mlflow.note.content"] == "Candidate retrieval for tenant acme."
    assert run.data.params == {"index": "page_content", "per_target": "2", "chunk_size": "512"}
    assert run.data.metrics == candidate_metrics(found.report)
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "report.json",
        "summary.md",
        "targets.csv",
    }
    artifacts = f"runs:/{run_id}"
    assert load_dict(f"{artifacts}/report.json") == found.report.model_dump(mode="json")
    assert load_text(f"{artifacts}/summary.md") == "Candidate retrieval for tenant acme."
    assert load_text(f"{artifacts}/targets.csv") == candidate_table(found)


def test_an_empty_candidate_run_is_logged_with_a_header_only_table(local_mlflow: str) -> None:
    run_id = log_candidates(candidate_set(empty=True), "No targets.")

    rows = list(csv.reader(io.StringIO(load_text(f"runs:/{run_id}/targets.csv"))))
    assert len(rows) == 1
    assert MlflowClient(local_mlflow).get_run(run_id).data.metrics["targets"] == 0


def test_candidate_runs_of_two_tenants_never_share_an_experiment(local_mlflow: str) -> None:
    client = MlflowClient(local_mlflow)
    acme = client.get_run(log_candidates(candidate_set("acme"), "a"))
    globex = client.get_run(log_candidates(candidate_set("globex"), "g"))
    analytics = client.get_run(log_analytics(*reports("acme"), "x"))

    assert acme.info.experiment_id == analytics.info.experiment_id
    assert globex.info.experiment_id != acme.info.experiment_id
    assert globex.data.tags["tenant_id"] == "globex"
