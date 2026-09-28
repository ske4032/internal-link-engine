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
    duplicate_metrics,
    feature_metrics,
    keyword_metrics,
    log_analytics,
    log_candidates,
    log_duplicates,
    log_features,
    log_keywords,
)
from linking_engine.models import (
    CandidateReport,
    CandidateSet,
    CentralityReport,
    CommunityReport,
    DuplicateGroup,
    DuplicateReport,
    FeatureReport,
    HubReport,
    KeywordReport,
    KeywordRung,
    KeywordSource,
    OrphanLabel,
    PassReport,
    TargetCandidates,
)

if TYPE_CHECKING:
    from pathlib import Path

# A deliberately low-entropy stand-in for a sha256 cache key.
CACHE_KEY = "f" * 64


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
        "non_canonical_excluded": 0,
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


# ── keyword resolution ──────────────────────────────────────────────────────


def keyword_report(tenant: str = "acme") -> KeywordReport:
    return KeywordReport(
        tenant_id=tenant,
        pages=6,
        resolved=5,
        by_rung={
            KeywordRung.STRATEGIC: 1,
            KeywordRung.GSC: 1,
            KeywordRung.H1: 2,
            KeywordRung.TITLE: 1,
        },
        gsc_enabled=True,
        gsc_rows=123,
        gsc_rejected=1,
        brand_suffix="Acme",
        brand_prefix=None,
        fallbacks_rejected={"h1_repeated": 2, "title_generic": 1},
        long_fallbacks=1,
        secondary_keywords=4,
        pages_with_secondaries=2,
        edges_written={
            KeywordSource.CLIENT_STRATEGIC: 2,
            KeywordSource.GSC_OBSERVED: 1,
            KeywordSource.INFERRED: 3,
        },
        stale_edges_deleted={
            KeywordSource.CLIENT_STRATEGIC: 1,
            KeywordSource.GSC_OBSERVED: 0,
            KeywordSource.INFERRED: 0,
        },
        skipped_rows=2,
        by_language={"en": 5, "de": 1},
        seconds=0.5,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_keyword_metrics_are_flat_counts_per_rung_source_and_language() -> None:
    assert keyword_metrics(keyword_report()) == {
        "pages": 6,
        "resolved": 5,
        "gsc_enabled": 1,
        "gsc_rows": 123,
        "gsc_rejected": 1,
        "skipped_rows": 2,
        "seconds": 0.5,
        "rung_strategic": 1,
        "rung_gsc": 1,
        "rung_h1": 2,
        "rung_title": 1,
        "edges_client_strategic": 2,
        "edges_gsc_observed": 1,
        "edges_inferred": 3,
        "stale_client_strategic": 1,
        "stale_gsc_observed": 0,
        "stale_inferred": 0,
        "pages_en": 5,
        "pages_de": 1,
        "rejected_h1_repeated": 2,
        "rejected_title_generic": 1,
        "long_fallbacks": 1,
        "secondary_keywords": 4,
        "pages_with_secondaries": 2,
    }


def test_a_keyword_run_is_logged_to_the_tenants_experiment(local_mlflow: str) -> None:
    report = keyword_report()

    run_id = log_keywords(report, "Keyword resolution for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert run.info.run_name == "keyword resolution"
    assert (run.data.tags["stage"], run.data.tags["tenant_id"]) == ("resolve-keywords", "acme")
    assert run.data.tags["mlflow.note.content"] == "Keyword resolution for tenant acme."
    assert run.data.params == {
        "min_curve_rows": "100",
        "min_curve_impressions": "10000",
        "min_query_impressions": "50",
        "brand_suffix_share": "0.3",
        "repeated_fallback_pages": "3",
        "max_keyword_tokens": "12",
        "max_secondary_queries": "4",
    }
    assert run.data.metrics == keyword_metrics(report)
    assert {a.path for a in client.list_artifacts(run_id)} == {"report.json", "summary.md"}
    assert load_dict(f"runs:/{run_id}/report.json") == report.model_dump(mode="json")


# ── feature assembly ────────────────────────────────────────────────────────


def feature_report(pairs: int = 12) -> FeatureReport:
    return FeatureReport(
        tenant_id="acme",
        pairs=pairs,
        columns=("content_cosine", "has_gsc_data", "context_relevance"),
        chunks=3 if pairs else 0,
        all_null_columns=("context_relevance",) if pairs else (),
        constant_columns=("has_gsc_data",) if pairs else (),
        null_share=(
            {"content_cosine": 0.0, "has_gsc_data": 0.0, "context_relevance": 1.0} if pairs else {}
        ),
        has_gsc_data_share=0.25 if pairs else None,
        cache_key=CACHE_KEY,
        cache_hit=False,
        seconds=1.5,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_feature_metrics_carry_counts_and_every_null_share() -> None:
    assert feature_metrics(feature_report()) == {
        "pairs": 12,
        "chunks": 3,
        "columns": 3,
        "all_null_columns": 1,
        "constant_columns": 1,
        "seconds": 1.5,
        "has_gsc_data_share": 0.25,
        "null_share_content_cosine": 0.0,
        "null_share_has_gsc_data": 0.0,
        "null_share_context_relevance": 1.0,
    }
    assert "has_gsc_data_share" not in feature_metrics(feature_report(pairs=0))


def test_a_feature_run_logs_its_column_order_but_never_the_matrix(local_mlflow: str) -> None:
    report = feature_report()

    run_id = log_features(report, "Feature assembly for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert (run.info.run_name, run.data.tags["stage"]) == ("feature assembly", "feature-assembly")
    assert run.data.params == {
        "cache_key": CACHE_KEY,
        "cache_hit": "False",
        "position_bands": "3,10,20,50",
    }
    assert run.data.metrics == feature_metrics(report)
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "report.json",
        "summary.md",
        "columns.json",
    }
    assert load_dict(f"runs:/{run_id}/columns.json") == {"feature_columns": list(report.columns)}


# ── duplicate pages ─────────────────────────────────────────────────────────


def duplicate_report(tenant: str = "acme") -> DuplicateReport:
    return DuplicateReport(
        tenant_id=tenant,
        groups=(
            DuplicateGroup(
                group_id=0,
                canonical="example.com/blog/post",
                copies=("example.com/news/post", "example.com/post"),
            ),
            DuplicateGroup(group_id=1, canonical="example.com/guide", copies=("example.com/g",)),
        ),
        pages_in_groups=5,
        non_canonical=3,
        largest_group=3,
        seconds=0.25,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_duplicate_metrics_are_the_runs_counts() -> None:
    assert duplicate_metrics(duplicate_report()) == {
        "groups": 2,
        "pages_in_groups": 5,
        "non_canonical": 3,
        "largest_group": 3,
        "seconds": 0.25,
    }


def test_a_duplicate_run_logs_every_group_to_the_tenants_experiment(local_mlflow: str) -> None:
    report = duplicate_report()

    run_id = log_duplicates(report, "Exact duplicate pages of tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert run.info.run_name == "duplicate pages"
    assert {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")} == {
        "tenant_id": "acme",
        "kind": "pipeline",
        "stage": "duplicates",
    }
    assert run.data.tags["mlflow.note.content"] == "Exact duplicate pages of tenant acme."
    assert run.data.metrics == duplicate_metrics(report)
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "report.json",
        "summary.md",
        "groups.json",
    }
    artifacts = f"runs:/{run_id}"
    assert load_dict(f"{artifacts}/report.json") == {
        "tenant_id": "acme",
        "groups": 2,
        "pages_in_groups": 5,
        "non_canonical": 3,
        "largest_group": 3,
        "seconds": 0.25,
        "finished_at": "2026-09-28T00:00:00Z",
    }
    assert load_dict(f"{artifacts}/groups.json") == {
        "groups": [
            {
                "group_id": 0,
                "canonical": "example.com/blog/post",
                "copies": ["example.com/news/post", "example.com/post"],
            },
            {"group_id": 1, "canonical": "example.com/guide", "copies": ["example.com/g"]},
        ]
    }
    assert load_text(f"{artifacts}/summary.md") == "Exact duplicate pages of tenant acme."
