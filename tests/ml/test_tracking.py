from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text

from linking_engine.ml.tracking import (
    EXPERIMENT_KIND,
    EXPERIMENT_KIND_TAG,
    analytics_experiment,
    analytics_metrics,
    anchor_language_table,
    anchor_metrics,
    anchor_rank_table,
    bridge_metrics,
    candidate_metrics,
    candidate_params,
    candidate_table,
    duplicate_metrics,
    feature_metrics,
    hub_pair_table,
    keyword_metrics,
    link_relevance_metrics,
    log_analytics,
    log_anchors,
    log_bridges,
    log_candidates,
    log_duplicates,
    log_features,
    log_keywords,
    log_link_relevance,
    log_scores,
    score_metrics,
)
from linking_engine.models import (
    AnchorReport,
    AnchorRung,
    BridgeReason,
    BridgeReport,
    CandidateReport,
    CandidateSet,
    CentralityReport,
    CommunityReport,
    DuplicateGroup,
    DuplicateReport,
    FeatureReport,
    FeatureWeight,
    HubPair,
    HubReport,
    KeywordReport,
    KeywordRung,
    KeywordSource,
    LinkRelevanceReport,
    OrphanLabel,
    PassReport,
    ScoreDistribution,
    ScoreReport,
    ScorerWeights,
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
    assert experiment.tags[EXPERIMENT_KIND_TAG] == EXPERIMENT_KIND == "custom_model_development"
    assert run.data.tags["mlflow.note.content"] == "Graph analytics for tenant acme."
    assert run.data.tags["tenant_id"] == "acme"
    assert run.data.params["resolution"] == "1.0"
    assert run.data.metrics["content_pages"] == 30
    assert run.data.params["hub_min_samples"] == "1"
    assert {a.path for a in client.list_artifacts(run_id)} == {"report.json", "summary.md"}


def test_an_existing_experiment_without_a_kind_is_tagged_for_the_model_training_view(
    local_mlflow: str,
) -> None:
    client = MlflowClient(local_mlflow)
    experiment_id = client.create_experiment(analytics_experiment("acme"))
    assert EXPERIMENT_KIND_TAG not in client.get_experiment(experiment_id).tags

    run_id = log_analytics(*reports(), "Graph analytics for tenant acme.")

    assert client.get_run(run_id).info.experiment_id == experiment_id
    assert client.get_experiment(experiment_id).tags[EXPERIMENT_KIND_TAG] == EXPERIMENT_KIND


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
        "pillar_pairs": 0,
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
            "pillar_pairs",
        ],
        ["example.com/a", "2", "5", "3", "1", "0.875", "0.5", "0"],
        ["example.com/new", "1", "1", "1", "0", "0.625", "0.625", "0"],
        ["example.com/lonely", "0", "0", "2", "0", "", "", "0"],
    ]


def with_pillar_pair() -> CandidateSet:
    """The first target as a hub pillar with one channel source, under per-language and
    tenant-wide floors."""
    found = candidate_set()
    pillar = TargetCandidates.model_validate(
        {
            **found.targets[0].model_dump(),
            "sources": ("example.com/b", "example.com/c", "example.com/d"),
            "similarities": (0.875, 0.5, 0.625),
            "pillar_pairs": 1,
        }
    )
    report = CandidateReport.model_validate(
        {
            **found.report.model_dump(),
            "candidates": 4,
            "pillar_pairs": 1,
            "pillar_floors": {"en": 0.5, "*": 0.25},
            "pillar_floor_basis": {"en": "existing_links", "*": "candidate_pairs"},
            "pillar_floor_links": {"en": 64, "*": 12},
        }
    )
    return CandidateSet(report=report, targets=(pillar, found.targets[1]))


def test_the_channels_floors_are_metrics_and_their_bases_params() -> None:
    found = with_pillar_pair()

    metrics = candidate_metrics(found.report)
    rows = list(csv.reader(io.StringIO(candidate_table(found))))

    assert {name: value for name, value in metrics.items() if "pillar" in name} == {
        "pillar_pairs": 1,
        "pillar_floor_en": 0.5,
        "pillar_floor_links_en": 64,
        "pillar_floor_tenant": 0.25,
        "pillar_floor_links_tenant": 12,
    }
    assert metrics["candidates"] == 4
    assert candidate_params(found.report) == {
        "index": "page_content",
        "per_target": "2",
        "chunk_size": "512",
        "pillar_floor_basis_en": "existing_links",
        "pillar_floor_basis_tenant": "candidate_pairs",
    }
    # The nearest sources' count and last similarity; the channel's pair counted apart.
    assert rows[1] == ["example.com/a", "2", "5", "3", "1", "0.875", "0.5", "1"]


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
    channel = client.get_run(log_candidates(with_pillar_pair(), "With the channel."))
    assert channel.data.params == candidate_params(with_pillar_pair().report)
    assert channel.data.metrics == candidate_metrics(with_pillar_pair().report)
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


# ── baseline scoring ────────────────────────────────────────────────────────

SCORE_WEIGHTS = ScorerWeights(
    version="baseline-1",
    features=(
        FeatureWeight(column="content_cosine", weight=0.6, normalisation="percentile"),
        FeatureWeight(column="target_ctr_gap", weight=0.4, direction="lower"),
    ),
)
HISTOGRAM = (2, 0, 0, 1, *([0] * 12), 3, 0, 1, 3)


def scored_report() -> ScoreReport:
    return ScoreReport(
        tenant_id="acme",
        pairs=10,
        weights=SCORE_WEIGHTS,
        weights_hash="f" * 64,
        tiers={1: 1, 2: 3, 3: 6},
        score_p10=0.0,
        score_p50=80.0,
        score_p90=100.0,
        score_histogram=HISTOGRAM,
        top_contributors={"content_cosine": 7, "target_ctr_gap": 3},
        missing_share={"content_cosine": 0.0, "target_ctr_gap": 0.4},
        feature_cache_key="e" * 64,
        seconds=0.5,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_score_metrics_are_counts_percentiles_shares_and_leaders() -> None:
    assert score_metrics(scored_report()) == {
        "pairs": 10,
        "seconds": 0.5,
        "tier_1": 1,
        "tier_2": 3,
        "tier_3": 6,
        "score_p10": 0.0,
        "score_p50": 80.0,
        "score_p90": 100.0,
        "missing_share_content_cosine": 0.0,
        "missing_share_target_ctr_gap": 0.4,
        "top_contributor_content_cosine": 7,
        "top_contributor_target_ctr_gap": 3,
    }


def table(run_id: str, name: str) -> dict[str, list[object]]:
    stored = load_dict(f"runs:/{run_id}/{name}")
    columns = stored["columns"]
    return {column: [row[i] for row in stored["data"]] for i, column in enumerate(columns)}


def test_a_scoring_run_logs_metrics_a_stepped_histogram_and_tables(local_mlflow: str) -> None:
    report = scored_report()

    run_id = log_scores(report, "Baseline scoring for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert (run.info.run_name, run.data.tags["stage"]) == ("baseline scoring", "score-pairs")
    assert run.data.tags["mlflow.note.content"] == "Baseline scoring for tenant acme."
    assert run.data.params == {
        "weights_version": "baseline-1",
        "weights_hash": "f" * 64,
        "feature_cache_key": "e" * 64,
        "tier_shares": "0.1,0.3",
    }
    assert score_metrics(report).items() <= run.data.metrics.items()
    history = sorted(client.get_metric_history(run_id, "score_hist"), key=lambda m: m.step)
    assert [(m.step, m.value) for m in history] == list(enumerate(map(float, HISTOGRAM)))
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "report.json",
        "summary.md",
        "score_histogram.json",
        "tiers.json",
        "weights.json",
    }
    histogram = table(run_id, "score_histogram.json")
    assert (histogram["bin"], histogram["count"]) == (list(range(20)), list(HISTOGRAM))
    assert (histogram["low"][0], histogram["high"][-1]) == (0.0, 100.0)
    tiers = table(run_id, "tiers.json")
    assert (tiers["tier"], tiers["pairs"]) == ([1, 2, 3], [1, 3, 6])
    assert tiers["share"] == pytest.approx([0.1, 0.3, 0.6])
    assert tiers["target_share"] == pytest.approx([0.1, 0.3, 0.6])
    weights = table(run_id, "weights.json")
    assert weights["column"] == ["content_cosine", "target_ctr_gap"]
    assert (weights["weight"], weights["direction"]) == ([0.6, 0.4], ["higher", "lower"])
    assert (weights["missing_share"], weights["top_contributor"]) == ([0.0, 0.4], [7, 3])


# ── link relevance ──────────────────────────────────────────────────────────

CONTEXT_HISTOGRAM = (0, 0, 1, 2, 4, 6, 8, 9, 7, 5, 3, 2, 4, 7, 9, 8, 6, 3, 1, 0)
ANCHOR_HISTOGRAM = (0,) * 10 + (1, 2, 3, 4, 5, 4, 3, 2, 1, 0)


def distribution(histogram: tuple[int, ...], split: float | None) -> ScoreDistribution:
    return ScoreDistribution(
        count=sum(histogram),
        mean=0.55,
        p10=0.2,
        p25=0.35,
        p50=0.5,
        p75=0.7,
        p90=0.8,
        histogram=histogram,
        split=split,
        low_share=None if split is None else 0.45,
    )


def relevance_report(*, scored: bool = True, anchor: bool = True) -> LinkRelevanceReport:
    return LinkRelevanceReport(
        tenant_id="acme",
        links=120,
        scored=85 if scored else 0,
        generic_anchors=12 if scored else 0,
        without_anchor_vector=48 if scored else 0,
        context=distribution(CONTEXT_HISTOGRAM, 0.52) if scored else None,
        anchor=distribution(ANCHOR_HISTOGRAM, None) if scored and anchor else None,
        seconds=3.5,
        finished_at=datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
    )


def test_a_run_logs_metrics_stepped_histograms_a_table_and_the_report(local_mlflow: str) -> None:
    report = relevance_report()

    run_id = log_link_relevance(report, "Link relevance for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert (run.info.run_name, run.data.tags["stage"]) == ("link relevance", "score-links")
    assert (run.data.tags["tenant_id"], run.data.tags["kind"]) == ("acme", "pipeline")
    assert run.data.tags["mlflow.note.content"] == "Link relevance for tenant acme."
    assert run.data.params == {
        "min_split_scores": "50",
        "min_mode_gap": "0.05",
        "split_seed": "0",
        "histogram_bins": "20",
    }
    assert link_relevance_metrics(report).items() <= run.data.metrics.items()
    for name, expected in (
        ("context_relevance_hist", CONTEXT_HISTOGRAM),
        ("anchor_target_fit_hist", ANCHOR_HISTOGRAM),
    ):
        history = sorted(client.get_metric_history(run_id, name), key=lambda m: m.step)
        assert [(m.step, m.value) for m in history] == list(enumerate(map(float, expected)))
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "report.json",
        "summary.md",
        "relevance_histogram.json",
    }
    rows = table(run_id, "relevance_histogram.json")
    assert rows["score"] == ["context_relevance"] * 20 + ["anchor_target_fit"] * 20
    assert rows["bin"] == list(range(20)) * 2
    assert rows["count"] == [*CONTEXT_HISTOGRAM, *ANCHOR_HISTOGRAM]
    edges = [round(i * 0.05, 6) for i in range(21)]
    assert rows["low"] == edges[:-1] * 2
    assert rows["high"] == edges[1:] * 2
    assert load_dict(f"runs:/{run_id}/report.json") == report.model_dump(mode="json")
    assert load_text(f"runs:/{run_id}/summary.md") == "Link relevance for tenant acme."


def test_a_run_without_scored_links_logs_no_histogram(local_mlflow: str) -> None:
    run_id = log_link_relevance(relevance_report(scored=False), "Nothing to score.")

    client = MlflowClient(local_mlflow)
    assert client.get_metric_history(run_id, "context_relevance_hist") == []
    assert {a.path for a in client.list_artifacts(run_id)} == {"report.json", "summary.md"}
    assert client.get_run(run_id).data.metrics["scored"] == 0


def test_the_metrics_are_exactly_the_counts_and_each_scores_statistics() -> None:
    stats = {"mean": 0.55, "p10": 0.2, "p25": 0.35, "p50": 0.5, "p75": 0.7, "p90": 0.8}

    assert link_relevance_metrics(relevance_report()) == {
        "links": 120,
        "scored": 85,
        "generic_anchors": 12,
        "without_anchor_vector": 48,
        "seconds": 3.5,
        "context_relevance_count": sum(CONTEXT_HISTOGRAM),
        **{f"context_relevance_{name}": value for name, value in stats.items()},
        "context_relevance_split": 0.52,
        "context_relevance_low_share": 0.45,
        "anchor_target_fit_count": sum(ANCHOR_HISTOGRAM),
        **{f"anchor_target_fit_{name}": value for name, value in stats.items()},
    }
    assert link_relevance_metrics(relevance_report(scored=False)) == {
        "links": 120,
        "scored": 0,
        "generic_anchors": 0,
        "without_anchor_vector": 0,
        "seconds": 3.5,
    }


def test_a_run_without_anchor_fits_logs_only_the_context_histogram(local_mlflow: str) -> None:
    run_id = log_link_relevance(relevance_report(anchor=False), "No anchor vectors yet.")

    client = MlflowClient(local_mlflow)
    assert client.get_metric_history(run_id, "anchor_target_fit_hist") == []
    history = sorted(
        client.get_metric_history(run_id, "context_relevance_hist"), key=lambda m: m.step
    )
    assert [(m.step, m.value) for m in history] == list(enumerate(map(float, CONTEXT_HISTOGRAM)))
    rows = table(run_id, "relevance_histogram.json")
    assert (rows["score"], rows["count"]) == (["context_relevance"] * 20, list(CONTEXT_HISTOGRAM))
    assert not any(
        name.startswith("anchor_target_fit") for name in client.get_run(run_id).data.metrics
    )


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


# ── hub bridges ─────────────────────────────────────────────────────────────

TREE, NEAREST, GAP = BridgeReason.SPANNING_TREE, BridgeReason.NEAREST_HUB, BridgeReason.BRIDGE_GAP
SHARED = ("trail shoes", "waterproof boots")


def bridges_report() -> BridgeReport:
    return BridgeReport(
        tenant_id="acme",
        floor_share=0.05,
        hubs=4,
        noise_pages=7,
        hub_pairs=2,
        components_before=3,
        components_after=1,
        directions_below_floor=3,
        links_needed=5,
        bridge_links=4,
        alternatives=6,
        directions_short=1,
        by_reason={TREE: 2, NEAREST: 1, GAP: 1},
        gsc_used=True,
        seconds=0.75,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def bridge_pairs() -> list[HubPair]:
    return [
        HubPair(
            language="en",
            hub_a=0,
            hub_b=3,
            size_a=40,
            size_b=12,
            pages_ab=1,
            pages_ba=0,
            link_density=0.002,
            centroid_cosine=0.7,
            query_jaccard=0.25,
            shared_queries=SHARED,
            bridge_gap=0.33,
            reasons=(TREE, GAP),
        ),
        HubPair(
            language=None,
            hub_a=1,
            hub_b=2,
            size_a=5,
            size_b=6,
            pages_ab=2,
            pages_ba=3,
            link_density=0.1,
            centroid_cosine=0.2,
            bridge_gap=-4.8,
        ),
    ]


def test_bridge_metrics_are_every_count_and_the_pairs_per_reason() -> None:
    assert bridge_metrics(bridges_report()) == {
        "hubs": 4,
        "noise_pages": 7,
        "hub_pairs": 2,
        "components_before": 3,
        "components_after": 1,
        "directions_below_floor": 3,
        "links_needed": 5,
        "bridge_links": 4,
        "alternatives": 6,
        "directions_short": 1,
        "gsc_used": 1,
        "seconds": 0.75,
        "pairs_spanning_tree": 2,
        "pairs_nearest_hub": 1,
        "pairs_bridge_gap": 1,
    }


def test_the_hub_pair_table_counts_shared_queries_without_naming_them() -> None:
    found = hub_pair_table(bridge_pairs())

    assert found["hub_a"] == [0, 1]
    assert found["language"] == ["en", None]
    assert found["shared_query_count"] == [2, 0]
    assert found["reasons"] == ["SPANNING_TREE,BRIDGE_GAP", ""]
    assert found["query_jaccard"] == [0.25, None]
    assert "shared_queries" not in found
    cells = " ".join(str(value) for column in found.values() for value in column)
    assert not [query for query in SHARED if query in cells]


def test_a_bridge_run_logs_counts_and_the_pair_table_but_no_queries_or_urls(
    local_mlflow: str,
) -> None:
    report = bridges_report()

    run_id = log_bridges(report, bridge_pairs(), "Hub bridges for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert (run.info.run_name, run.data.tags["stage"]) == ("hub bridges", "hub-bridges")
    assert run.data.tags["mlflow.note.content"] == "Hub bridges for tenant acme."
    assert run.data.params == {
        "floor_share": "0.05",
        "nearest_hubs": "2",
        "top_gap_pairs": "3",
        "alternatives": "2",
        "density_weight": "50.0",
        "relevance_decimals": "2",
        "cosine_weight": "0.4",
        "jaccard_weight": "0.6",
        "shared_queries": "10",
    }
    assert run.data.metrics == bridge_metrics(report)
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "hub_pairs.json",
        "report.json",
        "summary.md",
    }
    rows = table(run_id, "hub_pairs.json")
    assert (rows["hub_a"], rows["hub_b"], rows["shared_query_count"]) == ([0, 1], [3, 2], [2, 0])
    assert load_dict(f"runs:/{run_id}/report.json") == report.model_dump(mode="json")
    logged = " ".join(
        [
            *map(str, run.data.params.values()),
            *map(str, run.data.tags.values()),
            *run.data.metrics,
            *(str(v) for column in rows.values() for v in column),
            load_text(f"runs:/{run_id}/report.json"),
            load_text(f"runs:/{run_id}/summary.md"),
        ]
    )
    assert not [query for query in SHARED if query in logged], "shared queries leaked into the run"


# ── anchor extraction ───────────────────────────────────────────────────────

EXACT, STEMMED, STEM_SET = AnchorRung.EXACT, AnchorRung.STEMMED, AnchorRung.STEM_SET
JACCARD_BINS = (*([0] * 10), 1, *([0] * 8), 1)
SENTENCE_BINS = (2, 1, 1, 1, 1, 1, 1)


def anchors_report() -> AnchorReport:
    return AnchorReport(
        tenant_id="acme",
        stem_set_threshold=0.5,
        pairs=10,
        bridge_pairs=2,
        pairs_with_keywords=8,
        pairs_matched=5,
        primary_matched=4,
        matches=8,
        by_rung={EXACT: 4, STEMMED: 2, STEM_SET: 2},
        best_rung={EXACT: 3, STEMMED: 1, STEM_SET: 1},
        by_keyword_rank={1: 4, 2: 3, 3: 1},
        by_rung_and_rank={EXACT: {1: 3, 2: 1}, STEMMED: {1: 1, 2: 1}, STEM_SET: {2: 1, 3: 1}},
        stem_jaccard_histogram=JACCARD_BINS,
        sentence_index_histogram=SENTENCE_BINS,
        overlapping_existing_anchors=3,
        existing_anchors_located=4,
        existing_anchors_unlocated=2,
        identifier_mismatches=3,
        single_token_keywords=2,
        source_pages=6,
        sources_without_body=1,
        stemmed_languages={"en": 4, "de": 1},
        unstemmed_languages={"und": 1},
        seconds=0.5,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_anchor_metrics_are_counts_rungs_ranks_and_match_shares() -> None:
    assert anchor_metrics(anchors_report()) == {
        "pairs": 10,
        "bridge_pairs": 2,
        "pairs_with_keywords": 8,
        "pairs_matched": 5,
        "primary_matched": 4,
        "matches": 8,
        "overlapping_existing_anchors": 3,
        "existing_anchors_located": 4,
        "existing_anchors_unlocated": 2,
        "identifier_mismatches": 3,
        "single_token_keywords": 2,
        "source_pages": 6,
        "sources_without_body": 1,
        "seconds": 0.5,
        "rung_exact": 4,
        "rung_stemmed": 2,
        "rung_stem_set": 2,
        "best_rung_exact": 3,
        "best_rung_stemmed": 1,
        "best_rung_stem_set": 1,
        "keyword_rank_1": 4,
        "keyword_rank_2": 3,
        "keyword_rank_3": 1,
        "matched_share": 5 / 8,
        "primary_matched_share": 4 / 8,
    }
    empty = AnchorReport(
        tenant_id="acme",
        stem_set_threshold=0.5,
        pairs=3,
        bridge_pairs=0,
        pairs_with_keywords=0,
        pairs_matched=0,
        primary_matched=0,
        matches=0,
        by_rung=dict.fromkeys(AnchorRung, 0),
        best_rung=dict.fromkeys(AnchorRung, 0),
        by_keyword_rank={},
        by_rung_and_rank={rung: {} for rung in AnchorRung},
        stem_jaccard_histogram=(0,) * 20,
        sentence_index_histogram=(0,) * 7,
        overlapping_existing_anchors=0,
        existing_anchors_located=0,
        existing_anchors_unlocated=0,
        single_token_keywords=0,
        source_pages=2,
        sources_without_body=0,
        stemmed_languages={"en": 2},
        unstemmed_languages={},
        seconds=0.1,
        finished_at=datetime(2026, 9, 28, tzinfo=UTC),
    )
    none = anchor_metrics(empty)
    assert (none["pairs"], none["matches"]) == (3, 0)
    assert "matched_share" not in none, "no share without a pair that has keywords"
    assert "primary_matched_share" not in none


def test_the_anchor_tables_count_rung_by_rank_and_pages_by_language() -> None:
    report = anchors_report()

    assert anchor_rank_table(report) == {
        "rung": ["EXACT", "EXACT", "STEMMED", "STEMMED", "STEM_SET", "STEM_SET"],
        "keyword_rank": [1, 2, 1, 2, 2, 3],
        "matches": [3, 1, 1, 1, 1, 1],
    }
    assert anchor_language_table(report) == {
        "language": ["de", "en", "und"],
        "stemmed": [True, True, False],
        "source_pages": [1, 4, 1],
    }


def test_an_anchor_run_logs_counts_histograms_and_tables_but_no_text(local_mlflow: str) -> None:
    report = anchors_report()

    run_id = log_anchors(report, "Anchor extraction for tenant acme.")

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment("acme")
    assert (run.info.run_name, run.data.tags["stage"]) == ("anchor extraction", "anchor-extraction")
    params = run.data.params
    assert {
        name: params[name]
        for name in ("stem_set_threshold", "min_span", "max_span", "min_shared_stems")
    } == {"stem_set_threshold": "0.5", "min_span": "2", "max_span": "5", "min_shared_stems": "2"}
    assert params["max_inner_stop_words"] == "1"
    assert params["sentence_index_bins"] == "0,1,2,3,6,11,21"
    assert {"en", "de", "fr", "es"} <= set(params["stemmer_languages"].split(","))
    assert anchor_metrics(report).items() <= run.data.metrics.items()
    for name, bins in (("stem_jaccard_hist", JACCARD_BINS), ("sentence_index_hist", SENTENCE_BINS)):
        history = sorted(client.get_metric_history(run_id, name), key=lambda m: m.step)
        assert [(m.step, m.value) for m in history] == list(enumerate(map(float, bins))), name
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "rung_by_rank.json",
        "languages.json",
        "report.json",
        "summary.md",
    }
    assert table(run_id, "rung_by_rank.json") == anchor_rank_table(report)
    assert table(run_id, "languages.json") == anchor_language_table(report)
    assert load_dict(f"runs:/{run_id}/report.json") == report.model_dump(mode="json")
