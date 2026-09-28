"""The quality-eval MLflow run: tags, params, stable metrics, step-indexed histograms, tables,
and the lookup of the tenant's previous run that alerts compare against."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import pytest
import quality_factories as make
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text
from mlflow.entities import Metric

from linking_engine.ml.quality import LINK_DERIVED_COLUMNS, quality_metrics
from linking_engine.ml.tracking import (
    analytics_experiment,
    log_quality,
    previous_quality_run,
    quality_step_metrics,
    quality_tables,
)
from linking_engine.models import FeatureAuc

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.models import QualityReport

TENANT = "acme"
SUMMARY = "Quality evaluation for tenant acme."
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf", ".html")


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs go to a throwaway local store, never the remote server."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def table(run_id: str, name: str) -> dict[str, list[object]]:
    stored = load_dict(f"runs:/{run_id}/{name}")
    return {
        column: [row[i] for row in stored["data"]] for i, column in enumerate(stored["columns"])
    }


def history(client: MlflowClient, run_id: str, name: str) -> list[tuple[int, float]]:
    return sorted((m.step, m.value) for m in client.get_metric_history(run_id, name))


def test_a_run_carries_versions_tags_params_metrics_histograms_and_tables(
    local_mlflow: str,
) -> None:
    report = make.report(TENANT)

    run_id = log_quality(report, SUMMARY)

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert client.get_experiment(run.info.experiment_id).name == analytics_experiment(TENANT)
    assert run.info.run_name == "quality eval"
    tags = run.data.tags
    assert {
        name: tags[name]
        for name in (
            "tenant_id",
            "kind",
            "stage",
            "git_sha",
            "feature_digest",
            "weights_version",
            "weights_hash",
            "not_applicable",
            "alerts",
            "baseline_run",
            "mlflow.note.content",
        )
    } == {
        "tenant_id": TENANT,
        "kind": "eval",
        "stage": "quality-eval",
        "git_sha": make.SHA,
        "feature_digest": make.DIGEST,
        "weights_version": "baseline-1",
        "weights_hash": make.WEIGHTS_HASH,
        "not_applicable": "none",
        "alerts": "none",
        "baseline_run": "none",
        "mlflow.note.content": SUMMARY,
    }
    assert run.data.params == {
        "hide_share": "0.1",
        "hide_seed": "42",
        "recall_ks": "10,20,50",
        "per_target": "50",
        "signal_margin": "0.05",
        "anchor_jaccard": "0.5",
        "alert_band": "0.2",
        "link_derived_columns": ",".join(LINK_DERIVED_COLUMNS),
    }
    metrics = quality_metrics(report)
    assert {name: run.data.metrics[name] for name in metrics} == pytest.approx(metrics)
    scorer, links = report.scorer, report.link_relevance
    assert scorer is not None
    assert links is not None
    assert links.anchor is not None
    for name, counts in (
        ("score_hist_hidden", scorer.hidden_histogram),
        ("score_hist_other", scorer.other_histogram),
        ("context_relevance_hist", links.context.histogram),
        ("anchor_target_fit_hist", links.anchor.histogram),
    ):
        assert history(client, run_id, name) == list(enumerate(map(float, counts))), name
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert artifacts == {
        "report.json",
        "metrics.json",
        "summary.md",
        "recall.json",
        "feature_auc.json",
        "score_histogram.json",
        "keyword_rungs.json",
        "keyword_rejections.json",
        "keyword_relevance.json",
        "keyword_relevance_by_origin.json",
        "keyword_relevance_by_length.json",
        "relevance_histogram.json",
    }
    assert not [a for a in artifacts if a.endswith(IMAGE_SUFFIXES)], "image artifacts"
    assert load_dict(f"runs:/{run_id}/report.json") == report.model_dump(mode="json")
    assert load_dict(f"runs:/{run_id}/metrics.json") == pytest.approx(metrics)
    assert load_text(f"runs:/{run_id}/summary.md") == SUMMARY


def test_the_tables_hold_the_per_item_results(local_mlflow: str) -> None:
    run_id = log_quality(make.report(TENANT), SUMMARY)

    recall = table(run_id, "recall.json")
    assert recall == {"k": [10, 20, 50], "recall": [0.5, 0.75, 1.0], "random": [0.25, 0.5, 1.0]}
    auc = table(run_id, "feature_auc.json")
    assert auc["column"] == ["content_cosine", "same_hub", "target_ctr_gap"]
    assert auc["auc"] == [0.9, 0.52, None]
    assert auc["signal"] == [True, False, False]
    assert auc["link_derived"] == [c in LINK_DERIVED_COLUMNS for c in auc["column"]]
    scores = table(run_id, "score_histogram.json")
    assert (scores["bin"], scores["hidden"]) == (list(range(20)), [0] * 19 + [3])
    assert (scores["low"][0], scores["high"][-1]) == (0.0, 100.0)
    rungs = table(run_id, "keyword_rungs.json")
    assert dict(zip(rungs["rung"], rungs["pages"], strict=True)) == {"STRATEGIC": 1, "H1": 3}
    rejected = table(run_id, "keyword_rejections.json")
    assert dict(zip(rejected["reason"], rejected["fallbacks"], strict=True)) == {
        "generic": 2,
        "repeated": 1,
    }
    ranks = table(run_id, "keyword_relevance.json")
    assert (ranks["rank"], ranks["keywords"], ranks["mean"]) == ([1, 2], [3, 2], [0.6, 0.4])
    assert table(run_id, "keyword_relevance_by_origin.json") == {
        "origin": ["primary_h1", "secondary_gsc_observed"],
        "keywords": [3, 2],
        "mean": [0.6, 0.4],
        "median": [0.6, 0.4],
    }
    assert table(run_id, "keyword_relevance_by_length.json") == {
        "words": ["1_2", "3_4"],
        "keywords": [4, 1],
        "mean": [0.55, 0.3],
        "median": [0.6, 0.3],
    }
    histogram = table(run_id, "relevance_histogram.json")
    assert histogram["score"] == ["context_relevance"] * 20 + ["anchor_target_fit"] * 20
    assert histogram["count"] == [*make.CONTEXT_BINS, *make.CONTEXT_BINS]


def test_checks_that_do_not_apply_leave_no_metric_histogram_or_table(local_mlflow: str) -> None:
    report = make.report(
        TENANT,
        not_applicable=("retrieval", "feature_signal", "scorer", "keyword_relevance"),
        link_relevance=make.link_relevance(anchor=None),
    )

    run_id = log_quality(report, SUMMARY)

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert run.data.tags["not_applicable"] == "retrieval,feature_signal,scorer,keyword_relevance"
    assert not [name for name in run.data.metrics if name.startswith(("recall_", "auc_", "score_"))]
    assert history(client, run_id, "score_hist_hidden") == []
    assert history(client, run_id, "anchor_target_fit_hist") == []
    assert len(history(client, run_id, "context_relevance_hist")) == 20
    assert {a.path for a in client.list_artifacts(run_id)} == {
        "report.json",
        "metrics.json",
        "summary.md",
        "keyword_rungs.json",
        "keyword_rejections.json",
        "relevance_histogram.json",
    }
    assert set(quality_step_metrics(report)) == {"context_relevance_hist"}


def test_alerts_are_a_tag_a_table_and_the_baseline_run_is_named(local_mlflow: str) -> None:
    report = make.report(
        TENANT,
        baseline_run_id="run-0",
        alerts=(
            make.alert(),
            make.alert(metric="gsc_pair_share", previous=0.0, current=0.25, change=None),
        ),
    )

    run_id = log_quality(report, SUMMARY)

    tags = MlflowClient(local_mlflow).get_run(run_id).data.tags
    assert (tags["alerts"], tags["baseline_run"]) == ("recall_at_10,gsc_pair_share", "run-0")
    alerts = table(run_id, "alerts.json")
    assert alerts["metric"] == ["recall_at_10", "gsc_pair_share"]
    assert alerts["change"] == [-0.5, None]
    assert alerts["relative"] == [True, True]


def test_no_rejected_fallback_leaves_out_the_rejection_table() -> None:
    report = make.report(TENANT, keywords=make.keywords(fallbacks_rejected={}))

    assert "keyword_rejections.json" not in quality_tables(report)
    assert "alerts.json" not in quality_tables(report)


# ── the previous run ────────────────────────────────────────────────────────


def logged(report: QualityReport) -> str:
    run_id = log_quality(report, SUMMARY)
    # MLflow orders by end time in milliseconds; keep consecutive runs apart.
    time.sleep(0.01)
    return run_id


def test_there_is_no_baseline_before_the_first_run_and_no_experiment_is_created(
    local_mlflow: str,
) -> None:
    assert previous_quality_run(TENANT) is None
    assert MlflowClient(local_mlflow).get_experiment_by_name(analytics_experiment(TENANT)) is None


def test_the_baseline_is_the_tenants_latest_finished_quality_run(local_mlflow: str) -> None:
    logged(make.report(TENANT, seconds=1.0))
    latest = logged(make.report(TENANT, seconds=2.0))
    # Later runs that must not count: another stage, a failed run and a running one.
    log_scores_run = _other_stage_run(local_mlflow)
    client = MlflowClient(local_mlflow)
    experiment = client.get_experiment_by_name(analytics_experiment(TENANT))
    assert experiment is not None
    for status in ("FAILED", "RUNNING"):
        run = client.create_run(
            experiment.experiment_id, tags={"stage": "quality-eval", "tenant_id": TENANT}
        )
        client.log_metric(run.info.run_id, "recall_at_10", 0.0)
        if status == "FAILED":
            client.set_terminated(run.info.run_id, status="FAILED")

    baseline = previous_quality_run(TENANT)

    assert baseline is not None
    assert baseline.run_id == latest
    assert log_scores_run != latest
    assert baseline.metrics["seconds"] == 2.0
    expected = quality_metrics(make.report(TENANT, seconds=2.0))
    assert {name: baseline.metrics[name] for name in expected} == pytest.approx(expected)


def _other_stage_run(uri: str) -> str:
    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name(analytics_experiment(TENANT))
    assert experiment is not None
    run = client.create_run(experiment.experiment_id, tags={"stage": "score-pairs"})
    client.log_metric(run.info.run_id, "recall_at_10", 0.0)
    client.set_terminated(run.info.run_id)
    return str(run.info.run_id)


def test_a_run_of_another_tenant_in_the_experiment_is_never_the_baseline(
    local_mlflow: str,
) -> None:
    mine = logged(make.report(TENANT))
    client = MlflowClient(local_mlflow)
    experiment = client.get_experiment_by_name(analytics_experiment(TENANT))
    assert experiment is not None
    stray = client.create_run(
        experiment.experiment_id, tags={"stage": "quality-eval", "tenant_id": "other"}
    )
    client.set_terminated(stray.info.run_id)
    logged(make.report("other"))

    baseline = previous_quality_run(TENANT)

    assert baseline is not None
    assert baseline.run_id == mine


def test_non_finite_metrics_of_the_previous_run_are_dropped(local_mlflow: str) -> None:
    run_id = logged(make.report(TENANT))
    MlflowClient(local_mlflow).log_batch(
        run_id, metrics=[Metric("recall_at_20", math.nan, int(time.time() * 1000), 1)]
    )

    baseline = previous_quality_run(TENANT)

    assert baseline is not None
    assert "recall_at_20" not in baseline.metrics
    assert baseline.metrics["recall_at_10"] == 0.5


def test_a_feature_auc_signal_flag_matches_the_models_rule() -> None:
    report = make.report(
        TENANT,
        feature_signal=make.feature_signal(
            columns=(FeatureAuc(column="same_hub", auc=0.45, coverage=1.0, ranker_auc=0.55),),
            features_with_signal=1,
        ),
    )

    assert quality_tables(report)["feature_auc.json"]["signal"] == [True]
