"""The ranker's MLflow run and registry on a throwaway sqlite store: tags, params, stable
metrics, step metrics, tables, artefacts and the registered model; the production alias that
moves only where promotion is allowed and only to the tenant's own version; the local model
copy; and a registry that cannot be reached."""

from __future__ import annotations

import json
import os
import pickle
import shutil
import traceback
from typing import TYPE_CHECKING

import numpy as np
import pytest
import ranking_factories as make
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text
from mlflow.exceptions import MlflowException
from ranking_seed import graded_frame

from linking_engine.ml.quality import LINK_DERIVED_COLUMNS
from linking_engine.ml.ranker_tracking import (
    ALIAS,
    PROMOTION_ENV,
    PROXY_ENV,
    RegistryUnavailableError,
    holder,
    load_production,
    log_ranker,
    move_alias,
    promotion_allowed,
    ranker_experiment,
    ranker_metrics,
    registered_model,
    save_local,
    tag_promoted,
    trained_holder,
)
from linking_engine.ml.ranking import (
    PLACEMENT_COLUMNS,
    Trained,
    split_sources,
    summarise_ranker,
    train,
)
from linking_engine.ml.tracking import EXPERIMENT_KIND, EXPERIMENT_KIND_TAG
from linking_engine.models import HeldOutSettings, RankerParams, ScorerName

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.models import RankerReport

TENANT = "acme"
# The unreachable registry's host; it must never reach a message or a traceback.
ADDRESS = "127.0.0.1"
OTHER = "acme-other"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf", ".html")


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs and models go to a throwaway local store, never the remote server; promotion is
    off as on a developer machine."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.delenv(PROMOTION_ENV, raising=False)
    return uri


@pytest.fixture(scope="module")
def trained() -> Trained:
    frame = graded_frame()
    valid = frame["source_url"].isin(split_sources(frame["source_url"], share=0.3, seed=7))
    return train(frame[~valid], frame[valid], make.COLUMNS, RankerParams(max_rounds=30))


def logged(report: RankerReport, model: Trained | None) -> tuple[str, str | None]:
    return log_ranker(report, model, summarise_ranker(report))


def history(client: MlflowClient, run_id: str, name: str) -> list[tuple[int, float]]:
    return sorted((m.step, m.value) for m in client.get_metric_history(run_id, name))


def table(run_id: str, name: str) -> dict[str, list[object]]:
    stored = load_dict(f"runs:/{run_id}/{name}")
    return {
        column: [row[i] for row in stored["data"]] for i, column in enumerate(stored["columns"])
    }


def test_names_are_per_tenant_and_promotion_needs_the_exact_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (ranker_experiment(TENANT), registered_model(TENANT)) == (
        "ranker-acme",
        "link-ranker-acme",
    )
    assert registered_model(OTHER) != registered_model(TENANT)
    assert ALIAS == "production"
    monkeypatch.delenv(PROMOTION_ENV, raising=False)
    assert not promotion_allowed()
    for value in ("ALLOWED", "true", "1", "allowed "):
        monkeypatch.setenv(PROMOTION_ENV, value)
        assert not promotion_allowed(), value
    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    assert promotion_allowed()


def test_train_ranker_run_logs_tags_params_metrics_tables_and_registers(
    local_mlflow: str, trained: Trained
) -> None:
    report = make.report(TENANT, best_iteration=trained.best_iteration)

    run_id, version = logged(report, trained)

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.name == ranker_experiment(TENANT)
    assert experiment.tags[EXPERIMENT_KIND_TAG] == EXPERIMENT_KIND
    tags = run.data.tags
    assert {
        name: tags[name]
        for name in (
            "tenant_id",
            "issue",
            "corpus_pages",
            "body_links",
            "seed",
            "split_seed",
            "feature_set_version",
            "git_sha",
            "skipped_reason",
            "promoted",
            "registered_version",
        )
    } == {
        "tenant_id": TENANT,
        "issue": "24-27",
        "corpus_pages": "80",
        "body_links": "432",
        "seed": "42",
        "split_seed": "7",
        "feature_set_version": make.DIGEST,
        "git_sha": make.SHA,
        "skipped_reason": "none",
        "promoted": "false",
        "registered_version": version,
    }
    assert "model_version" not in tags, "a run is tagged with a model version before promotion"
    assert run.data.metrics["promoted"] == 0.0
    assert tags["mlflow.note.content"] == summarise_ranker(report)
    params = run.data.params
    assert (params["rounds"], params["share"], params["test_share"]) == ("2", "0.1", "0.2")
    assert (params["learning_rate"], params["num_leaves"], params["eval_at"]) == (
        "0.05",
        "31",
        "10",
    )
    assert params["excluded_columns"] == "target_crawl_depth"
    assert params["placement_columns"] == ",".join(PLACEMENT_COLUMNS)
    assert params["link_derived_columns"] == ",".join(LINK_DERIVED_COLUMNS)

    metrics = ranker_metrics(report)
    assert {name: run.data.metrics[name] for name in metrics} == pytest.approx(metrics)
    for scorer in (ScorerName.LEARNED, ScorerName.LEARNED_EXCL_LINK_COUNTS, ScorerName.BASELINE):
        for suffix in ("ndcg_at_10", "ndcg_at_10_ci_low", "ndcg_at_10_ci_high", "precision_at_5"):
            assert f"{scorer.value}_{suffix}" in run.data.metrics
    assert run.data.metrics["promotion_delta_ci_low"] == pytest.approx(0.03)
    learned = report.metrics[0]
    assert history(client, run_id, "learned_ndcg_at_10_by_round") == sorted(
        learned.per_round.items()
    )
    assert history(client, run_id, "learned_ndcg_hist") == list(
        enumerate(map(float, learned.histogram))
    )
    assert history(client, run_id, "round_positives") == [(0, 36.0), (1, 30.0)]
    assert history(client, run_id, "round_positive_placement_share") == [(0, 0.9), (1, 0.9)]
    assert history(client, run_id, "round_negative_placement_share") == [(0, 0.05), (1, 0.05)]
    assert run.data.metrics["placement_gain_share"] == 0.0

    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {
        "summary.md",
        "metrics.json",
        "columns.json",
        "report.json",
        "importance.json",
        "rounds.json",
        "ranking_metrics.json",
    } <= artifacts
    assert not [a for a in artifacts if a.endswith(IMAGE_SUFFIXES)], "image artifacts"
    assert load_text(f"runs:/{run_id}/summary.md") == summarise_ranker(report)
    assert load_dict(f"runs:/{run_id}/metrics.json") == pytest.approx(metrics)
    assert load_dict(f"runs:/{run_id}/columns.json") == {
        "columns": list(make.COLUMNS),
        "excluded": {"target_crawl_depth": "stored BFS over links"},
    }
    importance = table(run_id, "importance.json")
    assert importance["column"] == list(make.COLUMNS)
    assert importance["placement"] == [False, False, False]
    rounds = table(run_id, "rounds.json")
    assert (rounds["round"], rounds["positives"], rounds["negatives"]) == (
        [0, 1],
        [36, 30],
        [2964, 2970],
    )
    assert rounds["positive_placement_share"] == [0.9, 0.9]
    assert rounds["negative_placement_share"] == [0.05, 0.05]
    assert table(run_id, "ranking_metrics.json")["scorer"] == [
        "learned",
        "learned_excl_link_counts",
        "baseline",
    ]

    assert version == "1"
    stored = client.get_model_version(registered_model(TENANT), version)
    assert stored.tags["tenant_id"] == TENANT
    assert stored.tags["feature_set_version"] == make.DIGEST
    assert stored.run_id == run_id
    assert client.get_registered_model(registered_model(TENANT)).aliases == {}, (
        "logging a run moved the alias"
    )


def test_a_skipped_run_is_logged_without_a_model(local_mlflow: str) -> None:
    report = make.skipped(TENANT)

    run_id, version = logged(report, None)

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    assert version is None
    assert run.data.tags["skipped_reason"] == report.skipped_reason
    assert run.data.metrics["skipped"] == 1.0
    assert "learned_ndcg_at_10" not in run.data.metrics
    assert "model" not in {a.path for a in client.list_artifacts(run_id)}
    with pytest.raises(MlflowException, match=r"not found|does not exist"):
        client.get_registered_model(registered_model(TENANT))


def test_log_ranker_refuses_a_model_that_does_not_match_the_report(
    local_mlflow: str, trained: Trained
) -> None:
    with pytest.raises(ValueError, match="exactly when the run was not skipped"):
        logged(make.report(TENANT), None)
    with pytest.raises(ValueError, match="exactly when the run was not skipped"):
        logged(make.skipped(TENANT), trained)
    with pytest.raises(ValueError, match="columns differ"):
        logged(make.report(TENANT, columns=(*make.COLUMNS, "target_kw_count")), trained)


def test_move_alias_only_where_allowed_and_only_to_this_tenants_version(
    local_mlflow: str, trained: Trained, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, version = logged(make.report(TENANT), trained)
    _, other_version = logged(make.report(OTHER), trained)
    assert version is not None
    assert other_version is not None
    client = MlflowClient(local_mlflow)

    with pytest.raises(RuntimeError, match=PROMOTION_ENV):
        move_alias(TENANT, version)
    assert holder(TENANT) is None, "the alias moved where promotion is not allowed"

    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    move_alias(TENANT, version)

    found = holder(TENANT)
    assert found is not None
    assert (found.version, found.columns) == (version, make.COLUMNS)
    rows = graded_frame().loc[:, list(make.COLUMNS)].to_numpy(dtype=np.float32)[:500]
    assert found.booster.predict(rows) == pytest.approx(trained.booster.predict(rows))
    assert holder(OTHER) is None, "the tenant's promotion reached another tenant"

    # A version of the tenant's model tagged with another tenant is never promoted.
    client.set_model_version_tag(registered_model(TENANT), version, "tenant_id", OTHER)
    with pytest.raises(ValueError, match="not tagged with this tenant"):
        move_alias(TENANT, version)
    with pytest.raises(ValueError, match="not tagged with this tenant"):
        holder(TENANT)


def test_load_production_reads_the_local_copy_first_and_sets_proxy_downloads(
    local_mlflow: str, trained: Trained, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    assert load_production(TENANT, cache) is None, "nothing promoted yet"
    run_id, version = logged(make.report(TENANT), trained)
    assert version is not None
    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    move_alias(TENANT, version)
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)

    downloaded = load_production(TENANT, cache)

    assert downloaded is not None
    assert all(os.environ[name] == "false" for name in PROXY_ENV)
    model_path = cache / TENANT / "ranker" / f"model-{version}.txt"
    columns_path = cache / TENANT / "ranker" / f"model-{version}.columns.json"
    assert model_path.is_file()
    assert json.loads(columns_path.read_text()) == {
        "version": version,
        "run_id": run_id,
        "columns": list(make.COLUMNS),
    }
    # A copy of the same version number from another training run is not trusted.
    columns_path.write_text(
        json.dumps({"version": version, "run_id": "0" * 32, "columns": list(make.COLUMNS)})
    )
    again = load_production(TENANT, cache)
    assert again is not None
    assert again.run_id == run_id
    assert json.loads(columns_path.read_text())["run_id"] == run_id, "the copy was not replaced"
    assert not [p for p in model_path.parent.iterdir() if p.name.endswith(".partial")]

    # The registered artefacts are gone: only the local copy can serve the model.
    stores = [tmp_path / name for name in ("mlruns", "mlartifacts") if (tmp_path / name).is_dir()]
    assert stores, "the registered artefacts are not in the local store"
    for store in stores:
        shutil.rmtree(store)
    local = load_production(TENANT, cache)
    assert local is not None
    assert (local.version, local.columns) == (version, make.COLUMNS)

    # A local copy of another version, or a broken one, is not trusted.
    columns_path.write_text(json.dumps({"version": "99", "columns": list(make.COLUMNS)}))
    with pytest.raises(RegistryUnavailableError):
        load_production(TENANT, cache)

    with pytest.raises(ValueError, match="directory name"):
        load_production("../escape", cache)


def test_proxy_settings_already_made_are_kept(
    local_mlflow: str, trained: Trained, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in PROXY_ENV:
        monkeypatch.setenv(name, "true")

    logged(make.report(TENANT), trained)

    assert all(os.environ[name] == "true" for name in PROXY_ENV)


def test_save_local_writes_the_model_and_its_columns_per_tenant(
    trained: Trained, tmp_path: Path
) -> None:
    found = trained_holder(make.report(TENANT), trained, "0" * 32, "4")

    path = save_local(TENANT, tmp_path, found)

    assert path == tmp_path / TENANT / "ranker" / "model-4.txt"
    assert sorted(p.name for p in path.parent.iterdir()) == ["model-4.columns.json", "model-4.txt"]
    assert json.loads((path.parent / "model-4.columns.json").read_text()) == {
        "version": "4",
        "run_id": "0" * 32,
        "columns": list(make.COLUMNS),
    }
    with pytest.raises(ValueError, match="directory name"):
        save_local("..", tmp_path, found)


def test_an_unreachable_registry_never_shows_its_address(
    monkeypatch: pytest.MonkeyPatch, trained: Trained, tmp_path: Path
) -> None:
    """Every registry call wraps the failure without its cause, whose text names the server;
    the formatted traceback, as a log or a Prefect run would print it, holds no address."""
    # Port 9 refuses at once; no retries, so the failure is quick.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"http://{ADDRESS}:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    calls = {
        "load_production": lambda: load_production(TENANT, tmp_path),
        "holder": lambda: holder(TENANT),
        "move_alias": lambda: move_alias(TENANT, "1"),
        "log_ranker": lambda: logged(make.report(TENANT), trained),
        "tag_promoted": lambda: tag_promoted("0" * 32, "1"),
    }

    for name, call in calls.items():
        with pytest.raises(RegistryUnavailableError) as raised:
            call()
        error = raised.value
        printed = "".join(traceback.format_exception(error))
        assert ADDRESS not in printed, f"{name}: the server's address in the traceback"
        assert error.__cause__ is None, name
        assert error.__suppress_context__, name
        assert error.cause_type, name
        assert str(error).startswith("model registry failed: "), name
        copied = pickle.loads(pickle.dumps(error))  # noqa: S301 - bytes this test made
        assert (type(copied), str(copied), copied.cause_type, copied.error_code) == (
            RegistryUnavailableError,
            str(error),
            error.cause_type,
            error.error_code,
        ), f"{name}: the error does not survive pickling"


def test_tag_promoted_marks_the_run_only_after_the_alias_moved(
    local_mlflow: str, trained: Trained, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, version = logged(make.report(TENANT), trained)
    assert version is not None
    client = MlflowClient(local_mlflow)
    assert client.get_run(run_id).data.tags["promoted"] == "false"
    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    move_alias(TENANT, version)

    tag_promoted(run_id, version)

    run = client.get_run(run_id)
    assert (run.data.tags["promoted"], run.data.tags["model_version"]) == ("true", version)
    assert history(client, run_id, "promoted")[-1][1] == 1.0


def test_a_holder_carries_how_it_was_trained_from_its_version_tags(
    local_mlflow: str, trained: Trained, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = make.report(TENANT, settings=HeldOutSettings(rounds=2, split_seed=11, test_share=0.3))
    run_id, version = logged(report, trained)
    assert version is not None
    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    move_alias(TENANT, version)

    found = holder(TENANT)

    assert found is not None
    assert (found.run_id, found.feature_set_version) == (run_id, make.DIGEST)
    assert (found.split_seed, found.test_share, found.valid_share) == (11, 0.3, 0.1)
    client = MlflowClient(local_mlflow)
    client.set_model_version_tag(registered_model(TENANT), version, "split_seed", "seven")
    client.delete_model_version_tag(registered_model(TENANT), version, "test_share")
    blurred = holder(TENANT)
    assert blurred is not None
    assert (blurred.split_seed, blurred.test_share, blurred.valid_share) == (None, None, 0.1)
