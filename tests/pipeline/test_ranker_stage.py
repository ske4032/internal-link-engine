"""train_ranker and rank_pairs on real Neo4j and Mongo with a throwaway sqlite MLflow store: the
guards, the logged and registered run, the promotion gate against the baseline and the holder,
the local run that never moves the alias, and ranking with the production model or the baseline
with its reason. Anchor selection's outputs are unchanged by the held-out view."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas
import pyarrow.parquet as pq
import pytest
import ranking_factories as make
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from ranking_seed import KEYWORDS, URLS, graded_frame, seed_ranking, voyage
from selection_seed import seed_selection
from selection_seed import voyage as selection_voyage
from store_state import graph_state, mongo_state
from structlog.testing import capture_logs
from test_ranking_data_stage import WeightsOnly, files
from voyage_fakes import client

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.discovery.features import KEY_COLUMNS, code_digest
from linking_engine.discovery.scoring import default_weights, score_frame
from linking_engine.ml.ranker_tracking import (
    ALIAS,
    PROMOTION_ENV,
    RegistryUnavailableError,
    log_ranker,
    move_alias,
    ranker_experiment,
    registered_model,
)
from linking_engine.ml.ranking import (
    EXCL_PLACEMENT_COLUMNS,
    LABEL_COLUMN,
    LIKE_FOR_LIKE_COLUMNS,
    MIN_TEST_GROUPS,
    MIN_TRAIN_GROUPS,
    MODEL_COLUMNS,
    PLACEMENT_COLUMNS,
    Trained,
    positive_groups,
    split_sources,
    summarise_ranker,
    train,
)
from linking_engine.models import HeldOutSettings, RankerParams, ScorerName
from linking_engine.pipeline import ranker
from linking_engine.pipeline.anchor_selection import compute_anchor_choices, select_anchors
from linking_engine.pipeline.anchors import AnchorView
from linking_engine.pipeline.features import assemble_features
from linking_engine.pipeline.ranker import RANKED_SCHEMA, rank_pairs, train_ranker
from linking_engine.pipeline.ranking_data import load_rounds

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mlflow.entities import Run

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import RankerReport

# Five rounds of a fifth of the links hide every link once; a quarter of the source pages is
# enough test groups on the planted tenant.
ENOUGH = HeldOutSettings(rounds=5, share=0.2, test_share=0.25)
TOO_FEW = HeldOutSettings(rounds=1)
LEARNED_SCORERS = {
    ScorerName.LEARNED,
    ScorerName.LEARNED_EXCL_LINK_COUNTS,
    ScorerName.LEARNED_EXCL_PLACEMENT,
    ScorerName.BASELINE,
}


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Runs and models go to a throwaway local store, never the remote server; promotion is
    off as on a developer machine."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    monkeypatch.delenv(PROMOTION_ENV, raising=False)
    monkeypatch.setenv("GIT_SHA", "c" * 40)
    return uri


def run_text(run_id: str, folder: Path) -> str:
    """Every tag, param, metric name and artefact file of a run, the model's included."""
    run = MlflowClient().get_run(run_id)
    root = download_artifacts(run_id=run_id, dst_path=str(folder))
    files = [path for path in sorted(folder.rglob("*")) if path.is_file()]
    assert files, root
    return " ".join(
        [
            *run.data.tags.values(),
            *run.data.params.values(),
            *run.data.metrics,
            *(path.read_text(encoding="utf-8", errors="ignore") for path in files),
        ]
    )


def run_of(tenant: str, version: str) -> Run:
    """The training run that registered ``version`` of the tenant's model."""
    client_ = MlflowClient()
    experiment = client_.get_experiment_by_name(ranker_experiment(tenant))
    assert experiment is not None
    [run] = client_.search_runs(
        [experiment.experiment_id], f"tags.registered_version = '{version}'"
    )
    return run


def aliases(tenant: str) -> dict[str, str]:
    found = MlflowClient().get_registered_model(registered_model(tenant)).aliases
    return {alias: str(version) for alias, version in found.items()}


def groups_of(frame: pandas.DataFrame, sources: frozenset[str] | set[str]) -> int:
    chosen = frame[frame["source_url"].isin(sources)]
    return len(chosen.drop_duplicates(["round", "source_url"]))


@pytest.mark.integration
async def test_train_ranker_skips_with_reason_when_too_few_groups(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path, local_mlflow: str
) -> None:
    await seed_ranking(graph, mongo, tenant)
    stored = (await graph_state(graph, tenant), await mongo_state(mongo))

    with capture_logs() as logs:
        report = await train_ranker(
            graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage()), settings=TOO_FEW
        )

    assert (await graph_state(graph, tenant), await mongo_state(mongo)) == stored
    assert report.skipped_reason is not None
    assert report.train_groups < MIN_TRAIN_GROUPS
    assert f"fewer than {MIN_TRAIN_GROUPS}" in report.skipped_reason
    assert str(report.train_groups) in report.skipped_reason
    assert (report.best_iteration, report.promotion, report.model_version) == (None, None, None)
    assert (report.metrics, report.importance) == ((), ())
    assert report.positives == report.rounds[0].positives > 0
    client_ = MlflowClient(local_mlflow)
    experiment = client_.get_experiment_by_name(ranker_experiment(tenant))
    assert experiment is not None
    [run] = client_.search_runs([experiment.experiment_id])
    assert run.data.tags["skipped_reason"] == report.skipped_reason
    assert client_.search_registered_models() == [], "a skipped run registered a model"
    assert not (tmp_path / tenant / "ranker" / "model-1.txt").exists()
    [line] = [entry for entry in logs if entry["event"] == "ranker.trained"]
    assert (line["tenant_id"], line["skipped"], line["promoted"]) == (tenant, True, False)


@pytest.mark.integration
async def test_train_ranker_end_to_end_logs_and_registers(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    local_mlflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_ranking(graph, mongo, tenant)
    cache = tmp_path / "cache"
    stored = (await graph_state(graph, tenant), await mongo_state(mongo))
    models: list[Trained] = []

    def recording(*args: Any, **kwargs: Any) -> Trained:
        found = train(*args, **kwargs)
        models.append(found)
        return found

    monkeypatch.setattr(ranker, "train", recording)

    with capture_logs() as logs:
        report = await train_ranker(
            graph, mongo, tenant, cache_dir=cache, voyage=client(voyage()), settings=ENOUGH
        )

    assert (await graph_state(graph, tenant), await mongo_state(mongo)) == stored
    assert report.skipped_reason is None, report.skipped_reason
    assert report.train_groups >= MIN_TRAIN_GROUPS
    assert report.test_groups >= MIN_TEST_GROUPS
    assert report.valid_groups > 0
    assert (report.corpus_pages, report.body_links) == (len(URLS), 432)
    assert report.git_sha == "c" * 40
    assert report.columns == MODEL_COLUMNS
    assert "target_crawl_depth" in report.excluded_columns
    assert [r.round for r in report.rounds] == list(range(ENOUGH.rounds))
    assert sum(r.hidden for r in report.rounds) == report.body_links, "five fifths hide all"

    # The split is by source page: recomputed from the round files, the counts agree.
    rows = load_rounds(sorted((cache / tenant / "ranker" / "rounds").iterdir()))
    sources = set(rows["source_url"])
    frame = positive_groups(rows)
    test = split_sources(sources, share=ENOUGH.test_share, seed=ENOUGH.split_seed)
    valid = split_sources(sources - test, share=ENOUGH.valid_share, seed=ENOUGH.split_seed + 1)
    assert not test & valid
    assert groups_of(frame, test) == report.test_groups
    assert groups_of(frame, valid) == report.valid_groups
    assert groups_of(frame, sources - test - valid) == report.train_groups

    # The full model, then without link counts, then without placement, on one split.
    assert [model.columns for model in models] == [
        MODEL_COLUMNS,
        LIKE_FOR_LIKE_COLUMNS,
        EXCL_PLACEMENT_COLUMNS,
    ]
    assert not set(PLACEMENT_COLUMNS) & set(models[2].booster.feature_name())
    assert set(PLACEMENT_COLUMNS) <= set(models[0].booster.feature_name())
    metrics = {entry.scorer: entry for entry in report.metrics}
    assert set(metrics) == LEARNED_SCORERS, "no holder yet"
    assert {entry.groups for entry in report.metrics} == {report.test_groups}
    assert {entry.column for entry in report.importance} == set(MODEL_COLUMNS)
    assert sum(entry.gain_share for entry in report.importance) == pytest.approx(1.0)
    # Synthetic pages cluster cleanly: a high score proves the plumbing, not the method.
    learned, baseline = metrics[ScorerName.LEARNED], metrics[ScorerName.BASELINE]
    assert learned.ndcg_at_10 > baseline.ndcg_at_10, (
        f"learned NDCG@10 {learned.ndcg_at_10:.3f} is not above the baseline's "
        f"{baseline.ndcg_at_10:.3f} on the planted tenant"
    )
    decision = report.promotion
    assert decision is not None
    assert (decision.rival, decision.holder_version) == (ScorerName.BASELINE, None)
    assert not decision.promoted, "a local run promoted its model"

    assert report.model_version == "1"
    client_ = MlflowClient(local_mlflow)
    version = client_.get_model_version(registered_model(tenant), "1")
    assert version.tags["tenant_id"] == tenant
    assert aliases(tenant) == {}
    [run] = client_.search_runs(
        [client_.get_experiment_by_name(ranker_experiment(tenant)).experiment_id]
    )
    assert run.data.tags["registered_version"] == "1"
    assert run.data.tags["promoted"] == "false"
    assert "model_version" not in run.data.tags, "a local run is tagged as the production model"
    assert run.data.tags["issue"] == "24-27"
    assert run.data.metrics["learned_ndcg_at_10"] == pytest.approx(learned.ndcg_at_10)
    assert run.data.tags["mlflow.note.content"] == summarise_ranker(
        report.model_copy(update={"model_version": None})
    )
    logged = run_text(run.info.run_id, tmp_path / "artifacts")
    assert [u for u in URLS if u in logged] == [], "page urls reached the MLflow run"
    folded = logged.casefold()
    assert [k for k in KEYWORDS if k.casefold() in folded] == [], "keywords reached the run"

    [line] = [entry for entry in logs if entry["event"] == "ranker.trained"]
    assert (line["tenant_id"], line["skipped"], line["promoted"]) == (tenant, False, False)
    text = " ".join(str(value) for entry in logs for value in entry.values())
    assert [u for u in URLS if u in text] == [], "page urls in the log"


@pytest.mark.integration
async def test_promotion_gate_rejects_worse_model_and_local_runs_never_move_alias(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    local_mlflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_ranking(graph, mongo, tenant)

    async def trained(params: RankerParams | None = None) -> RankerReport:
        return await train_ranker(
            graph,
            mongo,
            tenant,
            cache_dir=tmp_path,
            voyage=client(voyage()),
            settings=ENOUGH,
            params=params,
        )

    local = await trained()
    assert local.promotion is not None
    assert local.promotion.would_promote, (
        f"the learned model does not beat the baseline on the planted tenant: "
        f"{local.promotion.reason}"
    )
    assert not local.promotion.promoted
    assert aliases(tenant) == {}, "a local run moved the alias"
    assert not list((tmp_path / tenant / "ranker").glob("model-*.txt"))

    # Promotion allowed, but the alias cannot move: the run never says promoted.
    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    moves = ranker.move_alias

    def refused(tenant_id: str, model_version: str) -> None:
        raise RegistryUnavailableError("ConnectionError")

    monkeypatch.setattr(ranker, "move_alias", refused)
    with pytest.raises(RegistryUnavailableError):
        await trained()
    tags = run_of(tenant, "2").data.tags
    assert (tags["promoted"], "model_version" in tags) == ("false", False)
    assert aliases(tenant) == {}
    assert not list((tmp_path / tenant / "ranker").glob("model-*.txt"))
    monkeypatch.setattr(ranker, "move_alias", moves)

    promoted = await trained()

    assert promoted.promotion is not None
    assert (promoted.promotion.rival, promoted.promotion.promoted) == (ScorerName.BASELINE, True)
    assert aliases(tenant) == {ALIAS: "3"}
    tags = run_of(tenant, "3").data.tags
    assert (tags["promoted"], tags["model_version"]) == ("true", "3")
    assert (tmp_path / tenant / "ranker" / "model-3.txt").is_file()
    copy = json.loads((tmp_path / tenant / "ranker" / "model-3.columns.json").read_text())
    assert copy["run_id"] == run_of(tenant, "3").info.run_id

    same = await trained()

    assert same.promotion is not None
    assert (same.promotion.rival, same.promotion.holder_version) == (ScorerName.HOLDER, "3")
    assert ScorerName.HOLDER in {entry.scorer for entry in same.metrics}
    assert same.promotion.delta == 0.0, "the holder is the same model"
    assert (same.promotion.would_promote, same.promotion.promoted) == (False, False)
    assert same.model_version == "4"
    assert run_of(tenant, "4").data.tags["promoted"] == "false"
    assert aliases(tenant) == {ALIAS: "3"}, "an equal model took the alias"

    # Trained on inverted labels, so it is worse on every platform: a weak but honest model can
    # tie or beat the holder on a planted tenant, depending on LightGBM's float arithmetic.
    def inverted(
        train_frame: pandas.DataFrame,
        valid_frame: pandas.DataFrame,
        columns: Sequence[str],
        params: RankerParams,
    ) -> Trained:
        return train(
            train_frame.assign(**{LABEL_COLUMN: 1 - train_frame[LABEL_COLUMN]}),
            valid_frame.assign(**{LABEL_COLUMN: 1 - valid_frame[LABEL_COLUMN]}),
            columns,
            params,
        )

    monkeypatch.setattr(ranker, "train", inverted)
    worse = await trained()

    assert worse.promotion is not None
    assert worse.promotion.rival is ScorerName.HOLDER
    assert worse.promotion.delta < 0
    assert (worse.promotion.would_promote, worse.promotion.promoted) == (False, False)
    assert aliases(tenant) == {ALIAS: "3"}, "a worse model took the alias"


def graded_model(columns: tuple[str, ...]) -> Trained:
    frame = graded_frame()
    for column in columns:
        if column not in frame:
            frame[column] = np.random.default_rng(0).random(len(frame))
    valid = frame["source_url"].isin(split_sources(frame["source_url"], share=0.3, seed=7))
    return train(frame[~valid], frame[valid], columns, RankerParams(max_rounds=20))


def promote(tenant: str, model: Trained, monkeypatch: pytest.MonkeyPatch, **fields: object) -> str:
    """Register ``model`` for the tenant and give it the production alias; by default trained
    on the current feature code with ENOUGH's split."""
    report = make.report(
        tenant,
        columns=model.columns,
        best_iteration=model.best_iteration,
        **{"feature_set_version": code_digest(), "settings": ENOUGH, **fields},
    )
    _, version = log_ranker(report, model, summarise_ranker(report))
    assert version is not None
    monkeypatch.setenv(PROMOTION_ENV, "allowed")
    move_alias(tenant, version)
    monkeypatch.delenv(PROMOTION_ENV)
    return version


def ranked(path: Path) -> pandas.DataFrame:
    table = pq.read_table(path)
    assert table.schema.remove_metadata().equals(RANKED_SCHEMA)
    return table.to_pandas()


@pytest.mark.integration
async def test_rank_pairs_learned_and_baseline_fallback(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    local_mlflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = f"{tenant}-other"
    await seed_ranking(graph, mongo, tenant)
    cache = tmp_path / "cache"
    stored = (await graph_state(graph, tenant), await mongo_state(mongo))
    promote(other, graded_model(MODEL_COLUMNS), monkeypatch)

    baseline, path = await rank_pairs(graph, mongo, tenant, cache_dir=cache)

    assert (await graph_state(graph, tenant), await mongo_state(mongo)) == stored
    assert path == cache / tenant / "ranked_pairs.parquet"
    assert pq.read_schema(path).metadata[b"tenant_id"] == tenant.encode()
    assert (baseline.scorer, baseline.model_version) == (ScorerName.BASELINE, None)
    assert baseline.fallback_reason == "no promoted model", "another tenant's model was used"
    rows = ranked(path)
    assert len(rows) == baseline.pairs > 0
    assert set(rows["scorer"]) == {"baseline"}
    assert rows["model_version"].isna().all()
    _, matrix = await assemble_features(graph, mongo, tenant, cache_dir=cache)
    features = pq.read_table(matrix).to_pandas()
    expected = score_frame(features, default_weights())
    by_pair = dict(
        zip(
            zip(expected["source_url"], expected["target_url"], strict=True),
            expected["score"],
            strict=True,
        )
    )
    assert rows["score"].tolist() == pytest.approx(
        [by_pair[pair] for pair in zip(rows["source_url"], rows["target_url"], strict=True)]
    ), "the fallback scores are not the #17 baseline's"
    for _, group in rows.groupby("source_url"):
        assert group["rank_in_source"].tolist() == list(range(1, len(group) + 1))
        assert group["score"].is_monotonic_decreasing

    model = graded_model(MODEL_COLUMNS)
    version = promote(tenant, model, monkeypatch)

    learned, path = await rank_pairs(graph, mongo, tenant, cache_dir=cache)

    assert (learned.scorer, learned.model_version, learned.fallback_reason) == (
        ScorerName.LEARNED,
        version,
        None,
    )
    rows = ranked(path)
    assert set(rows["scorer"]) == {"learned"}
    assert set(rows["model_version"]) == {version}
    matrix_rows = features.set_index(list(KEY_COLUMNS)).loc[
        list(zip(rows["source_url"], rows["target_url"], strict=True))
    ]
    expected_scores = model.booster.predict(
        matrix_rows.loc[:, list(MODEL_COLUMNS)].to_numpy(dtype=np.float64)
    )
    assert rows["score"].to_numpy() == pytest.approx(expected_scores)
    for _, group in rows.groupby("source_url"):
        assert group["rank_in_source"].tolist() == list(range(1, len(group) + 1))
        assert group["score"].is_monotonic_decreasing
    assert (cache / tenant / "ranker" / f"model-{version}.txt").is_file()

    retired = graded_model((*make.COLUMNS, "retired_feature"))
    promote(tenant, retired, monkeypatch)
    missing, _ = await rank_pairs(graph, mongo, tenant, cache_dir=cache)
    assert missing.scorer is ScorerName.BASELINE
    assert missing.fallback_reason == "feature matrix lacks model columns: retired_feature"

    promote(tenant, graded_model(MODEL_COLUMNS), monkeypatch, feature_set_version=make.DIGEST)
    older, _ = await rank_pairs(graph, mongo, tenant, cache_dir=cache)
    assert older.scorer is ScorerName.BASELINE
    assert older.fallback_reason == (
        "model trained on another feature set than the current feature code"
    )

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    with capture_logs() as logs:
        unreachable, path = await rank_pairs(graph, mongo, tenant, cache_dir=cache)
    assert unreachable.scorer is ScorerName.BASELINE
    assert unreachable.fallback_reason is not None
    assert re.fullmatch(r"model registry unreachable \(\w+\)", unreachable.fallback_reason)
    assert "127.0.0.1" not in unreachable.fallback_reason
    assert set(ranked(path)["scorer"]) == {"baseline"}
    [line] = [entry for entry in logs if entry["event"] == "ranker.ranked"]
    assert line["fallback_reason"] == unreachable.fallback_reason
    text = " ".join(str(value) for entry in logs for value in entry.values())
    assert [u for u in URLS if u in text] == [], "page urls in the log"


@pytest.mark.integration
async def test_a_holder_that_saw_other_test_sources_or_features_is_no_rival(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    tmp_path: Path,
    local_mlflow: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A holder trained on another split has seen some of today's test source pages, and one of
    another feature set or with a missing column cannot score today's rows: the baseline is the
    rival instead, with the reason logged."""
    await seed_ranking(graph, mongo, tenant)
    model = graded_model(MODEL_COLUMNS)
    holders = {
        "split_differs": (model, {"settings": HeldOutSettings(rounds=5, test_share=0.3)}),
        "feature_set_differs": (model, {"feature_set_version": make.DIGEST}),
        "columns_missing": (graded_model((*make.COLUMNS, "retired_feature")), {}),
    }

    for reason, (found, fields) in holders.items():
        version = promote(tenant, found, monkeypatch, **fields)
        with capture_logs() as logs:
            report = await train_ranker(
                graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage()), settings=ENOUGH
            )

        assert report.promotion is not None
        assert (report.promotion.rival, report.promotion.holder_version) == (
            ScorerName.BASELINE,
            None,
        ), reason
        assert ScorerName.HOLDER not in {entry.scorer for entry in report.metrics}, reason
        [line] = [entry for entry in logs if entry["event"] == "ranker.holder_refused"]
        assert (line["holder_version"], line["reason"]) == (version, reason)
        assert not report.promotion.promoted, "a local run promoted its model"
        assert aliases(tenant) == {ALIAS: version}


@pytest.mark.integration
async def test_select_anchors_outputs_unchanged(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    """The #23 stage writes what it wrote before the held-out view existed, and a view that
    hides nothing over the production candidates chooses exactly the same anchors."""
    await seed_selection(graph, mongo, tenant)

    report, path = await select_anchors(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(selection_voyage())
    )
    view = AnchorView(await retrieve_candidates(graph, tenant), frozenset())
    selection = await compute_anchor_choices(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(selection_voyage()), view=view
    )

    written = pq.read_table(path).to_pandas()
    columns = ["source_url", "target_url", "rank", "phrase", "anchor_type", "score_total"]
    found = pandas.DataFrame(
        [
            (
                c.match.source_url,
                c.match.target_url,
                c.rank,
                c.match.phrase,
                c.anchor_type.value,
                c.score.total,
            )
            for c in selection.choices
        ],
        columns=columns,
    )
    key = ["source_url", "target_url", "rank"]
    left = written[columns].sort_values(key, ignore_index=True)
    right = found.sort_values(key, ignore_index=True)
    pandas.testing.assert_frame_equal(left, right, check_dtype=False)
    assert selection.report.chosen == report.chosen
    assert selection.report.unanchored == report.unanchored


async def test_rank_pairs_refuses_a_bad_tenant_and_unknown_weights_before_the_graph_is_read(
    tmp_path: Path,
) -> None:
    unused = object()
    with pytest.raises(ValueError, match="tenant_id"):
        await rank_pairs(
            cast("GraphRepo", unused), cast("MongoRepo", unused), "../escape", cache_dir=tmp_path
        )
    with pytest.raises(ValueError, match="no_such_column"):
        await rank_pairs(
            cast("GraphRepo", unused),
            cast("MongoRepo", WeightsOnly()),
            "test-weights",
            cache_dir=tmp_path,
        )
    assert files(tmp_path) == set()
