from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING

import pytest
from mlflow import MlflowClient
from mlflow.artifacts import load_dict, load_text
from test_embed import seed_plain, url

from linking_engine.graph.repo import GraphRepo
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.ml.tracking import (
    EXPERIMENT_KIND,
    EXPERIMENT_KIND_TAG,
    analytics_experiment,
    log_duplicates,
    log_pipeline,
)
from linking_engine.models import (
    DuplicateReport,
    Link,
    PipelineReport,
    StageResult,
    StageStatus,
)
from linking_engine.pipeline import flows
from linking_engine.pipeline.ranker import RANKED_PAIRS_FILE
from linking_engine.pipeline.recommendations import REQUIRED_FILES
from linking_engine.pipeline.tenant_pipeline import (
    NEEDS,
    REPORT_STAGES,
    PipelineFailedError,
    StaleInputError,
    StoredOutputs,
    peak_rss_mb,
    plan_stages,
    run_pipeline,
    summarise_pipeline,
)

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.pipeline.tenant_pipeline import Runner

TENANT = "acme"
OK = StageStatus.OK
CORE = [stage for stage in NEEDS if stage not in REPORT_STAGES and stage != "train-ranker"]
WRITES = {
    writer: [name for name, w in REQUIRED_FILES if w == writer] for _, writer in REQUIRED_FILES
}


class Tenant:
    """Fake stage runners and stored outputs of one tenant. Each stage sleeps, then fails or
    writes the files recommendations reads; link-audit completes an audit."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        seconds: dict[str, float] | None = None,
        failing: frozenset[str] = frozenset(),
        silent: frozenset[str] = frozenset(),
        stored: frozenset[str] = frozenset(NEEDS),
        audit_at: datetime | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.folder = cache_dir / TENANT
        self.folder.mkdir(parents=True, exist_ok=True)
        self.seconds = seconds or {}
        self.failing = failing
        # Stages that finish without writing their outputs.
        self.silent = silent
        self.stored = stored
        self.audit_at = audit_at
        self.events: list[tuple[str, str]] = []
        self.begun = {stage: asyncio.Event() for stage in NEEDS}
        self.probed: list[str] = []
        self.records: list[PipelineReport] = []

    def runners(self, *, source: bool = True) -> dict[str, Runner]:
        return {
            stage: partial(self.run, stage)
            for stage in NEEDS
            if source or stage != "prepare-corpus"
        }

    async def run(self, stage: str) -> tuple[str, object]:
        self.events.append(("start", stage))
        self.begun[stage].set()
        await asyncio.sleep(self.seconds.get(stage, 0.01))
        self.events.append(("end", stage))
        if stage in self.failing:
            raise RuntimeError(f"{stage} broke")
        if stage not in self.silent:
            for name in WRITES.get(stage, ()):
                (self.folder / name).write_bytes(b"")
            if stage == "link-audit":
                self.audit_at = datetime.now(UTC)
        if stage == "rank-pairs":
            return stage, self.folder / RANKED_PAIRS_FILE
        return stage, f"mlflow-{stage}"

    async def present(self, stage: str) -> bool:
        self.probed.append(stage)
        return stage in self.stored

    async def audit_completed(self) -> datetime | None:
        return self.audit_at

    def record(self, report: PipelineReport) -> str:
        self.records.append(report)
        return "mlflow-pipeline"

    def started(self) -> list[str]:
        return [stage for kind, stage in self.events if kind == "start"]

    def at(self, kind: str, stage: str) -> int:
        return self.events.index((kind, stage))


async def run(
    tenant: Tenant,
    *,
    source: bool = True,
    retrain: bool = False,
    reports: bool = False,
    from_stage: str | None = None,
    pipeline_run_id: str | None = None,
) -> PipelineReport:
    return await run_pipeline(
        TENANT,
        tenant.runners(source=source),
        tenant,
        tenant.record,
        cache_dir=tenant.cache_dir,
        retrain=retrain,
        reports=reports,
        from_stage=from_stage,
        pipeline_run_id=pipeline_run_id,
    )


def outdated(tenant: Tenant, stage: str) -> None:
    """The stage's files as an earlier run left them."""
    old = time.time() - 3600
    for name in WRITES[stage]:
        path = tenant.folder / name
        path.write_bytes(b"")
        os.utime(path, (old, old))


async def test_every_stage_starts_only_after_all_it_needs_have_finished(tmp_path: Path) -> None:
    tenant = Tenant(tmp_path, seconds={"embed-links": 0.05, "quality-eval": 0.05})

    report = await run(tenant, retrain=True, reports=True)

    assert [(r.stage, r.status) for r in report.stages] == [(stage, OK) for stage in NEEDS]
    assert sorted(tenant.started()) == sorted(NEEDS)
    for stage, needs in NEEDS.items():
        for need in needs:
            assert tenant.at("end", need) < tenant.at("start", stage), f"{stage} before {need}"


async def test_independent_stages_run_side_by_side(tmp_path: Path) -> None:
    tenant = Tenant(tmp_path, seconds={"embed-links": 0.2})

    await run(tenant)

    third = ("embed-pages", "embed-links", "resolve-keywords")
    assert max(tenant.at("start", s) for s in third) < min(tenant.at("end", s) for s in third)
    # graph-analytics needs no link vectors and hub-bridges no link scores.
    links_embedded = tenant.at("end", "embed-links")
    assert tenant.at("start", "graph-analytics") < links_embedded
    assert tenant.at("start", "hub-bridges") < links_embedded
    assert tenant.at("start", "score-links") > links_embedded


async def test_a_failure_starts_nothing_new_lets_running_stages_finish_and_raises_after_the_record(
    tmp_path: Path,
) -> None:
    tenant = Tenant(
        tmp_path,
        seconds={"hub-bridges": 0.05, "link-audit": 0.15},
        failing=frozenset({"quality-eval"}),
    )

    with pytest.raises(PipelineFailedError) as raised:
        await run(tenant, retrain=True)

    assert raised.value.failed == ("quality-eval",)
    assert str(raised.value.__cause__) == "quality-eval broke"
    [report] = tenant.records
    statuses = {r.stage: r.status for r in report.stages}
    assert statuses == {
        **dict.fromkeys(CORE[:7], OK),
        "hub-bridges": OK,
        "quality-eval": StageStatus.FAILED,
        "anchor-selection": StageStatus.NOT_RUN,
        "train-ranker": StageStatus.SKIPPED,
        "rank-pairs": StageStatus.SKIPPED,
        "link-audit": OK,
        "recommendations": StageStatus.SKIPPED,
    }
    assert "anchor-selection" not in tenant.started()
    by_stage = {r.stage: r for r in report.stages}
    assert by_stage["quality-eval"].error == "RuntimeError"
    assert by_stage["link-audit"].mlflow_run_id == "mlflow-link-audit"


async def test_from_stage_runs_that_stage_and_those_after_it_once_the_outputs_before_it_are_found(
    tmp_path: Path,
) -> None:
    tenant = Tenant(tmp_path, audit_at=datetime.now(UTC) - timedelta(hours=1))
    outdated(tenant, "hub-bridges")

    report = await run(tenant, source=False, from_stage="anchor-selection")

    ran = ["anchor-selection", "rank-pairs", "recommendations"]
    assert [(r.stage, r.status) for r in report.stages] == [(stage, OK) for stage in ran]
    assert tenant.started() == ran
    assert tenant.probed == [
        "prepare-corpus",
        "load-graph",
        "embed-pages",
        "embed-links",
        "resolve-keywords",
        "graph-analytics",
        "score-links",
        "hub-bridges",
        "quality-eval",
        "link-audit",
    ]
    assert report.from_stage == "anchor-selection"


async def test_a_missing_upstream_output_names_the_stage_to_run_first_and_runs_nothing(
    tmp_path: Path,
) -> None:
    tenant = Tenant(tmp_path, stored=frozenset(NEEDS) - {"hub-bridges", "embed-links"})

    with pytest.raises(ValueError, match=r"of embed-links, hub-bridges .*; run from embed-links"):
        await run(tenant, source=False, from_stage="rank-pairs")

    assert (tenant.events, tenant.records) == ([], [])


@pytest.mark.parametrize(
    ("stage", "options", "message"),
    [
        ("embed", {}, "unknown stage 'embed'; valid stages: prepare-corpus, load-graph, "),
        ("train-ranker", {"reports": True}, "train-ranker runs only with retrain"),
        ("score-pairs", {"retrain": True}, "score-pairs runs only with reports"),
    ],
)
def test_an_unknown_or_unplanned_stage_is_refused(
    stage: str, options: dict[str, bool], message: str
) -> None:
    flags = {"retrain": False, "reports": False, **options}
    with pytest.raises(ValueError, match=message):
        plan_stages(from_stage=stage, with_source=True, **flags)


async def test_retrain_runs_train_ranker_after_anchor_selection_and_quality_eval_before_rank_pairs(
    tmp_path: Path,
) -> None:
    slow_quality = {"quality-eval": 0.2, "train-ranker": 0.05}
    retrained = Tenant(tmp_path / "retrain", seconds=slow_quality)
    plain = Tenant(tmp_path / "plain", seconds=slow_quality)

    await run(retrained, retrain=True)
    await run(plain)

    assert retrained.at("end", "anchor-selection") < retrained.at("start", "train-ranker")
    assert retrained.at("end", "quality-eval") < retrained.at("start", "train-ranker")
    assert retrained.at("end", "train-ranker") < retrained.at("start", "rank-pairs")
    assert "train-ranker" not in plain.started()
    assert plain.at("start", "rank-pairs") < plain.at("end", "quality-eval")


async def test_reports_run_only_when_asked_each_after_what_it_reads(tmp_path: Path) -> None:
    without = Tenant(tmp_path / "without")
    reported = Tenant(tmp_path / "with", seconds={"rank-pairs": 0.05})

    await run(without)
    await run(reported, reports=True)

    assert sorted(without.started()) == sorted(CORE)
    assert sorted(reported.started()) == sorted([*CORE, *REPORT_STAGES])
    for stage, need in (
        ("candidate-retrieval", "graph-analytics"),
        ("anchor-extraction", "hub-bridges"),
        ("feature-assembly", "rank-pairs"),
        ("score-pairs", "rank-pairs"),
    ):
        assert reported.at("end", need) < reported.at("start", stage)


async def test_recommendations_refuse_a_file_this_run_did_not_write(tmp_path: Path) -> None:
    tenant = Tenant(tmp_path, silent=frozenset({"anchor-selection"}))
    outdated(tenant, "anchor-selection")
    (tenant.folder / "unanchored_pairs.parquet").unlink()

    with pytest.raises(PipelineFailedError) as raised:
        await run(tenant)

    stale = raised.value.__cause__
    assert isinstance(stale, StaleInputError)
    assert str(stale).endswith("anchor_choices.parquet, unanchored_pairs.parquet")
    assert "recommendations" not in tenant.started()
    [report] = tenant.records
    assert report.stages[-1] == StageResult(
        stage="recommendations",
        status=StageStatus.FAILED,
        seconds=report.stages[-1].seconds,
        peak_mb=report.stages[-1].peak_mb,
        error="StaleInputError",
    )


async def test_recommendations_refuse_an_audit_older_than_the_run(tmp_path: Path) -> None:
    tenant = Tenant(
        tmp_path,
        silent=frozenset({"link-audit"}),
        audit_at=datetime.now(UTC) - timedelta(minutes=5),
    )

    with pytest.raises(PipelineFailedError) as raised:
        await run(tenant)

    assert isinstance(raised.value.__cause__, StaleInputError)
    assert str(raised.value.__cause__).endswith(": the latest link audit")
    assert "recommendations" not in tenant.started()


async def test_prepare_corpus_needs_a_crawl_source(tmp_path: Path) -> None:
    tenant = Tenant(tmp_path)

    with pytest.raises(ValueError, match="prepare-corpus needs source_db and source_collection"):
        await run(tenant, source=False)
    assert tenant.events == []

    report = await run(tenant, source=False, from_stage="load-graph")

    assert report.stages[0].stage == "load-graph"
    assert tenant.probed == ["prepare-corpus"]


def test_the_flow_runs_the_stage_flow_of_each_name_and_prepare_corpus_only_with_a_source(
    tmp_path: Path,
) -> None:
    runners = flows.stage_runners(TENANT, "crawls", "acme_pages", tmp_path)
    without = flows.stage_runners(TENANT, "crawls", None, tmp_path)

    assert {stage: runner.func.name for stage, runner in runners.items()} == {  # type: ignore[attr-defined]
        stage: stage for stage in NEEDS
    }
    assert set(without) == set(NEEDS) - {"prepare-corpus"}


async def test_the_run_record_holds_every_stages_status_time_memory_and_mlflow_run(
    tmp_path: Path,
) -> None:
    tenant = Tenant(tmp_path)

    report = await run(tenant, pipeline_run_id="flow-run-1")

    assert tenant.records == [report]
    assert (report.tenant_id, report.pipeline_run_id, report.retrain, report.reports) == (
        TENANT,
        "flow-run-1",
        False,
        False,
    )
    assert report.seconds >= sum(r.seconds or 0 for r in report.stages[:2])
    runs = {r.stage: r.mlflow_run_id for r in report.stages}
    assert runs == {stage: None if stage == "rank-pairs" else f"mlflow-{stage}" for stage in CORE}
    assert all(r.seconds and r.seconds >= 0.01 and r.peak_mb for r in report.stages)


async def test_a_record_that_cannot_be_logged_never_hides_a_stage_failure(
    tmp_path: Path,
) -> None:
    def unreachable(report: PipelineReport) -> str:
        raise ConnectionError("mlflow is down")

    failing = Tenant(tmp_path / "failing", failing=frozenset({"score-links"}))
    healthy = Tenant(tmp_path / "healthy")

    with pytest.raises(PipelineFailedError) as raised:
        await run_pipeline(
            TENANT, failing.runners(), failing, unreachable, cache_dir=failing.cache_dir
        )
    with pytest.raises(ConnectionError, match="mlflow is down"):
        await run_pipeline(
            TENANT, healthy.runners(), healthy, unreachable, cache_dir=healthy.cache_dir
        )

    assert raised.value.failed == ("score-links",)
    assert str(raised.value.__cause__) == "score-links broke"


async def test_cancelling_the_pipeline_cancels_the_stages_still_running(tmp_path: Path) -> None:
    tenant = Tenant(tmp_path, seconds={"embed-links": 5})
    pipeline = asyncio.create_task(run(tenant))
    await asyncio.wait_for(tenant.begun["embed-links"].wait(), timeout=5)

    pipeline.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pipeline

    assert ("end", "embed-links") not in tenant.events
    assert tenant.records == []


@pytest.mark.parametrize(
    "fields",
    [
        {"status": OK},
        {"status": StageStatus.SKIPPED, "seconds": 1.0, "peak_mb": 1.0},
        {"status": StageStatus.FAILED, "seconds": 1.0, "peak_mb": 1.0},
        {"status": OK, "seconds": 1.0, "peak_mb": 1.0, "error": "RuntimeError"},
        {
            "status": StageStatus.FAILED,
            "seconds": 1.0,
            "peak_mb": 1.0,
            "error": "RuntimeError",
            "mlflow_run_id": "run-1",
        },
    ],
)
def test_a_stage_result_is_consistent_with_its_status(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"a stage has|only a stage that finished"):
        StageResult.model_validate({"stage": "load-graph", **fields})


def test_peak_memory_is_in_megabytes_on_this_platform() -> None:
    assert 10 < peak_rss_mb() < 64 * 1024


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def pipeline_report() -> PipelineReport:
    return PipelineReport(
        tenant_id=TENANT,
        pipeline_run_id="flow-run-1",
        retrain=True,
        reports=False,
        from_stage="rank-pairs",
        started_at=datetime(2026, 9, 29, 12, tzinfo=UTC),
        seconds=12.5,
        stages=(
            StageResult(stage="rank-pairs", status=OK, seconds=4.0, peak_mb=512.0),
            StageResult(
                stage="recommendations",
                status=StageStatus.FAILED,
                seconds=8.0,
                peak_mb=768.0,
                error="StaleInputError",
            ),
            StageResult(stage="score-pairs", status=StageStatus.SKIPPED),
        ),
    )


def test_the_pipeline_run_is_logged_with_its_stages_and_no_urls(local_mlflow: str) -> None:
    report = pipeline_report()
    summary = summarise_pipeline(report)

    run_id = log_pipeline(report, summary)

    client = MlflowClient(local_mlflow)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    assert (experiment.name, experiment.tags[EXPERIMENT_KIND_TAG]) == (
        analytics_experiment(TENANT),
        EXPERIMENT_KIND,
    )
    assert {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")} == {
        "tenant_id": TENANT,
        "kind": "pipeline",
        "stage": "tenant-pipeline",
        "status": "failed",
        "pipeline_run_id": "flow-run-1",
    }
    assert run.data.tags["mlflow.note.content"] == summary
    assert run.data.params == {
        "tenant": TENANT,
        "retrain": "True",
        "from_stage": "rank-pairs",
        "reports": "False",
    }
    assert run.data.metrics == {
        "seconds": 12.5,
        "rank_pairs_seconds": 4.0,
        "rank_pairs_peak_mb": 512.0,
        "recommendations_seconds": 8.0,
        "recommendations_peak_mb": 768.0,
        "peak_mb": 768.0,
    }
    artifacts = f"runs:/{run_id}"
    assert load_dict(f"{artifacts}/stages.json") == {
        "rank-pairs": {
            "status": "ok",
            "seconds": 4.0,
            "peak_mb": 512.0,
            "mlflow_run_id": None,
            "error": None,
        },
        "recommendations": {
            "status": "failed",
            "seconds": 8.0,
            "peak_mb": 768.0,
            "mlflow_run_id": None,
            "error": "StaleInputError",
        },
        "score-pairs": {
            "status": "skipped",
            "seconds": None,
            "peak_mb": None,
            "mlflow_run_id": None,
            "error": None,
        },
    }
    assert load_dict(f"{artifacts}/metrics.json") == run.data.metrics
    assert load_text(f"{artifacts}/summary.md") == summary
    assert "Failed: recommendations (StaleInputError)." in summary
    assert "Skipped after a failure: score-pairs." in summary
    assert not any("http" in text or ".com" in text for text in (summary, *run.data.tags.values()))


def test_stage_runs_carry_the_pipeline_run_id_of_their_prefect_root_flow_run(
    local_mlflow: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = DuplicateReport(
        tenant_id=TENANT,
        groups=(),
        pages_in_groups=0,
        non_canonical=0,
        largest_group=0,
        seconds=0,
        finished_at=datetime(2026, 9, 29, 12, tzinfo=UTC),
    )
    client = MlflowClient(local_mlflow)

    outside = log_duplicates(report, "No duplicates.")
    monkeypatch.setenv("PREFECT__RUNTIME__FLOW_RUN__ROOT_FLOW_RUN_ID", "flow-run-1")
    inside = log_duplicates(report, "No duplicates.")

    assert "pipeline_run_id" not in client.get_run(outside).data.tags
    assert client.get_run(inside).data.tags["pipeline_run_id"] == "flow-run-1"


async def analyse(graph: GraphRepo, tenant: str) -> None:
    """A PageRank percentile and a scored body link, as graph-analytics and score-links leave
    them."""
    link = Link(
        source_url=url(0), target_url=url(1), position=0, anchor_text="x", surrounding_text=""
    )
    await graph.replace_links(tenant, [url(0)], [link])
    await graph._auto("MATCH (p:Page {tenantId: $t}) SET p.pageRankPercentile = 0.5", t=tenant)
    await graph._auto(
        "MATCH (:Page {tenantId: $t})-[r:LINKS_TO]->() SET r.contextRelevance = 0.5", t=tenant
    )


@pytest.mark.integration
async def test_stored_outputs_are_found_only_once_the_tenant_has_them(
    graph: GraphRepo,
    mongo: MongoRepo,
    neo4j_server: tuple[str, str, str],
    mongo_uri: str,
    tenant: str,
    tmp_path: Path,
    local_mlflow: str,
) -> None:
    outputs = StoredOutputs(
        tenant,
        tmp_path,
        partial(GraphRepo.connect, *neo4j_server),
        partial(MongoRepo.connect, mongo_uri, "linking_engine_test"),
    )
    upstream = [stage for stage in NEEDS if stage not in REPORT_STAGES | {"recommendations"}]

    empty = [stage for stage in upstream if await outputs.present(stage)]
    await seed_plain(mongo, graph, tenant, 3)
    (tmp_path / tenant).mkdir()
    for name, _ in REQUIRED_FILES:
        (tmp_path / tenant / name).write_bytes(b"")
    seeded = [stage for stage in upstream if await outputs.present(stage)]
    other = f"{tenant}-other"
    await seed_plain(mongo, graph, other, 3)
    await analyse(graph, other)
    beside_other = [stage for stage in upstream if await outputs.present(stage)]
    await analyse(graph, tenant)
    analysed = [stage for stage in upstream if await outputs.present(stage)]

    # Without a promoted model rank-pairs uses the baseline scorer, so none is required.
    assert empty == ["train-ranker"]
    loaded = ["prepare-corpus", "load-graph"]
    files = ["hub-bridges", "anchor-selection", "train-ranker", "rank-pairs"]
    assert seeded == beside_other == [*loaded, *files]
    assert analysed == [*loaded, "graph-analytics", "score-links", *files]
    assert await outputs.audit_completed() is None
