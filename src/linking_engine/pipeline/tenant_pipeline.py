"""The tenant pipeline: every stage flow of one tenant, each started once the stages it needs
have finished, independent stages side by side."""

from __future__ import annotations

import asyncio
import resource
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Protocol

import structlog
from structlog.contextvars import bound_contextvars

from linking_engine.graph.repo import GraphRepo
from linking_engine.ml.tracking import previous_quality_run
from linking_engine.models import PipelineReport, StageResult, StageStatus
from linking_engine.pipeline.anchors import cache_folder
from linking_engine.pipeline.recommendations import REQUIRED_FILES

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from linking_engine.ingest.mongo_repo import MongoRepo

log = structlog.get_logger(__name__)

type Runner = Callable[[], Awaitable[object]]

PIPELINE_STAGE: Final = "tenant-pipeline"
PREPARE_CORPUS: Final = "prepare-corpus"
QUALITY_EVAL: Final = "quality-eval"
TRAIN_RANKER: Final = "train-ranker"
LINK_AUDIT: Final = "link-audit"
RECOMMENDATIONS: Final = "recommendations"

# Every stage's own dependencies, each stage listed after all of them. A dependency the run does
# not plan is dropped: rank-pairs waits for train-ranker only when retraining.
NEEDS: Final[Mapping[str, tuple[str, ...]]] = {
    "prepare-corpus": (),
    "load-graph": ("prepare-corpus",),
    "embed-pages": ("load-graph",),
    "embed-links": ("load-graph",),
    "resolve-keywords": ("load-graph",),
    "graph-analytics": ("embed-pages", "resolve-keywords"),
    "score-links": ("embed-pages", "embed-links"),
    "hub-bridges": ("graph-analytics",),
    # Link scores are read through a check that rejects a generic anchor still carrying the fit
    # of an earlier run, and only score-links clears it.
    "quality-eval": ("graph-analytics", "score-links"),
    "candidate-retrieval": ("graph-analytics",),
    "anchor-selection": ("hub-bridges", "score-links"),
    "anchor-extraction": ("hub-bridges",),
    "train-ranker": ("anchor-selection", "quality-eval"),
    "rank-pairs": ("anchor-selection", "train-ranker"),
    "link-audit": ("graph-analytics", "score-links", "resolve-keywords"),
    # After rank-pairs, so they reuse its feature matrix.
    "feature-assembly": ("rank-pairs",),
    "score-pairs": ("rank-pairs",),
    "recommendations": (
        "rank-pairs",
        "anchor-selection",
        "hub-bridges",
        "link-audit",
        "quality-eval",
    ),
}
REPORT_STAGES: Final = frozenset(
    {"candidate-retrieval", "anchor-extraction", "feature-assembly", "score-pairs"}
)


class StaleInputError(ValueError):
    """An input of recommendations that a stage of this run writes is older than the run."""


class PipelineFailedError(RuntimeError):
    """One or more stages failed; raised after the run record was logged."""

    def __init__(self, tenant_id: str, failed: tuple[str, ...]) -> None:
        super().__init__(f"the pipeline of tenant {tenant_id!r} failed at {', '.join(failed)}")
        self.failed = failed


@dataclass(frozen=True, slots=True)
class Plan:
    """The stages a run starts, in dependency order, each with the stages of the run it waits
    for, and the stages before them whose stored outputs the run reads."""

    stages: tuple[str, ...]
    waits_for: Mapping[str, tuple[str, ...]]
    upstream: tuple[str, ...]


def plan_stages(*, retrain: bool, reports: bool, from_stage: str | None, with_source: bool) -> Plan:
    """The core stages, the reports with ``reports`` and train-ranker with ``retrain``; from
    ``from_stage`` only that stage and those after it. Raises ValueError for an unknown or
    unplanned stage, and for prepare-corpus without a crawl source."""
    planned = [
        stage
        for stage in NEEDS
        if (reports or stage not in REPORT_STAGES) and (retrain or stage != TRAIN_RANKER)
    ]
    needs = {stage: tuple(n for n in NEEDS[stage] if n in planned) for stage in planned}
    if from_stage is None:
        chosen = set(planned)
    elif from_stage not in NEEDS:
        raise ValueError(f"unknown stage {from_stage!r}; valid stages: {', '.join(NEEDS)}")
    elif from_stage not in needs:
        flag = "retrain" if from_stage == TRAIN_RANKER else "reports"
        raise ValueError(f"{from_stage} runs only with {flag}")
    else:
        chosen = {from_stage}
        for stage in planned:
            if chosen.intersection(needs[stage]):
                chosen.add(stage)
    if PREPARE_CORPUS in chosen and not with_source:
        raise ValueError("prepare-corpus needs source_db and source_collection")
    required: set[str] = set()
    for stage in reversed(planned):
        if stage in chosen or stage in required:
            required.update(needs[stage])
    stages = tuple(stage for stage in planned if stage in chosen)
    return Plan(
        stages=stages,
        waits_for={stage: tuple(n for n in needs[stage] if n in chosen) for stage in stages},
        upstream=tuple(stage for stage in planned if stage in required - chosen),
    )


class StageOutputs(Protocol):
    async def present(self, stage: str) -> bool:
        """Whether the stored outputs of ``stage`` exist for the tenant."""
        ...

    async def audit_completed(self) -> datetime | None:
        """When the tenant's latest completed link audit completed; None without one."""
        ...


async def _pages(graph: GraphRepo, tenant_id: str) -> bool:
    return (await graph.counts(tenant_id)).pages > 0


async def _page_vectors(graph: GraphRepo, tenant_id: str) -> bool:
    return bool(await graph.embedding_models(tenant_id))


async def _link_vectors(graph: GraphRepo, tenant_id: str) -> bool:
    return bool(await graph.surrounding_embedding_models(tenant_id))


async def _keywords(graph: GraphRepo, tenant_id: str) -> bool:
    return bool(await graph.keyword_targets(tenant_id))


_GRAPH_OUTPUTS: Final[Mapping[str, Callable[[GraphRepo, str], Awaitable[bool]]]] = {
    "load-graph": _pages,
    "embed-pages": _page_vectors,
    "embed-links": _link_vectors,
    "resolve-keywords": _keywords,
    "graph-analytics": GraphRepo.has_centrality,
    "score-links": GraphRepo.has_link_relevance,
}


class StoredOutputs:
    """The tenant's stage outputs in Neo4j, MongoDB, MLflow and its cache folder; each read
    opens its own connection."""

    def __init__(
        self,
        tenant_id: str,
        cache_dir: Path,
        graph: Callable[[], Awaitable[GraphRepo]],
        mongo: Callable[[], Awaitable[MongoRepo]],
    ) -> None:
        self._tenant_id = tenant_id
        self._folder = cache_folder(cache_dir, tenant_id)
        self._graph = graph
        self._mongo = mongo

    async def present(self, stage: str) -> bool:
        files = [name for name, writer in REQUIRED_FILES if writer == stage]
        if files:
            return all((self._folder / name).is_file() for name in files)
        if stage == TRAIN_RANKER:
            # Without a promoted model rank-pairs falls back to the baseline scorer.
            return True
        if stage == QUALITY_EVAL:
            return await asyncio.to_thread(previous_quality_run, self._tenant_id) is not None
        if stage == LINK_AUDIT:
            return await self.audit_completed() is not None
        if stage == PREPARE_CORPUS:
            async with await self._mongo() as mongo:
                pages = mongo.iter_page_summaries(self._tenant_id, batch_size=1)
                return bool(await anext(pages, []))
        probe = _GRAPH_OUTPUTS.get(stage)
        if probe is None:
            raise ValueError(f"{stage} stores no output a later stage reads")
        async with await self._graph() as graph:
            return await probe(graph, self._tenant_id)

    async def audit_completed(self) -> datetime | None:
        async with await self._mongo() as mongo:
            marker = await mongo.latest_link_audit_run(self._tenant_id)
        return None if marker is None else marker[1]


def peak_rss_mb() -> float:
    """The process's peak resident memory so far in MiB; ru_maxrss counts bytes on macOS and
    KiB on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _written_since(path: Path, moment: datetime) -> bool:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC) >= moment
    except FileNotFoundError:
        return False


async def check_fresh(
    plan: Plan, folder: Path, outputs: StageOutputs, started_at: datetime
) -> None:
    """Raise StaleInputError unless every recommendations input that a stage of ``plan``
    writes, and the latest link audit when the plan audits, is dated at or after
    ``started_at``."""
    stale = [
        name
        for name, writer in REQUIRED_FILES
        if writer in plan.stages and not _written_since(folder / name, started_at)
    ]
    if LINK_AUDIT in plan.stages:
        completed = await outputs.audit_completed()
        if completed is None or completed < started_at:
            stale.append("the latest link audit")
    if stale:
        raise StaleInputError(
            f"recommendations inputs not written by this pipeline run: {', '.join(stale)}"
        )


def _mlflow_run(result: object) -> str | None:
    if isinstance(result, tuple) and result and isinstance(result[-1], str):
        return result[-1]
    return None


async def _run_stage(stage: str, runner: Runner) -> tuple[StageResult, Exception | None]:
    started = time.perf_counter()
    try:
        with bound_contextvars(stage=stage):
            result = await runner()
    # Recorded with the stage and raised once the run record is logged.
    except Exception as error:  # noqa: BLE001
        seconds = time.perf_counter() - started
        failed = StageResult(
            stage=stage,
            status=StageStatus.FAILED,
            seconds=seconds,
            peak_mb=peak_rss_mb(),
            error=type(error).__name__,
        )
        return failed, error
    seconds = time.perf_counter() - started
    finished = StageResult(
        stage=stage,
        status=StageStatus.OK,
        seconds=seconds,
        peak_mb=peak_rss_mb(),
        mlflow_run_id=_mlflow_run(result),
    )
    return finished, None


async def _schedule(
    plan: Plan, runners: Mapping[str, Runner]
) -> tuple[tuple[StageResult, ...], dict[str, Exception]]:
    """Start every stage whose dependencies succeeded; after a failure start none and let the
    running ones finish."""
    done: dict[str, StageResult] = {}
    errors: dict[str, Exception] = {}
    waiting = list(plan.stages)
    running: dict[asyncio.Task[tuple[StageResult, Exception | None]], str] = {}
    try:
        while True:
            if not errors:
                for stage in [s for s in waiting if all(n in done for n in plan.waits_for[s])]:
                    waiting.remove(stage)
                    log.info("pipeline.stage.start", stage=stage)
                    running[asyncio.create_task(_run_stage(stage, runners[stage]))] = stage
            if not running:
                break
            finished, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                stage = running.pop(task)
                result, error = task.result()
                done[stage] = result
                if error is not None:
                    errors[stage] = error
                log.info(
                    "pipeline.stage.done",
                    stage=stage,
                    status=result.status.value,
                    seconds=result.seconds,
                    peak_mb=result.peak_mb,
                    error=result.error,
                )
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
    blocked = {StageStatus.FAILED, StageStatus.SKIPPED}
    for stage in waiting:
        skipped = any(done[n].status in blocked for n in plan.waits_for[stage])
        done[stage] = StageResult(
            stage=stage, status=StageStatus.SKIPPED if skipped else StageStatus.NOT_RUN
        )
    return tuple(done[stage] for stage in plan.stages), errors


async def run_pipeline(
    tenant_id: str,
    runners: Mapping[str, Runner],
    outputs: StageOutputs,
    record: Callable[[PipelineReport], str],
    *,
    cache_dir: Path,
    retrain: bool = False,
    from_stage: str | None = None,
    reports: bool = False,
    pipeline_run_id: str | None = None,
) -> PipelineReport:
    """Run the planned stages of the tenant, then log the run record with ``record``. A plan
    that cannot run, or a stored output missing before ``from_stage``, raises ValueError before
    anything runs; a failed stage raises PipelineFailedError after the record."""
    plan = plan_stages(
        retrain=retrain,
        reports=reports,
        from_stage=from_stage,
        with_source=PREPARE_CORPUS in runners,
    )
    unknown = [stage for stage in plan.stages if stage not in runners]
    if unknown:
        raise ValueError(f"no runner for {', '.join(unknown)}")
    folder = cache_folder(cache_dir, tenant_id)
    with bound_contextvars(tenant_id=tenant_id, stage=PIPELINE_STAGE):
        missing = [stage for stage in plan.upstream if not await outputs.present(stage)]
        if missing:
            raise ValueError(
                f"no stored output of {', '.join(missing)} for tenant {tenant_id!r}; "
                f"run from {missing[0]} first"
            )
        started_at = datetime.now(UTC)
        started = time.perf_counter()

        async def recommendations() -> object:
            await check_fresh(plan, folder, outputs, started_at)
            return await runners[RECOMMENDATIONS]()

        results, errors = await _schedule(plan, {**runners, RECOMMENDATIONS: recommendations})
        report = PipelineReport(
            tenant_id=tenant_id,
            pipeline_run_id=pipeline_run_id,
            retrain=retrain,
            reports=reports,
            from_stage=from_stage,
            started_at=started_at,
            seconds=time.perf_counter() - started,
            stages=results,
        )
        failure = next((errors[stage] for stage in report.failed), None)
        try:
            mlflow_run = record(report)
        except Exception as error:
            if failure is None:
                raise
            log.error("pipeline.record_failed", error=type(error).__name__)
            raise PipelineFailedError(tenant_id, report.failed) from failure
        log.info(
            "pipeline.complete",
            seconds=report.seconds,
            failed=list(report.failed),
            mlflow_run=mlflow_run,
        )
    if failure is not None:
        raise PipelineFailedError(tenant_id, report.failed) from failure
    return report


def summarise_pipeline(report: PipelineReport) -> str:
    """A short prose record of one pipeline run, for the MLflow run description; no urls."""
    started = [r for r in report.stages if r.seconds is not None]
    peaks = [r.peak_mb for r in started if r.peak_mb is not None]
    finished = sum(r.status is StageStatus.OK for r in report.stages)
    run = f" {report.pipeline_run_id}" if report.pipeline_run_id else ""
    origin = f" from {report.from_stage}" if report.from_stage else ""
    lines = [
        f"Tenant pipeline{run} of tenant {report.tenant_id}{origin}: {finished} of "
        f"{len(report.stages)} stages finished in {report.seconds:.1f}s"
        + (f", peak memory {max(peaks):.0f} MB." if peaks else "."),
        f"Retrain {'on' if report.retrain else 'off'}, reports "
        f"{'on' if report.reports else 'off'}.",
    ]
    for status, label in (
        (StageStatus.FAILED, "Failed"),
        (StageStatus.SKIPPED, "Skipped after a failure"),
        (StageStatus.NOT_RUN, "Not started after a failure"),
    ):
        names = [
            f"{r.stage} ({r.error})" if r.error else r.stage
            for r in report.stages
            if r.status is status
        ]
        if names:
            lines.append(f"{label}: {', '.join(names)}.")
    if started:
        lines.append(
            "Stage times: " + ", ".join(f"{r.stage} {r.seconds:.1f}s" for r in started) + "."
        )
    return "\n".join(lines)
