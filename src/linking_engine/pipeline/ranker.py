"""Train a tenant's learned ranker on its held-out rounds and gate its promotion, then rank every
candidate pair with the production model, else with the baseline scorer. Read-only against both
stores; the training run, the model and its registry alias live in MLflow, the ranked pairs and
the local model copy under the tenant's cache folder."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from linking_engine.discovery.features import FEATURE_COLUMNS, KEY_COLUMNS, code_digest
from linking_engine.discovery.scoring import default_weights, score_frame
from linking_engine.ml.ranker_tracking import (
    RegistryUnavailableError,
    holder,
    load_production,
    log_ranker,
    move_alias,
    promotion_allowed,
    save_local,
    tag_promoted,
    trained_holder,
)
from linking_engine.ml.ranking import (
    EXCL_PLACEMENT_COLUMNS,
    EXCLUDED_COLUMNS,
    GROUP_COLUMNS,
    LIKE_FOR_LIKE_COLUMNS,
    MIN_TEST_GROUPS,
    MIN_TRAIN_GROUPS,
    MODEL_COLUMNS,
    NDCG_K,
    PREDICT_CHUNK,
    dominant_feature,
    group_ndcg,
    group_sources,
    importance,
    positive_groups,
    predict,
    promotion,
    ranking_metrics,
    split_sources,
    summarise_ranker,
    train,
)
from linking_engine.models import (
    HeldOutSettings,
    RankerParams,
    RankerReport,
    RankReport,
    ScorerName,
)
from linking_engine.pipeline.anchors import cache_folder, write_atomically
from linking_engine.pipeline.features import assemble_features
from linking_engine.pipeline.quality import git_sha
from linking_engine.pipeline.ranking_data import BASELINE_COLUMN, held_out_rounds, load_rounds

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from pathlib import Path

    import lightgbm
    import numpy.typing as npt

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.ml.ranker_tracking import Holder
    from linking_engine.ml.ranking import Trained
    from linking_engine.models import PromotionDecision, RankingMetrics, ScorerWeights
    from linking_engine.pipeline.ranking_data import HeldOutRounds

log = structlog.get_logger(__name__)

STAGE: Final = "ranker"
RANKED_PAIRS_FILE: Final = "ranked_pairs.parquet"
RANKED_SCHEMA: Final = pa.schema(
    [
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("target_url", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("rank_in_source", pa.int32(), nullable=False),
        pa.field("scorer", pa.string(), nullable=False),
        pa.field("model_version", pa.string()),
    ]
)
NO_MODEL: Final = "no promoted model"


async def train_ranker(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    cache_dir: Path,
    voyage: VoyageClient | None,
    settings: HeldOutSettings | None = None,
    params: RankerParams | None = None,
) -> RankerReport:
    """The tenant's held-out rounds, split by source page, trained with and without the
    link-derived columns and evaluated on the test source pages against the baseline scorer
    and the production model. The run is logged to MLflow, the model registered; the production
    alias moves to it only when its NDCG@10 beats the holder's, else the baseline's, over the
    whole interval and promotion is allowed here. Too little signal skips training with the
    reason and still logs the run."""
    started = time.perf_counter()
    settings = settings or HeldOutSettings()
    params = params or RankerParams()
    rounds = await held_out_rounds(
        graph, mongo, tenant_id, settings=settings, cache_dir=cache_dir, voyage=voyage
    )
    rival = await asyncio.to_thread(holder, tenant_id)
    sha = await asyncio.to_thread(git_sha)
    report, trained = await asyncio.to_thread(
        _train, tenant_id, rounds, settings, params, rival, promotion_allowed(), sha, started
    )
    run_id, version = await asyncio.to_thread(log_ranker, report, trained, summarise_ranker(report))
    report = RankerReport.model_validate({**report.model_dump(), "model_version": version})
    if report.promotion is not None and report.promotion.promoted:
        if trained is None or version is None:
            raise RuntimeError(f"the promoted ranker of {tenant_id!r} was not registered")
        await asyncio.to_thread(move_alias, tenant_id, version)
        # The run says promoted only once the alias has moved.
        await asyncio.to_thread(tag_promoted, run_id, version)
        await asyncio.to_thread(
            save_local, tenant_id, cache_dir, trained_holder(report, trained, run_id, version)
        )
    log.info(
        "ranker.trained",
        stage=STAGE,
        tenant_id=tenant_id,
        rounds=len(report.rounds),
        rounds_cached=sum(rounds.cache_hit),
        positives=report.positives,
        train_groups=report.train_groups,
        valid_groups=report.valid_groups,
        test_groups=report.test_groups,
        best_iteration=report.best_iteration,
        skipped=report.skipped_reason is not None,
        would_promote=report.promotion is not None and report.promotion.would_promote,
        promoted=report.promotion is not None and report.promotion.promoted,
        seconds=report.seconds,
    )
    return report


def _groups(frame: pandas.DataFrame) -> int:
    return len(frame.drop_duplicates(list(GROUP_COLUMNS))) if len(frame) else 0


def _scores(chunks: Iterable[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    found = list(chunks)
    return np.concatenate(found) if found else np.zeros(0, dtype=np.float64)


def _scored(
    model: Trained | lightgbm.Booster, columns: Sequence[str], frame: pandas.DataFrame
) -> npt.NDArray[np.float64]:
    chunks = (
        frame.iloc[start : start + PREDICT_CHUNK] for start in range(0, len(frame), PREDICT_CHUNK)
    )
    return _scores(predict(model, columns, chunks))


def _train(
    tenant_id: str,
    rounds: HeldOutRounds,
    settings: HeldOutSettings,
    params: RankerParams,
    rival: Holder | None,
    allowed: bool,
    sha: str,
    started: float,
) -> tuple[RankerReport, Trained | None]:
    sources = load_rounds(rounds.paths, ["source_url"])["source_url"].unique().tolist()
    test = split_sources(sources, share=settings.test_share, seed=settings.split_seed)
    valid = split_sources(
        [source for source in sources if source not in test],
        share=settings.valid_share,
        seed=settings.split_seed + 1,
    )
    # A round at a time, so only one round's rows are held in full.
    all_test_groups = 0
    found: list[pandas.DataFrame] = []
    for path in rounds.paths:
        frame = load_rounds([path])
        all_test_groups += _groups(frame[frame["source_url"].isin(test).to_numpy()])
        found.append(positive_groups(frame))
        del frame
    kept = pandas.concat(found, ignore_index=True)
    del found
    in_test = kept["source_url"].isin(test).to_numpy()
    in_valid = kept["source_url"].isin(valid).to_numpy()
    parts = {
        "train": kept[~in_test & ~in_valid],
        "valid": kept[in_valid],
        "test": kept[in_test],
    }
    del kept
    counts = {name: _groups(part) for name, part in parts.items()}
    fields: dict[str, object] = {
        "tenant_id": tenant_id,
        "corpus_pages": rounds.pages,
        "body_links": rounds.body_links,
        "feature_set_version": code_digest(),
        "git_sha": sha,
        "settings": settings,
        "params": params,
        "rounds": rounds.summaries,
        "columns": MODEL_COLUMNS,
        "excluded_columns": dict(EXCLUDED_COLUMNS),
        "train_groups": counts["train"],
        "valid_groups": counts["valid"],
        "test_groups": counts["test"],
        "positives": sum(summary.positives for summary in rounds.summaries),
    }
    reason = _too_little(counts)
    if reason is not None:
        return _report(fields, started, skipped_reason=reason), None

    trained = train(parts["train"], parts["valid"], MODEL_COLUMNS, params)
    tested = parts["test"].copy()
    tested[ScorerName.LEARNED.value] = _scored(trained, MODEL_COLUMNS, tested)
    # Like-for-like models, evaluated only: without the link-derived, then the placement columns.
    for scorer, columns in (
        (ScorerName.LEARNED_EXCL_LINK_COUNTS, LIKE_FOR_LIKE_COLUMNS),
        (ScorerName.LEARNED_EXCL_PLACEMENT, EXCL_PLACEMENT_COLUMNS),
    ):
        model = train(parts["train"], parts["valid"], columns, params)
        tested[scorer.value] = _scored(model, columns, tested)
    tested[ScorerName.BASELINE.value] = tested[BASELINE_COLUMN]
    scorers = [
        ScorerName.LEARNED,
        ScorerName.LEARNED_EXCL_LINK_COUNTS,
        ScorerName.LEARNED_EXCL_PLACEMENT,
        ScorerName.BASELINE,
    ]
    if rival is not None:
        refused = _refused_rival(rival, settings, tested)
        if refused is not None:
            # The baseline is the rival instead.
            log.warning(
                "ranker.holder_refused",
                stage=STAGE,
                tenant_id=tenant_id,
                holder_version=rival.version,
                reason=refused,
            )
            rival = None
        else:
            tested[ScorerName.HOLDER.value] = _scored(rival.booster, rival.columns, tested)
            scorers.append(ScorerName.HOLDER)
    metrics: list[RankingMetrics] = [
        ranking_metrics(tested, scorer.value, scorer, all_groups=all_test_groups, seed=params.seed)
        for scorer in scorers
    ]
    entries = importance(trained)
    report = _report(
        fields,
        started,
        best_iteration=trained.best_iteration,
        metrics=tuple(metrics),
        importance=entries,
        dominant_feature=dominant_feature(entries),
        promotion=_promotion(tested, rival, allowed=allowed, seed=params.seed),
    )
    return report, trained


def _refused_rival(
    rival: Holder, settings: HeldOutSettings, tested: pandas.DataFrame
) -> str | None:
    """Why the production model cannot be compared on these test rows, if it cannot: another
    split has shown it some of today's test sources, or it reads other or missing features."""
    split = (rival.split_seed, rival.test_share, rival.valid_share)
    if split != (settings.split_seed, settings.test_share, settings.valid_share):
        return "split_differs"
    if rival.feature_set_version != code_digest():
        return "feature_set_differs"
    if any(column not in tested.columns for column in rival.columns):
        return "columns_missing"
    return None


def _too_little(counts: dict[str, int]) -> str | None:
    if counts["train"] < MIN_TRAIN_GROUPS:
        return (
            f"{counts['train']} training groups with a hidden link, fewer than {MIN_TRAIN_GROUPS}"
        )
    if counts["test"] < MIN_TEST_GROUPS:
        return f"{counts['test']} test groups with a hidden link, fewer than {MIN_TEST_GROUPS}"
    if not counts["valid"]:
        return "no validation group with a hidden link to stop training on"
    return None


def _promotion(
    tested: pandas.DataFrame, rival: Holder | None, *, allowed: bool, seed: int
) -> PromotionDecision:
    name = ScorerName.HOLDER if rival is not None else ScorerName.BASELINE
    return promotion(
        group_ndcg(tested, ScorerName.LEARNED.value, NDCG_K),
        group_ndcg(tested, name.value, NDCG_K),
        sources=group_sources(tested),
        rival_name=name,
        holder_version=None if rival is None else rival.version,
        allowed=allowed,
        seed=seed,
    )


def _report(fields: Mapping[str, object], started: float, **outcome: object) -> RankerReport:
    return RankerReport.model_validate(
        {
            **fields,
            **outcome,
            "seconds": round(time.perf_counter() - started, 3),
            "finished_at": datetime.now(UTC),
        }
    )


async def rank_pairs(
    graph: GraphRepo, mongo: MongoRepo, tenant_id: str, *, cache_dir: Path
) -> tuple[RankReport, Path]:
    """Every candidate pair's score and rank within its source page at
    ``<cache_dir>/<tenant>/ranked_pairs.parquet``, from the tenant's production model; from the
    baseline scorer, with the reason, when there is no promoted model, the registry is
    unreachable or the feature matrix lacks one of the model's columns."""
    started = time.perf_counter()
    folder = cache_folder(cache_dir, tenant_id)
    weights = await mongo.get_scorer_weights(tenant_id) or default_weights()
    unknown = [f.column for f in weights.features if f.column not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"scorer weights {weights.version!r} of {tenant_id!r} name columns the feature "
            f"matrix does not have: {', '.join(unknown)}"
        )
    _, matrix = await assemble_features(graph, mongo, tenant_id, cache_dir=cache_dir)
    model: Holder | None = None
    reason: str | None = NO_MODEL
    try:
        model = await asyncio.to_thread(load_production, tenant_id, cache_dir)
    except RegistryUnavailableError as error:
        reason = f"model registry unreachable ({error.cause_type})"
    if model is not None:
        present = set(pq.read_schema(matrix).names)
        missing = [column for column in model.columns if column not in present]
        reason = None
        if model.feature_set_version != code_digest():
            reason = "model trained on another feature set than the current feature code"
        elif missing:
            reason = f"feature matrix lacks model columns: {', '.join(missing)}"
    path = folder / RANKED_PAIRS_FILE
    pairs = await asyncio.to_thread(
        _rank, tenant_id, matrix, path, None if reason else model, weights
    )
    report = RankReport(
        tenant_id=tenant_id,
        pairs=pairs,
        scorer=ScorerName.BASELINE if reason else ScorerName.LEARNED,
        model_version=None if reason or model is None else model.version,
        fallback_reason=reason,
        seconds=round(time.perf_counter() - started, 3),
    )
    log.info(
        "ranker.ranked",
        stage=STAGE,
        tenant_id=tenant_id,
        pairs=report.pairs,
        scorer=report.scorer.value,
        model_version=report.model_version,
        fallback_reason=report.fallback_reason,
        seconds=report.seconds,
    )
    return report, path


def _batches(matrix: Path, columns: Sequence[str]) -> Iterator[pandas.DataFrame]:
    with pq.ParquetFile(matrix) as file:
        for batch in file.iter_batches(batch_size=PREDICT_CHUNK, columns=list(columns)):
            yield batch.to_pandas()


def _rank(
    tenant_id: str,
    matrix: Path,
    path: Path,
    model: Holder | None,
    weights: ScorerWeights,
) -> int:
    """The ranked pairs written to ``path``; their count."""
    if model is None:
        columns = [feature.column for feature in weights.features]
        frame = pq.read_table(matrix, columns=[*KEY_COLUMNS, *columns]).to_pandas()
        scores = score_frame(frame, weights)["score"].to_numpy(dtype=np.float64)
        keys = frame.loc[:, list(KEY_COLUMNS)]
        del frame
    else:
        keys = pq.read_table(matrix, columns=list(KEY_COLUMNS)).to_pandas()
        scores = _scores(predict(model.booster, model.columns, _batches(matrix, model.columns)))
    # Best first within each source page, ties by target url.
    ranked = keys.assign(score=scores).sort_values(
        ["source_url", "score", "target_url"], ascending=[True, False, True], kind="stable"
    )
    del keys
    ranked["rank_in_source"] = ranked.groupby("source_url", sort=False).cumcount() + 1
    ranked["scorer"] = (ScorerName.BASELINE if model is None else ScorerName.LEARNED).value
    ranked["model_version"] = None if model is None else model.version
    table = pa.Table.from_pandas(
        ranked, schema=RANKED_SCHEMA.with_metadata({"tenant_id": tenant_id}), preserve_index=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(table, path)
    return len(ranked)
