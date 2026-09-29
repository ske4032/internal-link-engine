"""Assemble a tenant's served output from the stage outputs and write it as one run: new links
per source page, ranked and capped, with their extracted anchors or the content gap to close
first; every verdict of the latest link audit; page profiles, hubs, bridges, duplicate groups,
pairs without an anchor and target pages without a keyword. Read-only against the graph and the
stage files; the run is served once complete and replaces the tenant's previous one."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import compress
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas
import pyarrow.parquet as pq
import structlog

from linking_engine.anchor.extraction import Stems
from linking_engine.anchor.generic import generic_overrides, is_generic
from linking_engine.anchor.scoring import brand_tokens, existing_type
from linking_engine.discovery.features import KEY_COLUMNS, code_digest
from linking_engine.discovery.scoring import (
    TOP_CONTRIBUTIONS,
    default_weights,
    rank_tiers,
    score_frame,
)
from linking_engine.ml.ranker_tracking import load_production
from linking_engine.ml.ranking import PREDICT_CHUNK
from linking_engine.models import (
    ActionType,
    AnchorCandidate,
    AnchorMix,
    AnchorPlacement,
    AnchorType,
    BridgeLink,
    BridgeLinkOut,
    BridgeMark,
    BridgePair,
    DuplicateGroup,
    HubSummary,
    PageProfile,
    Recommendation,
    RecommendationReport,
    RecommendationStatus,
    RunInfo,
    ScorerName,
    SiteSummary,
    TargetFix,
    TenantConfig,
    UnanchoredOut,
    UnanchoredReason,
)
from linking_engine.models.anchors import content_gap_finding
from linking_engine.output.collections import (
    BRIDGES,
    DUPLICATES,
    HUBS,
    PAGES,
    RECOMMENDATIONS,
    TARGET_FIXES,
    UNANCHORED,
    recommendation_id,
)
from linking_engine.output.writer import milliseconds
from linking_engine.pipeline.anchor_selection import UNANCHORED_FILE
from linking_engine.pipeline.anchors import cache_folder
from linking_engine.pipeline.bridges import BRIDGES_FILE, HUB_PAIRS_FILE, read_hub_pairs
from linking_engine.pipeline.features import ANCHOR_CHOICES_FILE, assemble_features
from linking_engine.pipeline.ranker import RANKED_PAIRS_FILE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import numpy.typing as npt
    from pydantic import BaseModel

    from linking_engine.anchor.scoring import Brand
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.ml.ranker_tracking import Holder
    from linking_engine.models import (
        ExcludedPage,
        HubPair,
        KeywordRung,
        LinkAuditResult,
        QualitySnapshot,
        ScorerWeights,
    )
    from linking_engine.models.page import HubNode, InboundAnchorText, PageFacts
    from linking_engine.output.writer import OutputWriter

log = structlog.get_logger(__name__)

STAGE: Final = "recommendations"
PACKAGE: Final = "linking-engine"
# AnchorCandidate.text's bound: a longer extracted phrase cannot be served as an anchor.
ANCHOR_MAX_CHARS: Final = 120
BEST_SOURCES: Final = 5
ADD_LINK_LABEL: Final = "add a link"
CONTENT_GAP_LABEL: Final = "content gap: add copy first"
VERDICT_LABELS: Final = {
    ActionType.FIX: "fix this link",
    ActionType.REANCHOR: "change the anchor text",
    ActionType.REMOVE: "review this link",
}
# The existing-link audit's dimensions, in the order they are served as a verdict's signals.
AUDIT_SIGNALS: Final = (
    "anchor_quality_score",
    "keyword_alignment",
    "context_relevance",
    "anchor_target_fit",
    "equity_efficiency",
)
# The stage files read, and the flow that writes each.
REQUIRED_FILES: Final = (
    (RANKED_PAIRS_FILE, "rank-pairs"),
    (ANCHOR_CHOICES_FILE, "anchor-selection"),
    (UNANCHORED_FILE, "anchor-selection"),
    (HUB_PAIRS_FILE, "hub-bridges"),
    (BRIDGES_FILE, "hub-bridges"),
)
STALE_RANKING: Final = "the features or weights changed since rank-pairs; rerun rank-pairs"
MODEL_CHANGED: Final = "production model changed since rank-pairs; rerun rank-pairs"
_GAP_REASONS: Final = frozenset(
    {
        UnanchoredReason.SOURCE_DOES_NOT_MENTION_TOPIC.value,
        UnanchoredReason.TOPIC_MENTIONED_BUT_NO_GOOD_PHRASE.value,
    }
)
_RANKED_COLUMNS: Final = (
    "source_url",
    "target_url",
    "score",
    "rank_in_source",
    "scorer",
    "model_version",
)
_CHOICE_COLUMNS: Final = (
    "source_url",
    "target_url",
    "rank",
    "anchor_type",
    "keyword",
    "phrase",
    "start",
    "end",
    "sentence",
    "sentence_index",
    "score_total",
)
_UNANCHORED_COLUMNS: Final = ("source_url", "target_url", "reason", "advice", "best_score")
_SCORER_WORDS: Final = {
    ScorerName.LEARNED: "the learned ranker",
    ScorerName.BASELINE: "the baseline scorer",
}

Pair = tuple[str, str]
Signals = dict[Pair, tuple[tuple[str, float], ...]]


@dataclass(frozen=True, slots=True)
class Inputs:
    """Everything one run reads of a tenant, read once."""

    tenant_id: str
    run_id: str
    started_at: datetime
    # ADD_LINKs, and content gaps beside them, per source page.
    limit: int
    gap_limit: int
    tier_shares: tuple[float, float]
    scorer: ScorerName
    # _RANKED_COLUMNS but the scorer's; _CHOICE_COLUMNS; _UNANCHORED_COLUMNS.
    ranked: pandas.DataFrame
    choices: pandas.DataFrame
    unanchored: pandas.DataFrame
    hub_pairs: Sequence[HubPair]
    bridge_links: Sequence[BridgeLink]
    audit: Sequence[LinkAuditResult]
    # The stored anchor text of the audited links, by (source url, position).
    anchors: Mapping[tuple[str, int], str]
    excluded: Sequence[ExcludedPage]
    pages: Sequence[PageFacts]
    hubs: Sequence[HubNode]
    titles: Mapping[str, str | None]
    keywords: Mapping[str, tuple[str, KeywordRung]]
    # Each page's ranked keyword texts, the resolved keyword first.
    ranked_keywords: Mapping[str, Sequence[str]]
    inbound: Sequence[InboundAnchorText]
    brand: Brand
    generic_add: frozenset[str] = frozenset()
    generic_remove: frozenset[str] = frozenset()

    @property
    def excluded_urls(self) -> frozenset[str]:
        return frozenset(page.url for page in self.excluded)


@dataclass(frozen=True, slots=True)
class Walk:
    """The ranked pairs, each with its score, tier and what it became, and the new links kept."""

    # The ranked pairs with percentile, tier, candidates (the source's ranked pairs), reason,
    # advice, best_score, excluded and action ("" for no record).
    pairs: pandas.DataFrame
    # The new links kept, source by source: its ADD_LINKs, then its content gaps, each in rank
    # order with place, their rank within their own list.
    emitted: pandas.DataFrame
    not_assessed: int
    # Pairs whose every extracted phrase is too long to serve as an anchor.
    anchors_too_long: int
    # Unanchored pairs outside the ranked pairs: hub bridges that were no candidate.
    unanchored_not_ranked: int


@dataclass(frozen=True, slots=True)
class Assembly:
    """One run's output, every listing in its served order, and its summary."""

    recommendations: tuple[Recommendation, ...]
    pages: tuple[PageProfile, ...]
    hubs: tuple[HubSummary, ...]
    bridges: tuple[BridgePair, ...]
    duplicates: tuple[DuplicateGroup, ...]
    unanchored: tuple[UnanchoredOut, ...]
    target_fixes: tuple[TargetFix, ...]
    summary: SiteSummary

    def collections(self) -> tuple[tuple[str, Sequence[BaseModel]], ...]:
        return (
            (RECOMMENDATIONS, self.recommendations),
            (PAGES, self.pages),
            (HUBS, self.hubs),
            (BRIDGES, self.bridges),
            (DUPLICATES, self.duplicates),
            (UNANCHORED, self.unanchored),
            (TARGET_FIXES, self.target_fixes),
        )


class _Typer:
    """Types anchor texts into a page against its ranked keywords, stemmed in the language of
    the page they are written on, as anchor selection does; repeated texts are typed once."""

    def __init__(self, keywords: Mapping[str, Sequence[str]], brand: Brand) -> None:
        self._keywords = keywords
        self._brand = brand
        self._stems: dict[str | None, Stems] = {}
        self._found: dict[tuple[str, str, str | None], AnchorType] = {}

    def __call__(self, text: str, target_url: str, language: str | None) -> AnchorType:
        wanted = (text, target_url, language)
        found = self._found.get(wanted)
        if found is None:
            if language not in self._stems:
                self._stems[language] = Stems(language)
            found = self._found[wanted] = existing_type(
                text, self._keywords.get(target_url, ()), self._stems[language], self._brand
            )
        return found


async def publish_recommendations(
    graph: GraphRepo,
    mongo: MongoRepo,
    writer: OutputWriter,
    tenant_id: str,
    *,
    cache_dir: Path,
    quality: QualitySnapshot | None = None,
) -> RecommendationReport:
    """The tenant's output assembled from its stage files under ``cache_dir``, the graph and
    its latest link audit, written as a new run that replaces the previous one once complete.
    A missing stage file or link audit raises ValueError naming the flow to run, before
    anything is written."""
    started = time.perf_counter()
    started_at = milliseconds(datetime.now(UTC))
    folder = cache_folder(cache_dir, tenant_id)
    paths = _stage_files(folder, tenant_id)
    marker = await mongo.latest_link_audit_run(tenant_id)
    if marker is None:
        raise ValueError(f"no completed link audit for tenant {tenant_id!r}; run link-audit first")
    audit_run_id, audit_completed = marker
    audit = await mongo.latest_link_audit(tenant_id)
    if any(result.run_id != audit_run_id for result in audit):
        raise ValueError(f"the link audit of {tenant_id!r} changed while it was read; run again")
    weights = await mongo.get_scorer_weights(tenant_id) or default_weights()
    inputs, model_version = await _read_inputs(
        graph, mongo, tenant_id, paths, audit, weights=weights, started_at=started_at
    )
    walked = await asyncio.to_thread(walk, inputs)
    signals, matrix = await _signals(
        graph, mongo, inputs, walked, model_version, weights, cache_dir=cache_dir
    )
    assembly = await asyncio.to_thread(assemble, inputs, walked, signals)

    files = {path.name: _modified(path) for path in (*paths.values(), *filter(None, [matrix]))}
    run = RunInfo(
        tenant_id=tenant_id,
        run_id=inputs.run_id,
        status="writing",
        started_at=started_at,
        scorer=inputs.scorer,
        model_version=model_version,
        weights_version=weights.version,
        feature_code=code_digest(),
        package_version=version(PACKAGE),
        link_audit_run_id=audit_run_id,
        inputs={**files, "link_audit": audit_completed},
        limit_per_source=inputs.limit,
        content_gap_limit=inputs.gap_limit,
        quality=quality,
    )
    await writer.ensure_indexes()
    await writer.begin(run)
    written = {
        name: await writer.write(name, tenant_id, inputs.run_id, models)
        for name, models in assembly.collections()
    }
    await writer.complete(tenant_id, inputs.run_id, assembly.summary, datetime.now(UTC))
    pruned = await writer.prune(tenant_id, inputs.run_id)
    report = RecommendationReport(
        tenant_id=tenant_id,
        run_id=inputs.run_id,
        scorer=inputs.scorer,
        model_version=model_version,
        limit_per_source=inputs.limit,
        content_gap_limit=inputs.gap_limit,
        summary=assembly.summary,
        pairs_not_assessed=walked.not_assessed,
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )
    log.info(
        "recommendations.complete",
        stage=STAGE,
        tenant_id=tenant_id,
        output_run_id=inputs.run_id,
        scorer=inputs.scorer.value,
        model_version=model_version,
        link_audit_run_id=audit_run_id,
        written=written,
        pruned=pruned,
        recommendations={kind.value: n for kind, n in assembly.summary.recommendations.items()},
        pairs_not_assessed=walked.not_assessed,
        anchors_too_long=walked.anchors_too_long,
        unanchored_not_ranked=walked.unanchored_not_ranked,
        quality_run=None if quality is None else quality.mlflow_run_id,
        seconds=report.seconds,
    )
    return report


def _stage_files(folder: Path, tenant_id: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name, flow in REQUIRED_FILES:
        path = folder / name
        if not path.is_file():
            raise ValueError(f"no {name} for tenant {tenant_id!r}; run {flow} first")
        found[name] = path
    return found


def _modified(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC)


async def _read_inputs(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    paths: Mapping[str, Path],
    audit: Sequence[LinkAuditResult],
    *,
    weights: ScorerWeights,
    started_at: datetime,
) -> tuple[Inputs, str | None]:
    ranked, scorer, model_version = await asyncio.to_thread(
        read_ranked, paths[RANKED_PAIRS_FILE], tenant_id
    )
    choices, unanchored = await asyncio.to_thread(
        _read_anchor_files, paths[ANCHOR_CHOICES_FILE], paths[UNANCHORED_FILE]
    )
    hub_pairs = await asyncio.to_thread(read_hub_pairs, paths[HUB_PAIRS_FILE])
    bridge_links = await asyncio.to_thread(read_bridge_links, paths[BRIDGES_FILE])
    excluded = await mongo.excluded_pages(tenant_id)
    titles = await mongo.page_titles_by_url(tenant_id)
    rules = await mongo.get_anchor_rules(tenant_id)
    generic_add, generic_remove = generic_overrides(rules.generic_add, rules.generic_remove)
    skip = frozenset(page.url for page in excluded)
    verdicts = sorted(
        {
            result.source_url
            for result in audit
            if result.verdict is not None
            and result.source_url not in skip
            and result.target_url not in skip
        }
    )
    anchors = {
        (link.source_url, link.position): link.anchor_text
        for link in await mongo.links_for(tenant_id, verdicts)
    }
    ranked_keywords = {
        url: [text for _, text, _ in found]
        for url, found in (await graph.ranked_keywords(tenant_id)).items()
    }
    config = TenantConfig(tenant_id=tenant_id)
    inputs = Inputs(
        tenant_id=tenant_id,
        run_id=uuid.uuid4().hex,
        started_at=started_at,
        limit=config.max_recommendations_per_source,
        gap_limit=config.max_content_gaps_per_source,
        tier_shares=weights.tier_shares,
        scorer=scorer,
        ranked=ranked,
        choices=choices,
        unanchored=unanchored,
        hub_pairs=hub_pairs,
        bridge_links=bridge_links,
        audit=audit,
        anchors=anchors,
        excluded=excluded,
        pages=await graph.page_facts(tenant_id),
        hubs=await graph.hub_nodes(tenant_id),
        titles=titles,
        keywords=await graph.resolved_keywords(tenant_id),
        ranked_keywords=ranked_keywords,
        inbound=await graph.inbound_anchor_texts(tenant_id),
        brand=brand_tokens(titles.values()),
        generic_add=generic_add,
        generic_remove=generic_remove,
    )
    return inputs, model_version


def read_ranked(path: Path, tenant_id: str) -> tuple[pandas.DataFrame, ScorerName, str | None]:
    """The ranked pairs of the tenant's file, and the scorer and model version that ranked
    them. A tenant without a single pair was ranked by nothing: the baseline, no model."""
    metadata = pq.read_schema(path).metadata or {}
    stored = metadata.get(b"tenant_id")
    if stored is not None and stored.decode() != tenant_id:
        raise ValueError(f"{path.name} holds another tenant's pairs; rerun rank-pairs")
    table = pq.read_table(path, columns=list(_RANKED_COLUMNS))
    scorers = table.column("scorer").unique().to_pylist()
    versions = table.column("model_version").unique().to_pylist()
    if len(scorers) > 1 or len(versions) > 1:
        raise ValueError(f"{path.name} mixes scorers or model versions; rerun rank-pairs")
    frame = table.drop_columns(["scorer", "model_version"]).to_pandas()
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{path.name} repeats a pair; rerun rank-pairs")
    if not scorers:
        return frame, ScorerName.BASELINE, None
    return frame, ScorerName(scorers[0]), versions[0]


def _read_anchor_files(
    choices: Path, unanchored: Path
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    found = pq.read_table(choices, columns=list(_CHOICE_COLUMNS)).to_pandas()
    missing = pq.read_table(unanchored, columns=list(_UNANCHORED_COLUMNS)).to_pandas()
    if (
        found.duplicated([*KEY_COLUMNS, "rank"]).any()
        or missing.duplicated(list(KEY_COLUMNS)).any()
    ):
        raise ValueError("the anchor choices repeat a pair; rerun anchor-selection")
    return found, missing


def read_bridge_links(path: Path) -> list[BridgeLink]:
    """The bridge links of a ``bridges.parquet`` file."""
    return [BridgeLink.model_validate(row) for row in pq.read_table(path).to_pylist()]


def percentiles(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Each value's rank among all of them, ties averaged, lowest first, scaled to [0, 1]; a
    single value sits at 0.5."""
    if len(values) < 2:
        return np.full(len(values), 0.5)
    ranks = pandas.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    scaled: npt.NDArray[np.float64] = (ranks - 1) / (len(values) - 1)
    return scaled


def walk(inputs: Inputs) -> Walk:
    """Each source page's ranked pairs in rank order, links first: its first ``limit`` pairs
    with an anchor become ADD_LINK. The pairs whose source lacks the copy become CONTENT_GAP,
    up to ``gap_limit``, only when ranked above the page's last ADD_LINK, or all of them on a
    page without one. Pairs of excluded pages are skipped and the other unanchored reasons add
    no record; a pair with neither an anchor nor a reason is counted as not assessed while the
    walk for links passes it."""
    pairs = inputs.ranked.loc[:, ["source_url", "target_url", "score", "rank_in_source"]].copy()
    values = pairs["score"].to_numpy(dtype=np.float64)
    pairs["percentile"] = percentiles(values)
    pairs["tier"] = rank_tiers(values, pairs["source_url"], pairs["target_url"], inputs.tier_shares)
    pairs["candidates"] = pairs.groupby("source_url")["target_url"].transform("size")
    pairs = pairs.merge(
        inputs.unanchored.loc[:, ["source_url", "target_url", "reason", "advice", "best_score"]],
        on=list(KEY_COLUMNS),
        how="left",
    )
    choices = inputs.choices
    usable = choices["phrase"].str.len() <= ANCHOR_MAX_CHARS
    keys = pandas.MultiIndex.from_frame(pairs.loc[:, list(KEY_COLUMNS)])
    anchored = keys.isin(pandas.MultiIndex.from_frame(choices.loc[usable, list(KEY_COLUMNS)]))
    too_long = ~anchored & keys.isin(
        pandas.MultiIndex.from_frame(choices.loc[:, list(KEY_COLUMNS)])
    )
    excluded = inputs.excluded_urls
    pairs["excluded"] = pairs["source_url"].isin(excluded) | pairs["target_url"].isin(excluded)
    gap = pairs["reason"].isin(_GAP_REASONS).to_numpy() & ~anchored
    pairs["action"] = np.select(
        [pairs["excluded"].to_numpy(), anchored, gap],
        ["", ActionType.ADD_LINK.value, ActionType.CONTENT_GAP.value],
        default="",
    )
    links = _placed(pairs.loc[pairs["action"] == ActionType.ADD_LINK.value], inputs.limit)
    # A gap would outrank a proposed link once the copy is written; on a page without a link,
    # every gap would.
    last = links.groupby("source_url")["rank_in_source"].max()
    gaps = pairs.loc[pairs["action"] == ActionType.CONTENT_GAP.value]
    last_link = gaps["source_url"].map(last)
    gaps = _placed(
        gaps.loc[last_link.isna() | (gaps["rank_in_source"] < last_link)], inputs.gap_limit
    )
    emitted = pandas.concat([links, gaps]).sort_values(
        ["source_url", "action", "place"], kind="stable"
    )
    emitted = emitted.reset_index(drop=True)
    # Where each source that reached the limit stopped; the walk never passes the pairs below.
    stops = links.loc[links["place"] == inputs.limit].set_index("source_url")["rank_in_source"]
    unassessed = pairs.loc[
        ~pairs["excluded"] & ~anchored & ~too_long & pairs["reason"].isna(),
        ["source_url", "rank_in_source"],
    ]
    stop = unassessed["source_url"].map(stops)
    walked = stop.isna() | (unassessed["rank_in_source"] < stop)
    return Walk(
        pairs=pairs,
        emitted=emitted,
        not_assessed=int(walked.sum()),
        anchors_too_long=int((too_long & ~pairs["excluded"].to_numpy()).sum()),
        unanchored_not_ranked=len(inputs.unanchored) - int(pairs["reason"].notna().sum()),
    )


def _placed(pairs: pandas.DataFrame, limit: int) -> pandas.DataFrame:
    """Each source's first ``limit`` pairs in rank order, with their place among them."""
    ordered = pairs.sort_values(["source_url", "rank_in_source"], kind="stable")
    ordered = ordered.assign(place=ordered.groupby("source_url").cumcount() + 1)
    return ordered.loc[ordered["place"] <= limit]


async def _signals(
    graph: GraphRepo,
    mongo: MongoRepo,
    inputs: Inputs,
    walked: Walk,
    model_version: str | None,
    weights: ScorerWeights,
    *,
    cache_dir: Path,
) -> tuple[Signals, Path | None]:
    """The strongest contributions behind each kept new link's ranking, and the feature
    matrix they were read from; none without a new link."""
    if walked.emitted.empty:
        return {}, None
    holder: Holder | None = None
    if inputs.scorer is ScorerName.LEARNED:
        holder = await asyncio.to_thread(load_production, inputs.tenant_id, cache_dir)
        if holder is None or holder.version != model_version:
            raise ValueError(MODEL_CHANGED)
    elif inputs.scorer is not ScorerName.BASELINE:
        raise ValueError(f"pairs ranked by {inputs.scorer.value}; rerun rank-pairs")
    _, matrix = await assemble_features(graph, mongo, inputs.tenant_id, cache_dir=cache_dir)
    emitted = walked.emitted.loc[:, [*KEY_COLUMNS, "score"]]
    if holder is None:
        found = await asyncio.to_thread(baseline_signals, matrix, weights, emitted)
    else:
        found = await asyncio.to_thread(learned_signals, matrix, holder, emitted)
    return found, matrix


def baseline_signals(matrix: Path, weights: ScorerWeights, emitted: pandas.DataFrame) -> Signals:
    """The baseline scorer's top contributions of the ``emitted`` pairs (keys and ranked score).
    Percentiles are relative to the run, so the whole matrix is scored as rank-pairs scored it;
    a pair missing, or scored otherwise, means the ranking is stale."""
    columns = [feature.column for feature in weights.features]
    frame = pq.read_table(matrix, columns=[*KEY_COLUMNS, *columns]).to_pandas()
    scored = score_frame(frame, weights)
    del frame
    joined = emitted.rename(columns={"score": "ranked"}).merge(
        scored, on=list(KEY_COLUMNS), how="left"
    )
    if joined["score"].isna().any() or not np.allclose(
        joined["score"].to_numpy(dtype=np.float64),
        joined["ranked"].to_numpy(dtype=np.float64),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError(STALE_RANKING)
    found: Signals = {}
    tops = [
        (joined[f"top{k}_feature"].tolist(), joined[f"top{k}_contribution"].tolist())
        for k in range(1, TOP_CONTRIBUTIONS + 1)
    ]
    for row, key in enumerate(zip(joined["source_url"], joined["target_url"], strict=True)):
        found[key] = tuple(
            (str(names[row]), round(float(values[row]), 6))
            for names, values in tops
            if isinstance(names[row], str)
        )
    return found


def learned_signals(matrix: Path, holder: Holder, emitted: pandas.DataFrame) -> Signals:
    """The production model's top contributions (TreeSHAP) of the ``emitted`` pairs (keys and
    ranked score), from their rows of the matrix only. The contributions must add up to the
    ranked score: a pair missing, or scored otherwise, means the ranking is stale."""
    columns = list(holder.columns)
    if columns != holder.booster.feature_name():
        raise ValueError("the model's columns differ from its features; rerun rank-pairs")
    wanted: dict[Pair, float] = dict(
        zip(
            zip(emitted["source_url"], emitted["target_url"], strict=True),
            emitted["score"].to_numpy(dtype=np.float64).tolist(),
            strict=True,
        )
    )
    found: Signals = {}
    with pq.ParquetFile(matrix) as file:
        for batch in file.iter_batches(batch_size=PREDICT_CHUNK, columns=[*KEY_COLUMNS, *columns]):
            chunk = batch.to_pandas()
            keys = list(zip(chunk["source_url"], chunk["target_url"], strict=True))
            mask = np.fromiter((key in wanted for key in keys), dtype=bool, count=len(keys))
            if not mask.any():
                continue
            values = chunk.loc[mask, columns].to_numpy(dtype=np.float32, na_value=np.nan)
            contributions = np.asarray(
                holder.booster.predict(values, pred_contrib=True), dtype=np.float64
            )
            for key, row in zip(compress(keys, mask), contributions, strict=True):
                if not math.isclose(float(row.sum()), wanted[key], rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(STALE_RANKING)
                found[key] = top_contributions(columns, row[:-1])
    if len(found) != len(wanted):
        raise ValueError(STALE_RANKING)
    return found


def top_contributions(
    names: Sequence[str], values: npt.NDArray[np.float64]
) -> tuple[tuple[str, float], ...]:
    """The TOP_CONTRIBUTIONS largest by absolute value, sign kept, largest first; ties keep the
    names' order and a zero is no signal."""
    order = np.argsort(-np.abs(values), kind="stable")[:TOP_CONTRIBUTIONS]
    return tuple((names[i], round(float(values[i]), 6)) for i in order if values[i] != 0)


def assemble(
    inputs: Inputs, walked: Walk, signals: Mapping[Pair, Sequence[tuple[str, float]]]
) -> Assembly:
    """Every listing of the run in its served order, and the summary over them."""
    excluded = inputs.excluded_urls
    typer = _Typer(inputs.ranked_keywords, inputs.brand)
    languages = {page.url: page.language for page in inputs.pages}
    added = {
        (source, target)
        for source, target, action in walked.emitted[
            ["source_url", "target_url", "action"]
        ].itertuples(index=False)
        if action == ActionType.ADD_LINK.value
    }
    links = [
        link
        for link in inputs.bridge_links
        if link.source_url not in excluded and link.target_url not in excluded
    ]
    marks = _bridge_marks(links, added)
    records = [
        *_new_links(inputs, walked, signals, marks),
        *_verdicts(inputs, typer, languages),
    ]
    records.sort(key=_record_order)
    bridges = _bridge_pairs(inputs, links, added)
    profiles = _profiles(inputs, records, typer)
    hubs = _hubs(inputs, profiles, records, bridges)
    duplicates = _duplicates(inputs)
    unanchored = _unanchored(walked)
    fixes = _target_fixes(inputs, walked)
    summary = _summary(
        inputs, walked, records, profiles, hubs, bridges, duplicates, unanchored, fixes
    )
    return Assembly(
        recommendations=tuple(records),
        pages=profiles,
        hubs=hubs,
        bridges=bridges,
        duplicates=duplicates,
        unanchored=unanchored,
        target_fixes=fixes,
        summary=summary,
    )


def _record_order(record: Recommendation) -> tuple[str, int, int, str, str]:
    lists = (ActionType.ADD_LINK, ActionType.CONTENT_GAP)
    place = record.rank_in_source if record.action_type in lists else record.position
    return (
        record.source_url,
        lists.index(record.action_type) if record.action_type in lists else len(lists),
        place or 0,
        record.action_type.value,
        record.target_url,
    )


def _bridge_marks(links: Sequence[BridgeLink], added: set[Pair]) -> dict[Pair, BridgeMark]:
    """The first bridge link of each new link, by rank then slot."""
    marks: dict[Pair, BridgeMark] = {}
    for link in sorted(links, key=lambda link: (link.rank, link.slot, link.hub_from, link.hub_to)):
        key = (link.source_url, link.target_url)
        if key in added and key not in marks:
            marks[key] = BridgeMark(
                hub_from=link.hub_from, hub_to=link.hub_to, reasons=link.reasons
            )
    return marks


def _new_links(
    inputs: Inputs,
    walked: Walk,
    signals: Mapping[Pair, Sequence[tuple[str, float]]],
    marks: Mapping[Pair, BridgeMark],
) -> list[Recommendation]:
    emitted = walked.emitted
    anchors = _anchor_candidates(inputs.choices, emitted)
    scorer = _SCORER_WORDS.get(inputs.scorer, inputs.scorer.value.replace("_", " "))
    found: list[Recommendation] = []
    for row in emitted.itertuples(index=False):
        key = (str(row.source_url), str(row.target_url))
        action = ActionType(str(row.action))
        pair_signals = tuple(signals.get(key, ()))
        gap = action is ActionType.CONTENT_GAP
        reason = UnanchoredReason(str(row.reason)) if gap else None
        found.append(
            Recommendation(
                id=recommendation_id(inputs.tenant_id, action, key[0], key[1], None),
                run_id=inputs.run_id,
                source_url=key[0],
                target_url=key[1],
                action_type=action,
                label=CONTENT_GAP_LABEL if gap else ADD_LINK_LABEL,
                finding=None if reason is None else content_gap_finding(reason),
                advice=str(row.advice) if gap else None,
                score=round(100 * float(row.percentile), 1),
                tier=int(row.tier),
                rank_in_source=int(row.place),
                status=RecommendationStatus.PENDING,
                proposed_anchors=None if gap else anchors[key],
                bridge=marks.get(key),
                rationale=_rationale(
                    scorer, int(row.rank_in_source), int(row.candidates), pair_signals
                ),
                signals=pair_signals,
                created_at=inputs.started_at,
            )
        )
    return found


def _anchor_candidates(
    choices: pandas.DataFrame, emitted: pandas.DataFrame
) -> dict[Pair, tuple[AnchorCandidate, ...]]:
    """Each kept ADD_LINK's servable anchors, the chosen one first, then the alternatives."""
    wanted = emitted.loc[emitted["action"] == ActionType.ADD_LINK.value, list(KEY_COLUMNS)]
    rows = choices.loc[choices["phrase"].str.len() <= ANCHOR_MAX_CHARS].merge(
        wanted, on=list(KEY_COLUMNS), how="inner"
    )
    found: defaultdict[Pair, list[AnchorCandidate]] = defaultdict(list)
    for row in rows.sort_values([*KEY_COLUMNS, "rank"], kind="stable").itertuples(index=False):
        found[(str(row.source_url), str(row.target_url))].append(
            AnchorCandidate(
                text=str(row.phrase),
                anchor_type=AnchorType(str(row.anchor_type)),
                source="EXTRACTED",
                # A total carries the profile bonus on top of a share, so it can pass 1.
                score=min(max(float(row.score_total), 0.0), 1.0),
                keyword=str(row.keyword),
                placement=AnchorPlacement(
                    sentence=str(row.sentence),
                    sentence_index=int(row.sentence_index),
                    start=int(row.start),
                    end=int(row.end),
                ),
            )
        )
    return {key: tuple(candidates) for key, candidates in found.items()}


def _rationale(
    scorer: str, rank: int, candidates: int, signals: Sequence[tuple[str, float]]
) -> str:
    lead = f"Ranked {rank} of the {candidates} candidate targets of this page by {scorer}"
    if not signals:
        return f"{lead}."
    named = [
        name.replace("_", " ") + (" (against)" if value < 0 else "") for name, value in signals
    ]
    listed = named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"
    return f"{lead}; strongest signals: {listed}."


def _verdicts(
    inputs: Inputs, typer: _Typer, languages: Mapping[str, str | None]
) -> list[Recommendation]:
    excluded = inputs.excluded_urls
    found: list[Recommendation] = []
    for result in inputs.audit:
        if result.verdict is None or result.source_url in excluded or result.target_url in excluded:
            continue
        action = result.verdict
        stored = inputs.anchors.get((result.source_url, result.position), "").strip()
        proposed: tuple[AnchorCandidate, ...] | None = None
        if action is ActionType.REANCHOR and result.proposed_anchor is not None:
            phrase = result.proposed_anchor
            if len(phrase) <= ANCHOR_MAX_CHARS:
                proposed = (
                    AnchorCandidate(
                        text=phrase,
                        anchor_type=typer(
                            phrase, result.target_url, languages.get(result.source_url)
                        ),
                        source="EXTRACTED",
                    ),
                )
        found.append(
            Recommendation(
                id=recommendation_id(
                    inputs.tenant_id,
                    action,
                    result.source_url,
                    result.target_url,
                    result.position,
                ),
                run_id=inputs.run_id,
                source_url=result.source_url,
                target_url=result.target_url,
                action_type=action,
                label=VERDICT_LABELS[action],
                position=result.position,
                status=RecommendationStatus.PENDING,
                current_anchor=stored or None,
                proposed_anchors=proposed,
                issue_flags=tuple(sorted(result.issue_flags)),
                fix_target=result.fix_target if action is ActionType.FIX else None,
                rationale="; ".join(result.reasons),
                signals=tuple(
                    (name, float(value))
                    for name in AUDIT_SIGNALS
                    if (value := getattr(result, name)) is not None
                ),
                created_at=inputs.started_at,
            )
        )
    return found


def _bridge_pairs(
    inputs: Inputs, links: Sequence[BridgeLink], added: set[Pair]
) -> tuple[BridgePair, ...]:
    """One per hub pair with a reason or a link, its links in slot order."""
    by_pair: defaultdict[tuple[str | None, int, int], list[BridgeLink]] = defaultdict(list)
    for link in links:
        low, high = sorted((link.hub_from, link.hub_to))
        by_pair[(link.language, low, high)].append(link)
    known = {(pair.language, pair.hub_a, pair.hub_b) for pair in inputs.hub_pairs}
    if set(by_pair) - known:
        raise ValueError(
            f"{BRIDGES_FILE} has links of hub pairs missing from {HUB_PAIRS_FILE}; "
            "rerun hub-bridges"
        )
    found: list[BridgePair] = []
    for pair in inputs.hub_pairs:
        joined = sorted(
            by_pair.get((pair.language, pair.hub_a, pair.hub_b), ()),
            key=lambda link: (link.hub_from, link.slot, link.rank, link.source_url),
        )
        if not pair.reasons and not joined:
            continue
        found.append(
            BridgePair(
                **pair.model_dump(),
                links=tuple(
                    BridgeLinkOut(
                        **link.model_dump(exclude={"language"}),
                        recommendation_id=recommendation_id(
                            inputs.tenant_id,
                            ActionType.ADD_LINK,
                            link.source_url,
                            link.target_url,
                            None,
                        )
                        if (link.source_url, link.target_url) in added
                        else None,
                    )
                    for link in joined
                ),
            )
        )
    found.sort(key=lambda pair: (pair.hub_a, pair.hub_b, pair.language or ""))
    return tuple(found)


def _profiles(
    inputs: Inputs, records: Sequence[Recommendation], typer: _Typer
) -> tuple[PageProfile, ...]:
    excluded = inputs.excluded_urls
    mix: defaultdict[str, Counter[AnchorType]] = defaultdict(Counter)
    for anchor in inputs.inbound:
        text = anchor.anchor_text
        if not text.strip() or is_generic(
            text, add=inputs.generic_add, remove=inputs.generic_remove
        ):
            continue
        mix[anchor.target_url][typer(text, anchor.target_url, anchor.source_language)] += (
            anchor.links
        )
    out: Counter[str] = Counter()
    into: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    for record in records:
        if record.position is None:
            out[record.source_url] += 1
            into[record.target_url] += 1
        else:
            verdicts[record.source_url] += 1
    found: list[PageProfile] = []
    for page in sorted(inputs.pages, key=lambda page: page.url):
        if page.url in excluded:
            continue
        keyword = inputs.keywords.get(page.url)
        kinds = mix.get(page.url, Counter())
        profile = PageProfile(
            url=page.url,
            title=inputs.titles.get(page.url),
            language=page.language,
            page_type=page.page_type,
            word_count=page.word_count,
            inbound=page.inbound,
            outbound=page.outbound,
            crawl_depth=page.crawl_depth,
            page_rank_percentile=page.page_rank_percentile,
            hub_id=page.hub_id if page.hub_id is not None and page.hub_id >= 0 else None,
            is_hub_pillar=page.is_hub_pillar,
            is_orphan=page.is_orphan,
            orphan_label=page.orphan_label,
            is_dead_end=page.is_dead_end,
            duplicate_group=page.duplicate_group,
            is_canonical=page.is_canonical,
            target_keyword=None if keyword is None else keyword[0],
            keyword_rung=None if keyword is None else keyword[1],
            anchor_mix=AnchorMix(
                exact=kinds[AnchorType.EXACT],
                partial=kinds[AnchorType.PARTIAL],
                natural=kinds[AnchorType.NATURAL],
                branded=kinds[AnchorType.BRANDED],
            ),
            recommendations_out=out[page.url],
            recommendations_in=into[page.url],
            audit_verdicts_out=verdicts[page.url],
        )
        found.append(profile)
    return tuple(found)


def _hubs(
    inputs: Inputs,
    profiles: Sequence[PageProfile],
    records: Sequence[Recommendation],
    bridges: Sequence[BridgePair],
) -> tuple[HubSummary, ...]:
    """The active hubs, by id."""
    excluded = inputs.excluded_urls
    members: defaultdict[int, list[PageProfile]] = defaultdict(list)
    for profile in profiles:
        if profile.hub_id is not None:
            members[profile.hub_id].append(profile)
    hub_of = {profile.url: profile.hub_id for profile in profiles}
    incoming = Counter(
        hub_of.get(record.target_url) for record in records if record.position is None
    )
    bridged: defaultdict[int, set[int]] = defaultdict(set)
    for pair in bridges:
        bridged[pair.hub_a].add(pair.hub_b)
        bridged[pair.hub_b].add(pair.hub_a)
    found: list[HubSummary] = []
    for hub in sorted(inputs.hubs, key=lambda hub: hub.hub_id):
        if not hub.active:
            continue
        pages = members.get(hub.hub_id, [])
        languages = {page.language for page in pages}
        pillar = hub.pillar_url if hub.pillar_url not in excluded else None
        found.append(
            HubSummary(
                hub_id=hub.hub_id,
                language=languages.pop() if len(languages) == 1 else None,
                size=hub.size,
                pillar_url=pillar,
                pillar_title=None if pillar is None else inputs.titles.get(pillar),
                orphan_pages=sum(page.is_orphan for page in pages),
                dead_end_pages=sum(page.is_dead_end for page in pages),
                recommendations_in=incoming[hub.hub_id],
                bridge_hubs=tuple(sorted(bridged.get(hub.hub_id, ()))),
            )
        )
    return tuple(found)


def _duplicates(inputs: Inputs) -> tuple[DuplicateGroup, ...]:
    """Each duplicate group with exactly one canonical page and a copy, by canonical url."""
    excluded = inputs.excluded_urls
    groups: defaultdict[int, list[PageFacts]] = defaultdict(list)
    for page in inputs.pages:
        if page.duplicate_group is not None and page.url not in excluded:
            groups[page.duplicate_group].append(page)
    found: list[DuplicateGroup] = []
    for group_id, pages in sorted(groups.items()):
        canonicals = [page.url for page in pages if page.is_canonical]
        copies = sorted(page.url for page in pages if not page.is_canonical)
        if len(canonicals) != 1 or not copies:
            log.warning(
                "recommendations.duplicate_group_skipped",
                stage=STAGE,
                tenant_id=inputs.tenant_id,
                group=group_id,
                canonicals=len(canonicals),
                copies=len(copies),
            )
            continue
        found.append(
            DuplicateGroup(group_id=group_id, canonical=canonicals[0], copies=tuple(copies))
        )
    found.sort(key=lambda group: group.canonical)
    return tuple(found)


def _unanchored(walked: Walk) -> tuple[UnanchoredOut, ...]:
    """Every ranked pair without an anchor, by source, its rank, then target."""
    rows = walked.pairs.loc[walked.pairs["reason"].notna() & ~walked.pairs["excluded"]]
    gaps = {
        (source, target)
        for source, target, action in walked.emitted[
            ["source_url", "target_url", "action"]
        ].itertuples(index=False)
        if action == ActionType.CONTENT_GAP.value
    }
    found = [
        UnanchoredOut(
            source_url=str(row.source_url),
            target_url=str(row.target_url),
            reason=UnanchoredReason(str(row.reason)),
            advice=str(row.advice),
            best_score=None if pandas.isna(row.best_score) else float(row.best_score),
            rank_in_source=int(row.rank_in_source),
            recommended=(row.source_url, row.target_url) in gaps,
        )
        for row in rows.sort_values(
            ["source_url", "rank_in_source", "target_url"], kind="stable"
        ).itertuples(index=False)
    ]
    return tuple(found)


def _target_fixes(inputs: Inputs, walked: Walk) -> tuple[TargetFix, ...]:
    """One per target a ranked pair failed to reach for want of a keyword: its best waiting
    sources by ranker score; most waiting first."""
    pairs = walked.pairs
    rows = pairs.loc[
        (pairs["reason"] == UnanchoredReason.TARGET_PAGE_HAS_NO_KEYWORD.value) & ~pairs["excluded"]
    ].sort_values(
        ["target_url", "score", "source_url"], ascending=[True, False, True], kind="stable"
    )
    found: list[TargetFix] = []
    for target, group in rows.groupby("target_url", sort=True):
        sources = list(dict.fromkeys(group["source_url"].tolist()))
        found.append(
            TargetFix(
                target_url=str(target),
                title=inputs.titles.get(str(target)),
                fix=str(group["advice"].iloc[0]),
                waiting_sources=len(sources),
                best_sources=tuple(sources[:BEST_SOURCES]),
            )
        )
    found.sort(key=lambda fix: (-fix.waiting_sources, fix.target_url))
    return tuple(found)


def _summary(
    inputs: Inputs,
    walked: Walk,
    records: Sequence[Recommendation],
    profiles: Sequence[PageProfile],
    hubs: Sequence[HubSummary],
    bridges: Sequence[BridgePair],
    duplicates: Sequence[DuplicateGroup],
    unanchored: Sequence[UnanchoredOut],
    fixes: Sequence[TargetFix],
) -> SiteSummary:
    excluded = inputs.excluded_urls
    new = [record for record in records if record.position is None]
    per_source = Counter(record.source_url for record in new)
    links = Counter(
        record.source_url for record in new if record.action_type is ActionType.ADD_LINK
    )
    sources = set(walked.pairs.loc[~walked.pairs["excluded"], "source_url"].tolist())
    audited = [
        result
        for result in inputs.audit
        if result.source_url not in excluded and result.target_url not in excluded
    ]
    return SiteSummary(
        pages=len(profiles),
        excluded_pages=Counter(page.reason for page in inputs.excluded),
        orphan_pages=Counter(
            profile.orphan_label for profile in profiles if profile.orphan_label is not None
        ),
        dead_end_pages=sum(profile.is_dead_end for profile in profiles),
        duplicate_groups=len(duplicates),
        duplicate_copies=sum(len(group.copies) for group in duplicates),
        hubs=len(hubs),
        bridge_pairs=len(bridges),
        # Rank 1 is a slot's proposal; ranks 2-3 are its alternatives.
        bridge_links=sum(link.rank == 1 for pair in bridges for link in pair.links),
        recommendations=Counter(record.action_type for record in records),
        tiers=Counter(record.tier for record in new if record.tier is not None),
        sources_with_recommendations=len(per_source),
        sources_below_limit=sum(links[source] < inputs.limit for source in sources),
        links_audited=len(audited),
        unverified_links=sum(result.unverified for result in audited),
        audit_flags=Counter(flag for result in audited for flag in result.issue_flags),
        unanchored=Counter(row.reason for row in unanchored),
        target_fixes=len(fixes),
    )


def summarise_recommendations(report: RecommendationReport) -> str:
    """A short prose record of one run, for the MLflow run description; no urls."""
    summary = report.summary
    actions = (
        ", ".join(f"{kind.value} {n}" for kind, n in sorted(summary.recommendations.items()))
        or "none"
    )
    tiers = ", ".join(f"tier {tier} {n}" for tier, n in sorted(summary.tiers.items())) or "none"
    unanchored = (
        ", ".join(f"{reason.value} {n}" for reason, n in sorted(summary.unanchored.items()))
        or "none"
    )
    excluded = (
        ", ".join(f"{reason.value} {n}" for reason, n in sorted(summary.excluded_pages.items()))
        or "none"
    )
    orphans = (
        ", ".join(f"{label.value} {n}" for label, n in sorted(summary.orphan_pages.items()))
        or "none"
    )
    ranked_by = report.scorer.value + (
        f" version {report.model_version}" if report.model_version else ""
    )
    return "\n".join(
        [
            f"Recommendations run {report.run_id} of tenant {report.tenant_id}, ranked by "
            f"{ranked_by}, at most {report.limit_per_source} links and "
            f"{report.content_gap_limit} content gaps per source page; {report.seconds:.1f}s.",
            f"Recommendations: {actions}. New links by tier: {tiers}.",
            f"{summary.sources_with_recommendations} source pages with a new link or a gap, "
            f"{summary.sources_below_limit} with fewer links than the limit; "
            f"{report.pairs_not_assessed} ranked pairs passed without an anchor choice or a "
            "reason.",
            f"{summary.links_audited} existing links audited, {summary.unverified_links} "
            "unverified.",
            f"Pairs without an anchor: {unanchored}. {summary.target_fixes} target pages need a "
            "keyword first.",
            f"{summary.pages} pages; excluded {excluded}; orphans {orphans}; "
            f"{summary.dead_end_pages} dead ends; {summary.duplicate_groups} duplicate groups "
            f"with {summary.duplicate_copies} copies; {summary.hubs} hubs, "
            f"{summary.bridge_pairs} bridged hub pairs with {summary.bridge_links} bridge links.",
        ]
    )
