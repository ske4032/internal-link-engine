"""Held-out rounds as ranker training data: each round hides a disjoint share of the body links,
recomputes everything link-derived on that view, retrieves and chooses anchors on it, and labels
the candidate pairs of the production feature matrix 1 when they are a hidden link. Read-only
against both stores; each round is one Parquet file under the tenant's cache folder."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas
import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from pydantic import BaseModel, ValidationError

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.discovery.features import (
    CHUNK_PAIRS,
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    code_digest,
    feature_chunks,
    missing_pages,
    page_contexts,
    to_frame,
)
from linking_engine.discovery.scoring import default_weights, score_frame, weights_hash
from linking_engine.discovery.signals import build_page_signals
from linking_engine.errors import DatabaseReadError
from linking_engine.gsc import fit_ctr_curve
from linking_engine.ingest.markdown_clean import body_hash
from linking_engine.ml.quality import hide_links
from linking_engine.ml.ranking import LABEL_COLUMN, PLACEMENT_COLUMNS, placement_shares
from linking_engine.models import AnchorTypeProfile, ExtractionSettings, RoundSummary
from linking_engine.pipeline.anchor_selection import compute_anchor_choices
from linking_engine.pipeline.anchors import AnchorView, cache_folder, write_atomically
from linking_engine.pipeline.quality import held_out_view

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set

    import numpy.typing as npt

    from linking_engine.discovery.features import PageContext
    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import (
        AnchorChoice,
        CandidateSet,
        HeldOutSettings,
        KeywordSource,
        LinkGraphSnapshot,
        ScorerWeights,
    )

log = structlog.get_logger(__name__)

STAGE: Final = "ranking-data"
ROUNDS_FOLDER: Final = Path("ranker") / "rounds"
ROUND_COLUMN: Final = "round"
BASELINE_COLUMN: Final = "baseline_score"
ROUND_SCHEMA: Final = pa.schema(
    [
        pa.field(ROUND_COLUMN, pa.int16(), nullable=False),
        *(pa.field(name, pa.string(), nullable=False) for name in KEY_COLUMNS),
        pa.field(LABEL_COLUMN, pa.int8(), nullable=False),
        pa.field(BASELINE_COLUMN, pa.float64(), nullable=False),
        *(pa.field(name, pa.float32()) for name in FEATURE_COLUMNS),
    ]
)
# The modules besides the feature code whose code decides a round's rows.
_CODE_MODULES: Final = (
    __name__,
    "linking_engine.pipeline.anchors",
    "linking_engine.pipeline.anchor_selection",
    "linking_engine.pipeline.semantic_anchors",
    "linking_engine.pipeline.quality",
    "linking_engine.pipeline.text_vectors",
    "linking_engine.anchor.extraction",
    "linking_engine.anchor.generic",
    "linking_engine.anchor.keywords",
    "linking_engine.anchor.scoring",
    "linking_engine.anchor.semantic",
    "linking_engine.discovery.candidates",
    "linking_engine.discovery.scoring",
    "linking_engine.graph.algorithms",
    "linking_engine.ml.quality",
)
# The word lists of linking_engine.anchor that extraction reads.
_DATA_FOLDERS: Final = ("stopwords", "months")


@dataclass(frozen=True, slots=True)
class HeldOutRounds:
    """One Parquet file per round, its summary and whether it came from the cache."""

    key: str
    paths: tuple[Path, ...]
    summaries: tuple[RoundSummary, ...]
    cache_hit: tuple[bool, ...]
    # Crawled pages, and the distinct body links between them that rounds hide from.
    pages: int
    body_links: int
    # Crawled pages without an inbound body link on the full graph: no round can hide a link to
    # them, so as targets they are labelled 0 by construction.
    orphan_targets: frozenset[str]


async def held_out_rounds(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    settings: HeldOutSettings,
    cache_dir: Path,
    voyage: VoyageClient | None,
) -> HeldOutRounds:
    """Every round's labelled candidate pairs at
    ``<cache_dir>/<tenant>/ranker/rounds/<key>.round-<r>.parquet``: round, source and target
    url, label, the baseline score over the round's pairs and FEATURE_COLUMNS as float32. The
    key covers the code, the stored inputs, the share and seed and the scorer weights, so a
    round is rebuilt only when something it depends on changed."""
    folder = cache_folder(cache_dir, tenant_id) / ROUNDS_FOLDER
    weights = await mongo.get_scorer_weights(tenant_id) or default_weights()
    unknown = [f.column for f in weights.features if f.column not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"scorer weights {weights.version!r} of {tenant_id!r} name columns the feature "
            f"matrix does not have: {', '.join(unknown)}"
        )
    snapshot = await graph.link_graph(tenant_id)
    structure = await graph.page_structure(tenant_id)
    context = await graph.community_context(tenant_id)
    vectors = await graph.content_vectors(tenant_id)
    copies = await graph.non_canonical_copies(tenant_id)
    queries = await mongo.gsc_queries(tenant_id)
    metrics = await mongo.gsc_metrics(tenant_id)
    stats = await mongo.gsc_query_stats(tenant_id)
    strategic = await mongo.strategic_keywords(tenant_id)
    # What anchor choice reads again each round, only to key the rounds on.
    anchor_inputs: list[Iterable[object]] = [
        _ranked_rows(await graph.ranked_keywords(tenant_id)),
        await graph.embedding_models(tenant_id),
        [await mongo.get_extraction_settings(tenant_id) or ExtractionSettings()],
        [await mongo.get_anchor_rules(tenant_id)],
        [await mongo.get_anchor_type_profile(tenant_id) or AnchorTypeProfile()],
        sorted(copies),
        [await graph.candidate_targets(tenant_id)],
        await mongo.page_titles(tenant_id),
        [await _link_text_digest(graph, tenant_id), await _record_digest(mongo, tenant_id)],
    ]
    key = await asyncio.to_thread(
        rounds_key,
        tenant_id,
        settings,
        weights,
        snapshot,
        vectors,
        [structure, context, queries, metrics, stats, strategic, *anchor_inputs],
        voyage_model=None if voyage is None else voyage.model,
    )
    del anchor_inputs
    crawled = {
        url
        for url, placeholder in zip(snapshot.pages, snapshot.placeholders, strict=True)
        if not placeholder
    }
    body_links = frozenset(
        (s, t) for s, t in snapshot.links if s != t and s in crawled and t in crawled
    )
    curve = fit_ctr_curve(stats)
    del stats
    pool = vectors.keys() - copies
    languages = {page.url: page.language for page in structure}
    strategic_pairs = [(row.url, row.keyword) for row in strategic]

    paths: list[Path] = []
    summaries: list[RoundSummary] = []
    hits: list[bool] = []
    for round_ in range(settings.rounds):
        started = time.perf_counter()
        path = folder / f"{key}.round-{round_}.parquet"
        summary = await asyncio.to_thread(_cached, tenant_id, path, key, round_)
        hit, partial = summary is not None, False
        partial_reason: str | None = None
        if summary is None:
            hidden = hide_links(body_links, share=settings.share, seed=settings.seed, fold=round_)
            view = await asyncio.to_thread(
                held_out_view, snapshot, structure, context, vectors, hidden
            )
            signals = build_page_signals(view.context, queries, strategic_pairs)
            try:
                pages = page_contexts(
                    view.structure, signals, view.links, metrics, strategic, curve
                )
            except ValueError as error:
                raise DatabaseReadError(
                    "neo4j", f"feature inputs of {tenant_id!r}: {error}"
                ) from error
            held = await retrieve_candidates(
                graph, tenant_id, links=view.links, vectors=vectors, stage=STAGE
            )
            missing = missing_pages(held.targets, pages)
            if missing:
                raise DatabaseReadError(
                    "neo4j",
                    f"{len(missing)} candidate pages of {tenant_id!r} are no longer crawled "
                    f"pages, first {missing[0]!r}",
                )
            selection = await compute_anchor_choices(
                graph,
                mongo,
                tenant_id,
                cache_dir=cache_dir,
                voyage=voyage,
                view=AnchorView(held, hidden),
            )
            # Voyage failed mid-round: a rerun can place more anchors, so this is never reused.
            partial = voyage is not None and selection.skipped_reason is not None
            partial_reason = selection.skipped_reason if partial else None
            summary = await asyncio.to_thread(
                _build_round,
                tenant_id,
                round_,
                path,
                key,
                held,
                pages,
                hidden,
                placements(selection.choices),
                weights,
                recoverable=recoverable(hidden, held, pool, languages),
                partial=partial,
            )
            del view, signals, pages, held, selection
        paths.append(path)
        summaries.append(summary)
        hits.append(hit)
        log.info(
            "ranker.round",
            stage=STAGE,
            tenant_id=tenant_id,
            round=round_,
            hidden=summary.hidden,
            recoverable=summary.recoverable,
            pairs=summary.pairs,
            positives=summary.positives,
            groups_with_positive=summary.groups_with_positive,
            positive_placement_share=summary.positive_placement_share,
            negative_placement_share=summary.negative_placement_share,
            cache_hit=hit,
            partial=partial,
            partial_reason=partial_reason,
            seconds=round(time.perf_counter() - started, 3),
        )
    pruned = await asyncio.to_thread(_prune, folder, key)
    if pruned:
        log.info("ranker.rounds_pruned", stage=STAGE, tenant_id=tenant_id, files=pruned)
    return HeldOutRounds(
        key=key,
        paths=tuple(paths),
        summaries=tuple(summaries),
        cache_hit=tuple(hits),
        pages=len(crawled),
        body_links=len(body_links),
        orphan_targets=frozenset(crawled - {target for _, target in body_links}),
    )


def load_rounds(paths: Iterable[Path], columns: Sequence[str] | None = None) -> pandas.DataFrame:
    """The rows of every round file, in file order; ``columns`` of ROUND_SCHEMA, else all."""
    tables = [
        pq.read_table(
            path, columns=None if columns is None else list(columns)
        ).replace_schema_metadata()
        for path in paths
    ]
    if not tables:
        schema = (
            ROUND_SCHEMA if columns is None else pa.schema([ROUND_SCHEMA.field(c) for c in columns])
        )
        return schema.empty_table().to_pandas()
    return pa.concat_tables(tables).to_pandas()


def placements(
    choices: Iterable[AnchorChoice],
) -> dict[tuple[str, str], tuple[float | None, float | None]]:
    """(source, target) of every chosen anchor with a placement feature, to its context
    relevance and anchor-target fit, as the feature matrix reads them from the choices file."""
    return {
        (choice.match.source_url, choice.match.target_url): (
            choice.context_relevance,
            choice.anchor_target_fit,
        )
        for choice in choices
        if choice.rank == 1
        and (choice.context_relevance is not None or choice.anchor_target_fit is not None)
    }


def recoverable(
    hidden: Set[tuple[str, str]],
    found: CandidateSet,
    pool: Set[str],
    languages: Mapping[str, str | None],
) -> int:
    """Hidden links a retrieval over the view could return: the target is a retrieval target
    and the source is in the pool in the target's language."""
    targets = {entry.target_url for entry in found.targets}
    return sum(
        1
        for source, target in hidden
        if target in targets and source in pool and languages.get(source) == languages.get(target)
    )


@functools.cache
def _code_digest() -> str:
    digest = hashlib.sha256()
    for name in _CODE_MODULES:
        path = importlib.import_module(name).__file__
        if path is None:
            raise RuntimeError(f"module {name} has no file to hash")
        digest.update(Path(path).read_bytes())
    anchor = resources.files("linking_engine.anchor")
    for folder in _DATA_FOLDERS:
        for item in sorted(anchor.joinpath(folder).iterdir(), key=lambda entry: entry.name):
            digest.update(f"{folder}/{item.name}\n".encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


async def _link_text_digest(graph: GraphRepo, tenant_id: str) -> str:
    """sha256 over every stored link's source, position, anchor, surrounding text and generic
    flag, in stored order, a batch at a time."""
    digest = hashlib.sha256()
    async for texts in graph.iter_link_texts(tenant_id):
        for text in texts:
            row = [
                text.source_url,
                text.position,
                text.anchor_text,
                text.surrounding_text,
                text.anchor_generic,
            ]
            digest.update(_json(row).encode())
            digest.update(b"\n")
    return digest.hexdigest()


async def _record_digest(mongo: MongoRepo, tenant_id: str) -> str:
    """sha256 over each stored page's url, language, heading hash and body hash, sorted by
    url; only the hashes are held, never the texts."""
    rows: list[str] = []
    async for records in mongo.iter_page_records(tenant_id):
        rows.extend(
            _json(
                [
                    record.url,
                    record.language,
                    body_hash("\n".join(heading.text for heading in record.headings)),
                    body_hash(record.body_text),
                ]
            )
            for record in records
        )
    rows.sort()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _prune(folder: Path, key: str) -> int:
    """Delete the round files of every other key in the tenant's rounds folder; their count."""
    if not folder.is_dir():
        return 0
    stale = [
        path for path in folder.glob("*.round-*.parquet") if not path.name.startswith(f"{key}.")
    ]
    for path in stale:
        path.unlink(missing_ok=True)
    return len(stale)


def _json(row: object) -> str:
    value = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _ranked_rows(
    ranked: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
) -> list[tuple[str, int, str, str]]:
    return [
        (url, rank, text, source.value)
        for url, rows in ranked.items()
        for rank, text, source in rows
    ]


def rounds_key(
    tenant_id: str,
    settings: HeldOutSettings,
    weights: ScorerWeights,
    snapshot: LinkGraphSnapshot,
    vectors: Mapping[str, npt.NDArray[np.float32]],
    inputs: Sequence[Iterable[object]],
    *,
    voyage_model: str | None,
) -> str:
    """sha256 over the tenant, the code of the features and of the rounds, the share and seed,
    the scorer weights, the Voyage model, the link snapshot, the content vectors and every
    other stored input, each collection hashed sorted so the key does not depend on read order."""
    digest = hashlib.sha256()

    def add(value: object) -> None:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")

    add(
        [
            tenant_id,
            code_digest(),
            _code_digest(),
            FEATURE_COLUMNS,
            settings.share,
            settings.seed,
            weights_hash(weights),
            voyage_model,
        ]
    )
    add(sorted(zip(snapshot.pages, snapshot.placeholders, strict=True)))
    add(sorted(snapshot.links))
    for rows in inputs:
        add(sorted(_json(row) for row in rows))
    # Streamed a vector at a time from its own buffer, never copied whole.
    for url in sorted(vectors):
        digest.update(url.encode())
        digest.update(memoryview(np.ascontiguousarray(vectors[url], dtype=np.float32)))
    return digest.hexdigest()


def _cached(tenant_id: str, path: Path, key: str, round_: int) -> RoundSummary | None:
    """The summary of the round's cached file; None when there is none, or it is partial,
    unreadable or not this key's round, so the round is built again."""
    if not path.is_file():
        return None
    try:
        schema = pq.read_schema(path)
    except (OSError, pa.ArrowException) as error:
        log.warning(
            "ranker.round_cache_unreadable",
            stage=STAGE,
            tenant_id=tenant_id,
            round=round_,
            error=type(error).__name__,
        )
        return None
    meta = schema.metadata or {}
    if meta.get(b"partial") == b"true":
        return None
    expected = {
        b"tenant_id": tenant_id.encode(),
        b"cache_key": key.encode(),
        b"round": str(round_).encode(),
        b"partial": b"false",
    }
    if schema.remove_metadata().equals(ROUND_SCHEMA) and all(
        meta.get(name) == value for name, value in expected.items()
    ):
        try:
            return RoundSummary.model_validate_json(meta.get(b"summary", b""))
        except ValidationError:
            pass
    log.warning("ranker.round_cache_invalid", stage=STAGE, tenant_id=tenant_id, round=round_)
    return None


def _build_round(
    tenant_id: str,
    round_: int,
    path: Path,
    key: str,
    found: CandidateSet,
    pages: Mapping[str, PageContext],
    hidden: Set[tuple[str, str]],
    placed: Mapping[tuple[str, str], tuple[float | None, float | None]],
    weights: ScorerWeights,
    *,
    recoverable: int,
    partial: bool,
) -> RoundSummary:
    """The round's rows written to ``path`` with its summary; the matrix is built a chunk at a
    time and kept as float32, the weighted columns as float64 for the baseline score."""
    targets = found.targets
    total = sum(len(entry.sources) for entry in targets)
    weighted = [feature.column for feature in weights.features]
    # Column-major, so each column is one contiguous array for Arrow.
    values = np.empty((total, len(FEATURE_COLUMNS)), dtype=np.float32, order="F")
    scored = np.empty((total, len(weighted)), dtype=np.float64)
    row = 0
    for chunk in feature_chunks(targets, pages, chunk_pairs=CHUNK_PAIRS, placements=placed):
        frame = to_frame(chunk)
        end = row + len(frame)
        values[row:end] = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        scored[row:end] = frame.loc[:, weighted].to_numpy(dtype=np.float64)
        row = end
    sources = [source for entry in targets for source in entry.sources]
    target_urls = [entry.target_url for entry in targets for _ in entry.sources]
    labels = np.fromiter(
        (pair in hidden for pair in zip(sources, target_urls, strict=True)),
        dtype=np.int8,
        count=total,
    )
    baseline = score_frame(
        pandas.DataFrame(
            {
                "source_url": sources,
                "target_url": target_urls,
                **{column: scored[:, i] for i, column in enumerate(weighted)},
            }
        ),
        weights,
    )["score"].to_numpy(dtype=np.float64)
    del scored
    position = {column: i for i, column in enumerate(FEATURE_COLUMNS)}
    positive_share, negative_share = placement_shares(
        pandas.DataFrame(
            {
                **{column: values[:, position[column]] for column in PLACEMENT_COLUMNS},
                LABEL_COLUMN: labels,
            }
        )
    )
    summary = RoundSummary(
        round=round_,
        hidden=len(hidden),
        recoverable=recoverable,
        pairs=total,
        positives=int(labels.sum()),
        groups_with_positive=len(
            {source for source, label in zip(sources, labels.tolist(), strict=True) if label}
        ),
        positive_placement_share=positive_share,
        negative_placement_share=negative_share,
    )
    table = pa.Table.from_arrays(
        [
            pa.array(np.full(total, round_, dtype=np.int16)),
            pa.array(sources, type=pa.string()),
            pa.array(target_urls, type=pa.string()),
            pa.array(labels),
            pa.array(baseline),
            *(pa.array(values[:, i], from_pandas=True) for i in range(len(FEATURE_COLUMNS))),
        ],
        schema=ROUND_SCHEMA.with_metadata(
            {
                "tenant_id": tenant_id,
                "cache_key": key,
                "round": str(round_),
                "partial": "true" if partial else "false",
                "summary": summary.model_dump_json(),
                "feature_columns": json.dumps(FEATURE_COLUMNS),
            }
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(table, path)
    return summary
