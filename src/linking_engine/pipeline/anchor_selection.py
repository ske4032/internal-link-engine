"""Choose every pair's anchor: the ladder's lexical rungs, the semantic rung for the pairs they
miss, then scoring and choice, with the placement features of each choice. Writes the choices
and the pairs left without an anchor, with advice, as Parquet per tenant; read-only against both
stores. `compute_anchor_choices` chooses without writing, also on a held-out view."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from linking_engine.anchor.extraction import Stems
from linking_engine.anchor.generic import generic_overrides, is_generic
from linking_engine.anchor.keywords import tokens as words_of
from linking_engine.anchor.scoring import (
    STAGE,
    Brand,
    Candidate,
    ExistingAnchor,
    PairCandidates,
    choose,
    existing_type,
    selection_report,
    stem_jaccard,
    word_count,
)
from linking_engine.models import AnchorTypeProfile
from linking_engine.models.anchors import UNANCHORED_ADVICE
from linking_engine.pipeline.anchors import cache_folder, lexical_run
from linking_engine.pipeline.features import ANCHOR_CHOICES_FILE
from linking_engine.pipeline.semantic_anchors import (
    AnchorVectors,
    placement_features,
    semantic_rung,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo
    from linking_engine.models import (
        AnchorChoice,
        AnchorMatch,
        AnchorSelectionReport,
        KeywordSource,
        UnanchoredPair,
    )
    from linking_engine.pipeline.anchors import AnchorView

log = structlog.get_logger(__name__)

UNANCHORED_FILE: Final = "unanchored_pairs.parquet"
_MATCH_FIELDS: Final = (
    "keyword",
    "keyword_rank",
    "keyword_source",
    "rung",
    "phrase",
    "start",
    "end",
    "sentence",
    "sentence_index",
    "sentence_start",
    "stem_jaccard",
    "semantic_similarity",
)
_SCORE_FIELDS: Final = (
    "semantic",
    "keyword",
    "diversity",
    "length",
    "rank_weight",
    "profile_bonus",
    "total",
)
CHOICES_SCHEMA: Final = pa.schema(
    [
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("target_url", pa.string(), nullable=False),
        pa.field("rank", pa.int64(), nullable=False),
        pa.field("anchor_type", pa.string(), nullable=False),
        pa.field("keyword", pa.string(), nullable=False),
        pa.field("keyword_rank", pa.int64(), nullable=False),
        pa.field("keyword_source", pa.string(), nullable=False),
        pa.field("rung", pa.string(), nullable=False),
        pa.field("phrase", pa.string(), nullable=False),
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        pa.field("sentence", pa.string(), nullable=False),
        pa.field("sentence_index", pa.int64(), nullable=False),
        pa.field("sentence_start", pa.int64(), nullable=False),
        pa.field("stem_jaccard", pa.float64()),
        pa.field("semantic_similarity", pa.float64()),
        pa.field("score_semantic", pa.float64()),
        pa.field("score_keyword", pa.float64(), nullable=False),
        pa.field("score_diversity", pa.float64(), nullable=False),
        pa.field("score_length", pa.float64(), nullable=False),
        pa.field("score_rank_weight", pa.float64(), nullable=False),
        pa.field("score_profile_bonus", pa.float64(), nullable=False),
        pa.field("score_total", pa.float64(), nullable=False),
        pa.field("context_relevance", pa.float64()),
        pa.field("anchor_target_fit", pa.float64()),
    ]
)
UNANCHORED_SCHEMA: Final = pa.schema(
    [
        pa.field("source_url", pa.string(), nullable=False),
        pa.field("target_url", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("advice", pa.string(), nullable=False),
        pa.field("best_score", pa.float64()),
    ]
)


@dataclass(frozen=True, slots=True)
class AnchorSelection:
    """Every chosen anchor and alternative, the pairs left without one, and the report."""

    choices: tuple[AnchorChoice, ...]
    unanchored: tuple[UnanchoredPair, ...]
    report: AnchorSelectionReport
    # Why Voyage stopped being used by the end of the run, the placement vectors included;
    # None when it served every text.
    skipped_reason: str | None = None


async def select_anchors(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    cache_dir: Path,
    voyage: VoyageClient | None,
) -> tuple[AnchorSelectionReport, Path]:
    """Every pair's anchor, alternatives and placement features at
    ``<cache_dir>/<tenant>/anchor_choices.parquet``, the pairs left without one beside it in
    ``unanchored_pairs.parquet`` with their reason and advice, and the report.
    Without ``voyage`` (no key), or after an outage, the semantic rung is skipped and the
    lexical rungs still choose, scored without the parts that need vectors."""
    folder = cache_folder(cache_dir, tenant_id)
    selection = await compute_anchor_choices(
        graph, mongo, tenant_id, cache_dir=cache_dir, voyage=voyage
    )
    report = selection.report
    path = await asyncio.to_thread(_write, folder, selection.choices, selection.unanchored)
    log.info(
        "anchors.selected",
        stage=STAGE,
        tenant_id=tenant_id,
        pairs=report.pairs,
        lexical_pairs=report.lexical_pairs,
        semantic_invocations=report.semantic_invocations,
        semantic_matched=report.semantic_matched,
        zero_overlap_matches=report.zero_overlap_matches,
        semantic_rejected_identifier=report.semantic_rejected_identifier,
        semantic_rejected_other_target=report.semantic_rejected_other_target,
        threshold=(None if report.semantic_skipped_reason is not None else report.threshold.value),
        threshold_overridden=report.threshold.overridden,
        semantic_skipped=report.semantic_skipped_reason is not None,
        embedding_skipped=report.embedding_skipped_reason is not None,
        chosen=report.chosen,
        alternatives=report.alternatives,
        unanchored={reason.value: n for reason, n in report.unanchored.items()},
        chosen_types={kind.value: n for kind, n in report.chosen_types.items()},
        targets=report.targets,
        targets_with_anchor=report.targets_with_anchor,
        features_filled=report.features_filled,
        seconds=report.seconds,
    )
    return report, path


async def compute_anchor_choices(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    cache_dir: Path,
    voyage: VoyageClient | None,
    view: AnchorView | None = None,
) -> AnchorSelection:
    """Every pair's anchor, alternatives and placement features, the pairs left without one,
    and the report; nothing is written but the tenant's append-only vector caches. On a
    held-out ``view``: its candidates, no bridges, and the hidden links' anchors neither in
    their sources' existing spans nor among their targets' inbound anchors."""
    started = time.perf_counter()
    cache_folder(cache_dir, tenant_id)
    run = await lexical_run(graph, mongo, tenant_id, cache_dir=cache_dir, view=view)
    profile = await mongo.get_anchor_type_profile(tenant_id) or AnchorTypeProfile()
    rules = await mongo.get_anchor_rules(tenant_id)
    generic_add, generic_remove = generic_overrides(rules.generic_add, rules.generic_remove)
    brand = run.brand
    languages = await graph.page_languages(tenant_id)
    inbound = await _inbound_anchors(
        graph,
        tenant_id,
        {pair.target_url for pair in run.pairs},
        generic_add=generic_add,
        generic_remove=generic_remove,
        hidden=frozenset() if view is None else view.hidden,
    )

    vectors = await AnchorVectors.load(graph, voyage, tenant_id, cache_dir=cache_dir)
    missed = [
        (pair.source_url, pair.target_url)
        for pair in run.pairs
        if run.keywords.get(pair.target_url)
        and pair.source_url in run.indexes
        and (pair.source_url, pair.target_url) not in run.matches
    ]
    semantic = await semantic_rung(
        graph,
        vectors,
        tenant_id,
        pairs=missed,
        all_pairs=[(pair.source_url, pair.target_url) for pair in run.pairs],
        indexes=run.indexes,
        keywords=run.keywords,
        existing=run.existing,
        inbound={target: [text for text, _ in anchors] for target, anchors in inbound.items()},
        settings=run.settings,
    )
    sent = set(missed)
    found: dict[tuple[str, str], list[AnchorMatch]] = {
        key: list(matches) for key, matches in run.matches.items()
    }
    for key, match in semantic.matches.items():
        found.setdefault(key, []).append(match)
    every = [match for matches in found.values() for match in matches]
    await vectors.ensure("phrases", [match.phrase for match in every])
    await vectors.ensure("keywords", [match.keyword for match in every])

    stems = dict(run.stems)
    existing = await asyncio.to_thread(_existing, inbound, languages, run.keywords, stems, brand)
    pairs = [
        PairCandidates(
            pair.source_url,
            pair.target_url,
            tuple(
                _candidate(match, run.indexes[pair.source_url].stems, vectors)
                for match in found.get((pair.source_url, pair.target_url), ())
            ),
            has_keywords=bool(run.keywords.get(pair.target_url)),
            has_text=pair.source_url in run.indexes,
            meaning_searched=semantic.skipped_reason is None
            or (pair.source_url, pair.target_url) not in sent,
        )
        for pair in run.pairs
    ]
    chosen, unanchored = await asyncio.to_thread(
        choose, pairs, existing=existing, profile=profile, brand=brand
    )
    await vectors.ensure("sentences", [choice.match.sentence for choice in chosen])
    choices = tuple(_placed(choice, vectors) for choice in chosen)

    sentences_embedded, sentences_cached = vectors.counts("sentences")
    phrases_embedded, phrases_cached = vectors.counts("phrases")
    report = selection_report(
        tenant_id,
        choices,
        unanchored,
        pairs=len(run.pairs),
        lexical_pairs=len(run.matches),
        semantic_invocations=semantic.invocations,
        semantic_similarities=[
            match.semantic_similarity
            for match in semantic.matches.values()
            if match.semantic_similarity is not None
        ],
        zero_overlap_matches=semantic.zero_overlap,
        semantic_rejected_identifier=semantic.rejected_identifier,
        semantic_rejected_other_target=semantic.rejected_other_target,
        threshold=semantic.threshold,
        semantic_skipped_reason=semantic.skipped_reason,
        embedding_skipped_reason=vectors.embedding_skipped_reason(),
        sentences_embedded=sentences_embedded,
        sentences_cached=sentences_cached,
        phrases_embedded=phrases_embedded,
        phrases_cached=phrases_cached,
        profile=profile,
        targets=len({pair.target_url for pair in run.pairs}),
        started=started,
    )
    return AnchorSelection(choices, tuple(unanchored), report, vectors.skipped_reason())


async def _inbound_anchors(
    graph: GraphRepo,
    tenant_id: str,
    targets: set[str],
    *,
    generic_add: frozenset[str],
    generic_remove: frozenset[str],
    hidden: Set[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
    """(anchor text, source url) of every descriptive existing link into each target: links
    between crawled pages, generic anchors and ``hidden`` links left out."""
    target_of = {
        (link.source_url, link.position): link.target_url
        for link in await graph.link_relevance(tenant_id)
        if link.target_url in targets
        and not link.anchor_generic
        and (link.source_url, link.target_url) not in hidden
    }
    found: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    if not target_of:
        return {}
    async for texts in graph.iter_link_texts(tenant_id):
        for text in texts:
            target = target_of.get((text.source_url, text.position))
            if (
                target is None
                or not text.anchor_text.strip()
                or text.anchor_generic
                or is_generic(text.anchor_text, add=generic_add, remove=generic_remove)
            ):
                continue
            found[target].append((text.anchor_text, text.source_url))
    return dict(found)


def _existing(
    inbound: Mapping[str, Sequence[tuple[str, str]]],
    languages: Mapping[str, str | None],
    keywords: Mapping[str, Sequence[tuple[int, str, KeywordSource]]],
    stems: dict[str | None, Stems],
    brand: Brand,
) -> dict[str, list[ExistingAnchor]]:
    """Every target's descriptive existing anchors as word sets with their types, each stemmed
    in the language of the page it links from."""
    typed: dict[str, list[ExistingAnchor]] = {}
    for target, anchors in inbound.items():
        texts = [text for _, text, _ in keywords.get(target, ())]
        typed[target] = []
        for text, source in anchors:
            language = languages.get(source)
            if language not in stems:
                stems[language] = Stems(language)
            typed[target].append(
                ExistingAnchor(
                    words=words_of(text),
                    anchor_type=existing_type(text, texts, stems[language], brand),
                )
            )
    return typed


def _candidate(match: AnchorMatch, stems: Stems, vectors: AnchorVectors) -> Candidate:
    return Candidate(
        match=match,
        stem_jaccard=stem_jaccard(match.phrase, match.keyword, stems),
        words=words_of(match.phrase),
        word_count=word_count(match.phrase),
        target_cosine=vectors.phrase_page(match.phrase, match.target_url),
        keyword_cosine=vectors.phrase_keyword(match.phrase, match.keyword),
    )


def _placed(choice: AnchorChoice, vectors: AnchorVectors) -> AnchorChoice:
    context_relevance, anchor_target_fit = placement_features(vectors, choice.match)
    return choice.model_copy(
        update={"context_relevance": context_relevance, "anchor_target_fit": anchor_target_fit}
    )


def _unanchored_row(pair: UnanchoredPair) -> dict[str, object]:
    return {
        "source_url": pair.source_url,
        "target_url": pair.target_url,
        "reason": pair.reason.value,
        "advice": UNANCHORED_ADVICE[pair.reason],
        "best_score": pair.best_score,
    }


def _choice_row(choice: AnchorChoice) -> dict[str, object]:
    match = choice.match.model_dump(mode="json")
    score = choice.score.model_dump(mode="json")
    return {
        "source_url": choice.match.source_url,
        "target_url": choice.match.target_url,
        "rank": choice.rank,
        "anchor_type": choice.anchor_type.value,
        **{name: match[name] for name in _MATCH_FIELDS},
        **{f"score_{name}": score[name] for name in _SCORE_FIELDS},
        "context_relevance": choice.context_relevance,
        "anchor_target_fit": choice.anchor_target_fit,
    }


def _write(
    folder: Path, choices: Iterable[AnchorChoice], unanchored: Iterable[UnanchoredPair]
) -> Path:
    """Both files are written beside their paths first and renamed only when both are complete."""
    folder.mkdir(parents=True, exist_ok=True)
    tables = [
        (
            pa.Table.from_pylist(
                [_choice_row(choice) for choice in choices], schema=CHOICES_SCHEMA
            ),
            folder / ANCHOR_CHOICES_FILE,
        ),
        (
            pa.Table.from_pylist(
                [_unanchored_row(pair) for pair in unanchored], schema=UNANCHORED_SCHEMA
            ),
            folder / UNANCHORED_FILE,
        ),
    ]
    temps: list[Path] = []
    try:
        for table, path in tables:
            handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
            os.close(handle)
            temps.append(Path(name))
            pq.write_table(table, temps[-1])
        for temp, (_, path) in zip(temps, tables, strict=True):
            temp.replace(path)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)
    return folder / ANCHOR_CHOICES_FILE
