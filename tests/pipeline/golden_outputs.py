"""Golden outputs of the semantic rung and of anchor selection (#91), and their comparison.

The expected files under ``golden/`` were captured from the implementation before #91's memory
rework. Discrete outputs must stay identical; floats may drift by float32 rounding only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from quality_seed import DIMENSION as QUALITY_DIMENSION
from quality_seed import keyword_vector, seed_quality
from selection_seed import seed_selection
from selection_seed import voyage as hashed_voyage
from structlog.testing import capture_logs
from test_semantic_anchors import (
    ALL_PAIRS,
    CONCEPTS,
    GUARDED_BODIES,
    GUARDED_KEYWORDS,
    INBOUND,
    KEYWORDS,
    MODEL,
    PAIRS,
    TENANT,
    FakeGraph,
    anchor_vectors,
    big_corpus,
    concept_vector,
    indexes,
    url,
)
from voyage_fakes import FakeVoyage, client

from linking_engine.anchor.extraction import SourceIndex, Stems
from linking_engine.models import ExtractionSettings, KeywordSource, PageStructure
from linking_engine.pipeline.anchor_selection import UNANCHORED_FILE, select_anchors
from linking_engine.pipeline.semantic_anchors import (
    AnchorVectors,
    SemanticRun,
    placement_features,
    semantic_rung,
)
from linking_engine.pipeline.text_vectors import text_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.embedding.voyage_client import VoyageClient
    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

GOLDEN = Path(__file__).resolve().parent / "golden"
RUNG_FILE = GOLDEN / "semantic_rung.json"
SELECTION_FILE = GOLDEN / "anchor_selection.json"
TOLERANCE = 1e-5
DIM = 64
# The big corpus's keyworded targets: one keyword each, as in the threshold test.
BIG_TARGETS = {
    "big0-0": ("trail shoes",),
    "big0-1": ("steep climbs",),
    "big1-0": ("winter tents",),
    "big1-1": ("warm shelter",),
}
# A second keyword, and a page of the tent topic whose keyword is closer than the trail pages'
# keywords to many trail phrases: a rival in one language, none when the topics' languages
# differ.
RIVAL_TARGETS = {
    **BIG_TARGETS,
    "big0-1": ("steep climbs", "wet rock grip"),
    "big1-2": ("rocky footwear boots",),
}
RIVAL_INBOUND = {"big0-0": ["footwear"], "big1-0": ["winter tents"]}
# Report fields that change from run to run.
VOLATILE = {"tenant_id", "seconds", "finished_at"}
# Per-text noise on the toy embeddings: texts naming the same concepts no longer share a vector,
# so no ordering the goldens pin rests on an exact tie that float32 products on another BLAS
# could break either way. Checked on the implementation the goldens came from: perturbing every
# vector by 1e-4 of itself changes no discrete output.
RUNG_JITTER, QUALITY_JITTER = 0.1, 0.05


def _noise(text: str, size: int, scale: float) -> np.ndarray:
    found: np.ndarray = scale * np.random.default_rng(int(text_key(text)[:16], 16)).normal(
        size=size
    )
    return found


def rung_voyage() -> FakeVoyage:
    """The rung tests' concept embedding, jittered on the dimensions no concept uses."""
    free = slice(len(CONCEPTS), DIM - 1)

    def respond(texts: Sequence[str]) -> list[list[float]]:
        found = []
        for text in texts:
            vector = np.asarray(concept_vector(text))
            vector[free] += _noise(text, DIM - 1 - len(CONCEPTS), RUNG_JITTER)
            found.append(vector.tolist())
        return found

    return FakeVoyage(dimension=DIM, respond=respond)


def quality_voyage() -> FakeVoyage:
    """The quality tenant's topic-centred embedding, jittered on every dimension."""

    def respond(texts: Sequence[str]) -> list[list[float]]:
        return [
            (
                np.asarray(keyword_vector(text)) + _noise(text, QUALITY_DIMENSION, QUALITY_JITTER)
            ).tolist()
            for text in texts
        ]

    return FakeVoyage(dimension=QUALITY_DIMENSION, respond=respond)


def load(path: Path) -> dict[str, Any]:
    found: dict[str, Any] = json.loads(path.read_text())
    return found


def _dumps(value: object, depth: int) -> str:
    if depth and isinstance(value, dict) and value:
        pad = " " * (4 - depth)
        items = (
            f"{pad} {json.dumps(key)}: {_dumps(value[key], depth - 1)}" for key in sorted(value)
        )
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if depth and isinstance(value, list) and value:
        pad = " " * (4 - depth)
        items = (f"{pad} {_dumps(item, depth - 1)}" for item in value)
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(value, sort_keys=True)


def save(path: Path, outputs: dict[str, Any]) -> None:
    """One line per match and per row: the files are data, read through ``differences``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(outputs, 3) + "\n")


def plain(value: object) -> Any:
    """``value`` as it reads back from JSON: string keys, lists for tuples, floats rounded to
    eight places so the files stay small."""

    def rounded(item: object) -> object:
        if isinstance(item, float) and math.isfinite(item):
            return round(item, 8)
        if isinstance(item, dict):
            return {key: rounded(inner) for key, inner in item.items()}
        if isinstance(item, list):
            return [rounded(inner) for inner in item]
        return item

    return rounded(json.loads(json.dumps(value, default=float)))


def differences(expected: object, actual: object, where: str = "") -> list[str]:
    """Where ``actual`` departs from ``expected``: floats beyond TOLERANCE, anything else
    unequal."""
    if isinstance(expected, float) or isinstance(actual, float):
        if (
            isinstance(expected, int | float)
            and isinstance(actual, int | float)
            and not isinstance(expected, bool)
            and not isinstance(actual, bool)
            and abs(expected - actual) <= TOLERANCE
        ):
            return []
        return [f"{where}: expected {expected!r}, got {actual!r}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = [f"{where}.{key}: missing" for key in sorted(expected.keys() - actual.keys())]
        found += [f"{where}.{key}: unexpected" for key in sorted(actual.keys() - expected.keys())]
        for key in sorted(expected.keys() & actual.keys()):
            found += differences(expected[key], actual[key], f"{where}.{key}")
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: expected {len(expected)} items, got {len(actual)}"]
        return [
            line
            for i, (left, right) in enumerate(zip(expected, actual, strict=True))
            for line in differences(left, right, f"{where}[{i}]")
        ]
    if type(expected) is type(actual) and expected == actual:
        return []
    return [f"{where}: expected {expected!r}, got {actual!r}"]


# ── the semantic rung ───────────────────────────────────────────────────────


def _pair(source: str, target: str) -> str:
    return f"{source} -> {target}"


async def _rung_outputs(
    vectors: AnchorVectors,
    graph: object,
    *,
    pairs: list[tuple[str, str]],
    all_pairs: list[tuple[str, str]],
    source_indexes: dict[str, SourceIndex],
    keywords: dict[str, list[tuple[int, str, KeywordSource]]],
    inbound: dict[str, list[str]],
    override: float | None = None,
) -> dict[str, Any]:
    with capture_logs() as logs:
        run: SemanticRun = await semantic_rung(
            graph,  # type: ignore[arg-type]
            vectors,
            TENANT,
            pairs=pairs,
            all_pairs=all_pairs,
            indexes=source_indexes,
            keywords=keywords,
            existing={},
            inbound=inbound,
            settings=ExtractionSettings(semantic_threshold=override),
        )
    [line] = [entry for entry in logs if entry["event"] == "anchors.semantic"]
    await vectors.ensure("sentences", [match.sentence for match in run.matches.values()])
    features = {}
    for (source, target), match in run.matches.items():
        context, fit = placement_features(vectors, match)
        features[_pair(source, target)] = {
            "context_relevance": context,
            "anchor_target_fit": fit,
            "phrase_page": vectors.phrase_page(match.phrase, target),
            "phrase_keyword": vectors.phrase_keyword(match.phrase, match.keyword),
        }
    return plain(
        {
            "matches": {
                _pair(source, target): match.model_dump(mode="json")
                for (source, target), match in run.matches.items()
            },
            "threshold": run.threshold.model_dump(mode="json"),
            "skipped_reason": run.skipped_reason,
            "zero_overlap": run.zero_overlap,
            "invocations": run.invocations,
            "rejected_identifier": run.rejected_identifier,
            "rejected_other_target": run.rejected_other_target,
            "log": {key: value for key, value in line.items() if key != "log_level"},
            "features": features,
        }
    )


class _BigGraph:
    """The big corpus's hubs; with ``languages``, each topic's pages in a language of its own,
    so the best-target check runs once per language."""

    def __init__(self, *, languages: bool) -> None:
        self._languages = languages

    async def page_structure(self, tenant_id: str) -> list[PageStructure]:
        _, hubs = big_corpus()
        return [
            PageStructure(
                url=url(name),
                inbound=0,
                outbound=0,
                hub_id=hub,
                language=("en", "de")[hub] if self._languages else None,
            )
            for name, hub in hubs.items()
        ]


def big_pages() -> dict[str, np.ndarray]:
    bodies, hubs = big_corpus()
    centre = {0: np.asarray(concept_vector("trail shoe")), 1: np.asarray(concept_vector("tents"))}
    rng = np.random.default_rng(2)
    return {
        url(name): (centre[hubs[name]] + 0.05 * rng.random(DIM)).astype(np.float32)
        for name in bodies
    }


def big_vectors(voyage_client: VoyageClient | None, cache_dir: Path) -> AnchorVectors:
    return AnchorVectors(
        voyage_client, TENANT, big_pages(), page_models=[MODEL], cache_dir=cache_dir
    )


async def big_outputs(
    vectors: AnchorVectors,
    *,
    targets: dict[str, tuple[str, ...]],
    languages: bool,
    inbound: dict[str, list[str]] | None = None,
    override: float | None = None,
) -> dict[str, Any]:
    """Every page of the big corpus paired with each keyworded target."""
    bodies, _ = big_corpus()
    stems = Stems("en")
    pairs = [(url(name), url(target)) for name in bodies for target in targets if name != target]
    return await _rung_outputs(
        vectors,
        _BigGraph(languages=languages),
        pairs=pairs,
        all_pairs=pairs,
        source_indexes={url(n): SourceIndex(url(n), body, (), stems) for n, body in bodies.items()},
        keywords={
            url(target): [
                (rank, text, KeywordSource.INFERRED) for rank, text in enumerate(texts, 1)
            ]
            for target, texts in targets.items()
        },
        inbound={url(target): texts for target, texts in (inbound or {}).items()},
        override=override,
    )


async def semantic_rung_outputs(cache_dir: Path) -> dict[str, Any]:
    """The rung on the planted pair, on the guarded pairs, on the big corpus with a derived
    threshold, and with a rival target in one and in two languages; each with a cold cache."""
    stems = Stems("en")
    guarded_pairs = [(url("a0"), url("a1")), (url("c0"), url("c1"))]
    return {
        "planted": await _rung_outputs(
            anchor_vectors(rung_voyage(), cache_dir / "planted"),
            FakeGraph(),
            pairs=PAIRS,
            all_pairs=ALL_PAIRS,
            source_indexes=indexes(),
            keywords=KEYWORDS,
            inbound=INBOUND,
        ),
        "guarded": await _rung_outputs(
            anchor_vectors(rung_voyage(), cache_dir / "guarded"),
            FakeGraph(),
            pairs=guarded_pairs,
            all_pairs=[*ALL_PAIRS, guarded_pairs[1]],
            source_indexes={
                **indexes(),
                url("c0"): SourceIndex(url("c0"), GUARDED_BODIES["c0"], (), stems),
            },
            keywords={**KEYWORDS, **GUARDED_KEYWORDS},
            inbound=INBOUND,
            override=0.6,
        ),
        "big": await big_outputs(
            big_vectors(client(rung_voyage()), cache_dir / "big"),
            targets=BIG_TARGETS,
            languages=False,
        ),
        **{
            name: await big_outputs(
                big_vectors(client(rung_voyage()), cache_dir / name),
                targets=RIVAL_TARGETS,
                languages=languages,
                inbound=RIVAL_INBOUND,
                override=0.5,
            )
            for name, languages in (("rivals", False), ("rivals-two-languages", True))
        },
    }


# ── anchor selection ────────────────────────────────────────────────────────


async def selection_outputs(
    graph: GraphRepo,
    mongo: MongoRepo,
    tenant: str,
    *,
    cache_dir: Path,
    voyage_client: VoyageClient | None,
) -> dict[str, Any]:
    """select_anchors' report, every choice row (chosen and alternatives), and every unanchored
    pair's reason and best score, rows sorted and as lists under their column names."""
    report, path = await select_anchors(
        graph, mongo, tenant, cache_dir=cache_dir, voyage=voyage_client
    )
    choices = pq.read_table(path)
    unanchored = pq.read_table(path.parent / UNANCHORED_FILE).to_pylist()
    by_reason: defaultdict[str, list[str]] = defaultdict(list)
    for row in unanchored:
        by_reason[row["reason"]].append(f"{row['source_url']}\t{row['target_url']}")
    return plain(
        {
            "report": report.model_dump(mode="json", exclude=VOLATILE),
            "choice_columns": choices.column_names,
            "choices": sorted(
                [list(row.values()) for row in choices.to_pylist()],
                key=lambda row: (row[0], row[1], row[2]),
            ),
            # Each reason's pairs as a count and a digest; the rows with a score in full.
            "unanchored": {
                reason: {
                    "pairs": len(pairs),
                    "sha256": hashlib.sha256("\n".join(sorted(pairs)).encode()).hexdigest(),
                }
                for reason, pairs in by_reason.items()
            },
            "unanchored_scored": sorted(
                [row["source_url"], row["target_url"], row["reason"], row["best_score"]]
                for row in unanchored
                if row["best_score"] is not None
            ),
        }
    )


async def planted_gate_outputs(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, cache_dir: Path
) -> dict[str, Any]:
    """The #23 planted gate: hashed vectors, so the lexical rungs choose and the placement
    features and the semantic score parts come from the vectors."""
    await seed_selection(graph, mongo, tenant, bare=True, textless=10)
    return await selection_outputs(
        graph, mongo, tenant, cache_dir=cache_dir, voyage_client=client(hashed_voyage())
    )


async def quality_outputs(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, cache_dir: Path
) -> dict[str, Any]:
    """The #74 quality tenant: topic-centred vectors, so the semantic rung matches, and refuses
    the phrases closer to another target of the topic."""
    await seed_quality(graph, mongo, tenant)
    return await selection_outputs(
        graph, mongo, tenant, cache_dir=cache_dir, voyage_client=client(quality_voyage())
    )
