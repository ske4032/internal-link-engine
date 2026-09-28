"""The semantic rung's memory (#91): one source paired with many targets holds its phrase
vectors once, not once per pair."""

from __future__ import annotations

import tracemalloc
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from test_semantic_anchors import MODEL, TENANT
from voyage_fakes import FakeVoyage, client

from linking_engine.anchor.extraction import SourceIndex, Stems
from linking_engine.anchor.semantic import TOP_SENTENCES, candidate_phrases
from linking_engine.models import ExtractionSettings, KeywordSource, PageStructure
from linking_engine.pipeline.semantic_anchors import AnchorVectors, SemanticRun, semantic_rung
from linking_engine.pipeline.text_vectors import text_key

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

SOURCE = "example.com/source"
TARGETS = 400
SENTENCES = 12
WIDE, NARROW = 1024, 16
FLOAT32 = 4
SYLLABLES = ("ka", "lo", "mi", "ne", "ru", "sa", "te", "vo")
# Two-syllable words for the copy, three-syllable ones for the keywords: no digits, so every
# phrase agrees with every keyword on identifiers.
COPY_WORDS = ["".join(pair) for pair in product(SYLLABLES, repeat=2)]
KEYWORD_WORDS = ["".join(triple) for triple in product(SYLLABLES, repeat=3)]


def body() -> str:
    return " ".join(
        " ".join(COPY_WORDS[(5 * s + w) % len(COPY_WORDS)] for w in range(6)).capitalize() + "."
        for s in range(SENTENCES)
    )


def target(i: int) -> str:
    return f"example.com/target-{i}"


def keywords() -> dict[str, list[tuple[int, str, KeywordSource]]]:
    return {
        target(i): [(1, f"{KEYWORD_WORDS[i]} {KEYWORD_WORDS[-1 - i]}", KeywordSource.INFERRED)]
        for i in range(TARGETS)
    }


def voyage(dimension: int) -> FakeVoyage:
    """Every sentence gets one vector, so each target's closest sentences are the first three
    at any dimension and the run has the same shape; phrases and keywords are drawn from their
    text's hash."""

    def respond(texts: Sequence[str]) -> list[list[float]]:
        found = []
        for text in texts:
            if text.endswith("."):
                found.append([1.0] * dimension)
            else:
                seed = int(text_key(text)[:16], 16)
                found.append(np.random.default_rng(seed).normal(size=dimension).tolist())
        return found

    return FakeVoyage(dimension=dimension, respond=respond, record_calls=False)


class Graph:
    async def page_structure(self, tenant_id: str) -> list[PageStructure]:
        return [
            PageStructure(url=url, inbound=0, outbound=0)
            for url in (SOURCE, *(target(i) for i in range(TARGETS)))
        ]


async def run(fake: FakeVoyage, cache_dir: Path) -> tuple[SemanticRun, AnchorVectors]:
    vectors = AnchorVectors(client(fake), TENANT, {}, page_models=[MODEL], cache_dir=cache_dir)
    pairs = [(SOURCE, target(i)) for i in range(TARGETS)]
    found = await semantic_rung(
        Graph(),  # type: ignore[arg-type]
        vectors,
        TENANT,
        pairs=pairs,
        all_pairs=pairs,
        indexes={SOURCE: SourceIndex(SOURCE, body(), (), Stems("en"))},
        keywords=keywords(),
        existing={},
        inbound={},
        # Every phrase reaches it, so every pair's phrases go to the best-target check.
        settings=ExtractionSettings(semantic_threshold=-1.0),
    )
    return found, vectors


async def peak(dimension: int, cache_dir: Path) -> tuple[int, SemanticRun, AnchorVectors]:
    """Traced peak of a warm-cache run over the memory in use before it, and the run."""
    await run(voyage(dimension), cache_dir)
    fake = voyage(dimension)
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        found, vectors = await run(fake, cache_dir)
        _, top = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert fake.call_count == 0, "the measured run embedded again"
    return top - before, found, vectors


async def test_semantic_rung_peak_memory_bounded(tmp_path: Path) -> None:
    wide, found, vectors = await peak(WIDE, tmp_path / "wide")
    narrow, _, _ = await peak(NARROW, tmp_path / "narrow")

    # The run did the work being measured: every pair had phrases that reached the check.
    assert found.invocations == TARGETS
    assert len(found.matches) + found.rejected_other_target == TARGETS
    index = SourceIndex(SOURCE, body(), (), Stems("en"))
    phrases = {
        phrase.text
        for position in range(TOP_SENTENCES)
        for phrase in candidate_phrases(index, position)
    }
    assert vectors.missing("phrases", phrases) == 0
    # Both runs have the same texts, pairs and python objects, so what grows with the
    # dimension is vector memory. A regression guard, not a limit: a float32 copy of the
    # phrases per pair would add TARGETS times the phrases, and the run stays well below it.
    per_pair = TARGETS * len(phrases) * FLOAT32 * (WIDE - NARROW)
    grown = wide - narrow
    assert grown < per_pair / 2, (
        f"peak grew {grown / 2**20:.1f} MiB with the dimension; per-pair phrase copies would "
        f"add {per_pair / 2**20:.1f} MiB"
    )
