"""The rung's best-target check at an exact tie (#91): the own keyword and another page's keyword
with one vector tie on every BLAS, so the pair matches; a rival a millionth closer still refuses
it, as the pure guard does."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from voyage_fakes import MODEL, FakeVoyage, client

from linking_engine.anchor.extraction import SourceIndex, Stems
from linking_engine.models import ExtractionSettings, KeywordSource, PageStructure
from linking_engine.pipeline.semantic_anchors import AnchorVectors, SemanticRun, semantic_rung

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

TENANT = "test-ties"
DIM = 64
SOURCE, TARGET = "example.com/source", "example.com/target"
# One sentence with one candidate phrase, so the pair's outcome is that phrase's check.
BODY = "Sturdy boots."
PHRASE = "Sturdy boots"
OWN, RIVAL = "trail shoes", "hiking footwear"
# Other pages' keywords, sorted between the rival's and the own keyword's, so the two sit in
# distant columns of one float32 product, where identical columns can differ by an ulp either
# way depending on the BLAS; several draws make one of them land against the tie.
FILLERS = [f"meadow {word}" for word in ("kayak", "lantern", "compass", "hammock", "stove")]
SEEDS = range(8)


@dataclass(frozen=True)
class Scene:
    keyword: np.ndarray
    # At cosine 0.8 from the keyword: every sentence and phrase.
    phrase: np.ndarray
    # Tilted towards the phrase so its cosine to it is 1e-6 above the keyword's.
    closer: np.ndarray
    others: dict[str, np.ndarray]


def scene(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    keyword = rng.normal(size=DIM)
    keyword /= np.linalg.norm(keyword)
    side = rng.normal(size=DIM)
    side -= (side @ keyword) * keyword
    side /= np.linalg.norm(side)
    return Scene(
        keyword,
        0.8 * keyword + 0.6 * side,
        keyword + 1e-6 / 0.6 * side,
        {text: rng.normal(size=DIM) for text in FILLERS},
    )


def page(text: str) -> str:
    return f"example.com/{text.replace(' ', '-')}"


class Graph:
    async def page_structure(self, tenant_id: str) -> list[PageStructure]:
        urls = (SOURCE, TARGET, page(RIVAL), *map(page, FILLERS))
        return [PageStructure(url=url, inbound=0, outbound=0) for url in urls]


async def run(
    found: Scene, rival: np.ndarray, cache_dir: Path
) -> tuple[SemanticRun, AnchorVectors]:
    table = {OWN: found.keyword, RIVAL: rival, **found.others}

    def respond(texts: Sequence[str]) -> list[list[float]]:
        return [table.get(text, found.phrase).tolist() for text in texts]

    vectors = AnchorVectors(
        client(FakeVoyage(dimension=DIM, respond=respond)),
        TENANT,
        {},
        page_models=[MODEL],
        cache_dir=cache_dir,
    )
    outcome = await semantic_rung(
        Graph(),  # type: ignore[arg-type]
        vectors,
        TENANT,
        pairs=[(SOURCE, TARGET)],
        all_pairs=[(SOURCE, TARGET)],
        indexes={SOURCE: SourceIndex(SOURCE, BODY, (), Stems("en"))},
        keywords={
            TARGET: [(1, OWN, KeywordSource.INFERRED)],
            **{page(text): [(1, text, KeywordSource.INFERRED)] for text in (RIVAL, *FILLERS)},
        },
        existing={},
        inbound={},
        settings=ExtractionSettings(semantic_threshold=0.5),
    )
    return outcome, vectors


def margin(vectors: AnchorVectors) -> float:
    """The phrase's float64 cosine to the rival's keyword less its cosine to the own keyword."""
    [row] = vectors.exact_rows("phrases", [PHRASE])
    rival, own = vectors.exact_rows("keywords", [RIVAL, OWN])
    return float(row @ rival - row @ own)


@pytest.mark.parametrize("seed", SEEDS)
async def test_an_own_keyword_and_a_rival_with_one_vector_tie_and_the_pair_matches(
    tmp_path: Path, seed: int
) -> None:
    drawn = scene(seed)

    found, vectors = await run(drawn, drawn.keyword, tmp_path)

    assert margin(vectors) == 0.0
    assert (found.rejected_identifier, found.rejected_other_target) == (0, 0), "tie refused"
    [match] = found.matches.values()
    assert (match.target_url, match.keyword, match.phrase) == (TARGET, OWN, PHRASE)
    assert match.semantic_similarity == pytest.approx(0.8, abs=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
async def test_a_rival_a_millionth_closer_refuses_the_pair(tmp_path: Path, seed: int) -> None:
    drawn = scene(seed)

    found, vectors = await run(drawn, drawn.closer, tmp_path)

    assert 5e-7 < margin(vectors) < 2e-6
    assert found.matches == {}
    assert (found.rejected_identifier, found.rejected_other_target) == (0, 1)
