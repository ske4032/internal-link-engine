"""Link-embedding acceptance (#6): dedupe, generic skip, round trip, resume, model guard, failures."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
from neo4j import AsyncGraphDatabase
from structlog.testing import capture_logs
from test_embed import TamperingVoyage, rebuilt
from voyage_fakes import API_TOKEN_DRIFT, MODEL, FakeVoyage, client, words
from voyageai.error import InvalidRequestError

from linking_engine.errors import (
    DatabaseWriteError,
    EmbeddingModelMismatchError,
    EmbeddingRequestError,
    EmbeddingResponseError,
    SchemaError,
)
from linking_engine.graph.repo import ANCHOR_KEY_MAX_BYTES, VECTOR_DIMENSIONS
from linking_engine.models import AnchorRules, EdgeRef, Link, LinkEmbedReport, Page, SentenceTarget
from linking_engine.pipeline.embed import FLUSH_SIZE
from linking_engine.pipeline.embed_links import embed_links, normalise_sentence, sentence_hash

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    import numpy.typing as npt

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.models import EmbeddingBatch

DIM = VECTOR_DIMENSIONS
BASE = "example.com"
OTHER_MODEL = "voyage-3-large"


def url(path: str) -> str:
    return f"{BASE}{path}"


def text_vector(text: str) -> list[float]:
    """A raw vector seeded by the text: equal texts get equal vectors, different texts differ."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    return list(np.random.default_rng(seed).standard_normal(DIM))


def unit(text: str) -> npt.NDArray[np.float32]:
    """What the client makes of text_vector(text): unit length, then float32 in the flush buffer."""
    vector = np.asarray(text_vector(text), dtype=np.float64)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def fake_voyage(**options: object) -> FakeVoyage:
    return FakeVoyage(
        dimension=DIM,
        respond=lambda texts: [text_vector(text) for text in texts],
        **options,  # type: ignore[arg-type]
    )


def sent_texts(fake: FakeVoyage) -> list[str]:
    return [text for call in fake.calls for text in call.texts]


def sdk_tokens(fake: FakeVoyage) -> int:
    return sum(words(text) for text in sent_texts(fake)) + API_TOKEN_DRIFT * len(fake.calls)


def events(logs: Sequence[Mapping[str, object]], name: str) -> list[Mapping[str, object]]:
    return [entry for entry in logs if entry.get("event") == name]


# ── a crafted tenant with every case, and its expected outcome ──────────────

# (source, position, target, anchor text, surrounding text)
Spec = tuple[str, int, str, str, str]
S1 = "Our trail shoes grip wet rock."
CRAFTED: tuple[Spec, ...] = (
    ("/a", 0, "/t1", "Trail Shoes", S1),
    ("/a", 1, "/t2", "trail shoes ", "Compare every trail shoe we sell."),
    ("/a", 2, "/t3", "trail  shoes.", S1),
    ("/a", 3, "/t1", "Click Here", "  Our trail   shoes grip\nwet rock. "),
    ("/a", 4, "/t2", "click here ", "Click here for sizing."),
    ("/a", 5, "/t3", "Read more →", "Read more → about lacing."),
    ("/b", 0, "/t1", "C#", "Our C# SDK."),
    ("/b", 1, "/t2", "C", "Our C SDK."),
    ("/b", 2, "/t3", "", "An image link."),
    ("/b", 3, "/t4", "!!!", "   "),
    ("/b", 4, "/t4", "Hiking Boots", ""),
    # Differs from S1 only in case; sentences are not casefolded.
    ("/b", 5, "/t1", "hiking boots", "our trail shoes grip wet rock."),
)
EDGES = 12
KEYS = {
    ("/a", 0): "trail shoes",
    ("/a", 1): "trail shoes",
    ("/a", 2): "trail shoes",
    ("/a", 3): "click here",
    ("/a", 4): "click here",
    ("/a", 5): "read more",
    ("/b", 0): "c#",
    ("/b", 1): "c",
    ("/b", 2): None,
    ("/b", 3): None,
    ("/b", 4): "hiking boots",
    ("/b", 5): "hiking boots",
}
GENERIC_KEYS = {"click here", "read more"}
EMBEDDED_ANCHORS = {"trail shoes", "c#", "c", "hiking boots"}
EMPTY_ANCHOR_EDGES = {("/b", 2), ("/b", 3)}
BLANK_SENTENCE_EDGES = {("/b", 3), ("/b", 4)}
SENTENCES = {normalise_sentence(spec[4]) for spec in CRAFTED} - {""}
# 12 edges, 2 empty anchors, 6 distinct keys; 2 blank sentences, 8 distinct sentences.
ANCHOR_RATIO = (12 - 2) / 6
SENTENCE_RATIO = (12 - 2) / 8
FLUSH = 3


def sentence_of(source: str, position: int, spec: Sequence[Spec] = CRAFTED) -> str:
    return normalise_sentence(next(s[4] for s in spec if (s[0], s[1]) == (source, position)))


async def seed(graph: GraphRepo, tenant: str, spec: Sequence[Spec] = CRAFTED) -> None:
    """Crawled pages for every endpoint; `spec` becomes the complete link set of its sources."""
    paths = sorted({s[0] for s in spec} | {s[2] for s in spec})
    await graph.upsert_pages(tenant, [Page(url=url(p), status_code=200) for p in paths])
    links = [
        Link(
            source_url=url(source),
            target_url=url(target),
            position=position,
            anchor_text=anchor,
            surrounding_text=sentence,
        )
        for source, position, target, anchor, sentence in spec
    ]
    sources = sorted({s[0] for s in spec})
    await graph.replace_links(tenant, [url(s) for s in sources], links)


def changed(spec: Sequence[Spec], edge: tuple[str, int], **fields: str) -> tuple[Spec, ...]:
    out = []
    for source, position, target, anchor, sentence in spec:
        if (source, position) == edge:
            anchor = fields.get("anchor", anchor)
            sentence = fields.get("sentence", sentence)
        out.append((source, position, target, anchor, sentence))
    return tuple(out)


@dataclass(frozen=True)
class Edge:
    key: str | None
    generic: bool | None
    has_key_property: bool
    vector: tuple[float, ...] | None
    digest: str | None
    model: str | None


async def stored_edges(graph: GraphRepo, tenant: str) -> dict[tuple[str, int], Edge]:
    rows = await graph._auto(
        "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "RETURN s.url AS source, r.position AS position, r.anchorKey AS key, "
        "r.anchorGeneric AS generic, 'anchorKey' IN keys(r) AS has_key, "
        "r.surroundingEmbedding AS vec, r.surroundingEmbeddedHash AS hash, "
        "r.surroundingEmbeddingModel AS model",
        t=tenant,
    )
    return {
        (str(row["source"]).removeprefix(BASE), row["position"]): Edge(  # type: ignore[misc]
            key=row["key"],  # type: ignore[arg-type]
            generic=row["generic"],  # type: ignore[arg-type]
            has_key_property=bool(row["has_key"]),
            vector=None if row["vec"] is None else tuple(row["vec"]),  # type: ignore[arg-type]
            digest=row["hash"],  # type: ignore[arg-type]
            model=row["model"],  # type: ignore[arg-type]
        )
        for row in rows
    }


async def stored_anchors(graph: GraphRepo, tenant: str) -> dict[str, tuple[object, ...]]:
    rows = await graph._auto(
        "MATCH (a:Anchor {tenantId: $t}) RETURN a.text AS text, a.embedding AS vec, "
        "a.embeddingModel AS model, a.embeddingDimensions AS dims, a.embeddedAt AS at",
        t=tenant,
    )
    return {
        str(row["text"]): (tuple(row["vec"]), row["model"], row["dims"], row["at"])  # type: ignore[arg-type]
        for row in rows
    }


def assert_close(stored: Sequence[float] | None, expected: npt.NDArray[np.float32]) -> None:
    assert stored is not None, "no vector stored"
    assert len(stored) == DIM
    np.testing.assert_allclose(np.asarray(stored, dtype=np.float32), expected, rtol=0, atol=1e-6)


async def run(
    graph: GraphRepo, tenant: str, fake: FakeVoyage | None = None, flush_size: int = FLUSH
) -> tuple[LinkEmbedReport, FakeVoyage, list[Mapping[str, object]]]:
    fake = fake or fake_voyage()
    with capture_logs() as logs:
        report = await embed_links(graph, client(fake), tenant, flush_size=flush_size)
    return report, fake, logs


# ── AC1: deduplication ratio reported, logged and correct ──────────────────


@pytest.mark.integration
async def test_ac1_dedupe_ratios_are_reported_and_logged_for_a_crafted_tenant(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    report, fake, logs = await run(graph, tenant)

    sent = sent_texts(fake)
    assert report.model_dump(exclude={"elapsed_s", "finished_at"}) == {
        "tenant_id": tenant,
        "embedding_model": MODEL,
        "dimensions": DIM,
        "edges": EDGES,
        "keys_written": EDGES,
        "empty_anchors": 2,
        "unique_anchors": 6,
        "anchor_dedupe_ratio": pytest.approx(ANCHOR_RATIO),
        "generic_anchors": 2,
        "generic_edges": 3,
        "anchors_cached": 0,
        "anchors_embedded": 4,
        "empty_sentences": 2,
        # The blank-sentence edges never had a vector, so there is nothing to clear.
        "surrounding_cleared": 0,
        "unique_sentences": 8,
        "sentence_dedupe_ratio": pytest.approx(SENTENCE_RATIO),
        "sentences_cached": 0,
        "sentences_reused": 0,
        "sentences_embedded": 8,
        "surrounding_edges_written": 10,
        "anchor_flushes": 2,
        "sentence_flushes": 3,
        "api_tokens": sdk_tokens(fake),
        "tokens": sum(words(text) for text in sent),
        "truncated": 0,
    }
    (done,) = events(logs, "embedding.links.done")
    assert done["anchor_dedupe_ratio"] == pytest.approx(ANCHOR_RATIO), (
        f"expected (12 edges - 2 empty) / 6 keys = {ANCHOR_RATIO:.4f} in embedding.links.done"
    )
    assert done["sentence_dedupe_ratio"] == pytest.approx(SENTENCE_RATIO)
    assert (done["edges"], done["unique_anchors"], done["anchors_embedded"]) == (12, 6, 4)
    assert (done["tenant_id"], done["stage"], done["log_level"]) == (
        tenant,
        "embedding.links",
        "info",
    )


# ── AC2: one Anchor per normalised text, joined by every edge ──────────────


@pytest.mark.integration
async def test_ac2_anchor_variants_share_one_anchor_and_every_edge_joins_the_identical_vector(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    _, fake, _ = await run(graph, tenant)

    anchors = await stored_anchors(graph, tenant)
    assert set(anchors) == EMBEDDED_ANCHORS
    for key, (vector, model, dims, _) in anchors.items():
        assert_close(vector, unit(key))  # type: ignore[arg-type]
        assert (model, dims) == (MODEL, DIM)
    rows = await graph._auto(
        "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "MATCH (a:Anchor {tenantId: s.tenantId, text: r.anchorKey}) "
        "RETURN s.url AS source, r.position AS position, a.embedding AS vec",
        t=tenant,
    )
    joined = {(str(r["source"]).removeprefix(BASE), r["position"]): r["vec"] for r in rows}
    expected = {edge for edge, key in KEYS.items() if key in EMBEDDED_ANCHORS}
    assert set(joined) == expected
    variants = [joined[("/a", position)] for position in (0, 1, 2)]
    assert variants[0] == variants[1] == variants[2], "variants joined different vectors"
    assert_close(variants[0], unit("trail shoes"))  # type: ignore[arg-type]
    assert sent_texts(fake).count("trail shoes") == 1


@pytest.mark.integration
async def test_gotcha_normalise_before_dedupe(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant)
    _, fake, _ = await run(graph, tenant)

    edges = await stored_edges(graph, tenant)
    assert {edge: stored.key for edge, stored in edges.items()} == KEYS
    anchors_sent = [text for text in sent_texts(fake) if text not in SENTENCES]
    assert sorted(anchors_sent) == sorted(EMBEDDED_ANCHORS), (
        "each normalised anchor must be sent once, never a raw variant"
    )


@pytest.mark.integration
async def test_gotcha_c_sharp_and_c_stay_distinct_keys(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)

    edges = await stored_edges(graph, tenant)
    assert (edges[("/b", 0)].key, edges[("/b", 1)].key) == ("c#", "c")
    anchors = await stored_anchors(graph, tenant)
    assert_close(anchors["c#"][0], unit("c#"))  # type: ignore[arg-type]
    assert_close(anchors["c"][0], unit("c"))  # type: ignore[arg-type]
    assert anchors["c#"][0] != anchors["c"][0]


# ── AC3: the relationship vector round-trips through the driver ────────────


@pytest.mark.integration
async def test_ac3_surrounding_embedding_round_trips_through_the_driver(
    graph: GraphRepo, tenant: str, neo4j_server: tuple[str, str, str]
) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)

    uri, user, password = neo4j_server
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            result = await session.run(
                "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
                "RETURN s.url AS source, r.position AS position, r.surroundingEmbedding AS vec, "
                "r.surroundingEmbeddedHash AS hash, r.surroundingEmbeddingModel AS model",
                t=tenant,
            )
            rows = [record.data() async for record in result]
    finally:
        await driver.close()

    assert len(rows) == EDGES
    for row in rows:
        edge = (row["source"].removeprefix(BASE), row["position"])
        if edge in BLANK_SENTENCE_EDGES:
            assert (row["vec"], row["hash"], row["model"]) == (None, None, None), edge
            continue
        sentence = sentence_of(*edge)
        assert_close(row["vec"], unit(sentence))
        assert row["hash"] == hashlib.sha256(sentence.encode()).hexdigest(), edge
        assert row["model"] == MODEL


@pytest.mark.integration
async def test_gotcha_same_sentence_on_two_edges_is_embedded_once_and_written_to_both(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    _, fake, _ = await run(graph, tenant)

    assert sent_texts(fake).count(S1) == 1
    edges = await stored_edges(graph, tenant)
    shared = [edges[("/a", position)] for position in (0, 2, 3)]
    assert shared[0].vector == shared[1].vector == shared[2].vector
    assert {edge.digest for edge in shared} == {hashlib.sha256(S1.encode()).hexdigest()}
    # The same words in lower case are another sentence.
    assert edges[("/b", 5)].vector != shared[0].vector


@pytest.mark.integration
async def test_gotcha_sentences_are_embedded_as_documents(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant)
    _, fake, _ = await run(graph, tenant)

    sentence_calls = [call for call in fake.calls if set(call.texts) <= SENTENCES]
    assert {text for call in sentence_calls for text in call.texts} == SENTENCES
    assert {call.input_type for call in fake.calls} == {"document"}


@pytest.mark.integration
async def test_gotcha_blank_surrounding_text_is_skipped(graph: GraphRepo, tenant: str) -> None:
    await seed(graph, tenant)
    report, fake, _ = await run(graph, tenant)

    assert all(text.strip() for text in sent_texts(fake))
    edges = await stored_edges(graph, tenant)
    for edge in BLANK_SENTENCE_EDGES:
        assert (edges[edge].vector, edges[edge].digest, edges[edge].model) == (None, None, None)
    assert report.empty_sentences == len(BLANK_SENTENCE_EDGES)


# ── AC4: generic anchors are flagged, never embedded ───────────────────────


@pytest.mark.integration
async def test_ac4_generic_anchors_are_flagged_on_their_edges_and_never_embedded(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    report, fake, _ = await run(graph, tenant)

    assert not GENERIC_KEYS & set(sent_texts(fake))
    assert not GENERIC_KEYS & set(await stored_anchors(graph, tenant))
    edges = await stored_edges(graph, tenant)
    assert {edge for edge, stored in edges.items() if stored.generic} == {
        ("/a", 3),
        ("/a", 4),
        ("/a", 5),
    }
    assert all(stored.generic is False for edge, stored in edges.items() if edge[0] == "/b")
    assert (report.generic_anchors, report.generic_edges) == (2, 3)


@pytest.mark.integration
async def test_gotcha_read_more_arrow_is_generic_and_trimmed_from_the_key(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)

    edge = (await stored_edges(graph, tenant))[("/a", 5)]
    assert (edge.key, edge.generic) == ("read more", True)


@pytest.mark.integration
async def test_gotcha_empty_anchor_has_no_key_and_no_anchor_node(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    report, fake, _ = await run(graph, tenant)

    edges = await stored_edges(graph, tenant)
    for edge in EMPTY_ANCHOR_EDGES:
        assert (edges[edge].has_key_property, edges[edge].generic) == (False, False), edge
    assert "" not in await stored_anchors(graph, tenant)
    assert "" not in sent_texts(fake)
    assert report.empty_anchors == len(EMPTY_ANCHOR_EDGES)


# ── AC5: an immediate rerun is free; a changed text re-embeds only itself ──


@pytest.mark.integration
async def test_ac5_immediate_rerun_makes_no_voyage_call_and_writes_nothing(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)
    before = (await stored_edges(graph, tenant), await stored_anchors(graph, tenant))

    report, fake, logs = await run(graph, tenant)

    assert fake.call_count == 0
    assert (report.keys_written, report.anchors_embedded, report.sentences_embedded) == (0, 0, 0)
    assert (report.anchors_cached, report.sentences_cached) == (4, 8)
    assert (report.surrounding_edges_written, report.anchor_flushes, report.sentence_flushes) == (
        0,
        0,
        0,
    )
    assert (report.api_tokens, report.tokens) == (0, 0)
    assert events(logs, "embedding.links.flush") == []
    assert (await stored_edges(graph, tenant), await stored_anchors(graph, tenant)) == before


@pytest.mark.integration
async def test_ac5_a_changed_sentence_re_embeds_only_that_sentence(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)
    before = await stored_edges(graph, tenant)
    anchors = await stored_anchors(graph, tenant)

    new = "Our trail shoes grip wet granite."
    await seed(graph, tenant, changed(CRAFTED, ("/a", 0), sentence=f" {new}  "))
    report, fake, _ = await run(graph, tenant)

    assert [call.texts for call in fake.calls] == [(new,)]
    assert (report.sentences_embedded, report.surrounding_edges_written) == (1, 1)
    # S1 is still on /a#2 and /a#3, so 9 distinct sentences of which 8 are cached.
    assert (report.unique_sentences, report.sentences_cached) == (9, 8)
    assert (report.keys_written, report.anchors_embedded) == (0, 0)
    after = await stored_edges(graph, tenant)
    assert_close(after[("/a", 0)].vector, unit(new))
    assert after[("/a", 0)].digest == hashlib.sha256(new.encode()).hexdigest()
    assert {k: v for k, v in after.items() if k != ("/a", 0)} == {
        k: v for k, v in before.items() if k != ("/a", 0)
    }
    assert await stored_anchors(graph, tenant) == anchors


@pytest.mark.integration
async def test_a_changed_anchor_rewrites_one_key_and_embeds_only_the_new_anchor(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)

    await seed(graph, tenant, changed(CRAFTED, ("/b", 1), anchor="Rust"))
    report, fake, _ = await run(graph, tenant)

    assert [call.texts for call in fake.calls] == [("rust",)]
    assert (report.keys_written, report.anchors_embedded, report.anchors_cached) == (1, 1, 3)
    assert (await stored_edges(graph, tenant))[("/b", 1)].key == "rust"
    assert_close((await stored_anchors(graph, tenant))["rust"][0], unit("rust"))  # type: ignore[arg-type]


# ── AC6: the model guard runs before any Voyage call or write ──────────────


async def anchor_from_another_model(graph: GraphRepo, tenant: str) -> None:
    await graph.write_anchor_embeddings(
        tenant, ["trail shoes"], unit("trail shoes")[None, :], model=OTHER_MODEL, dimensions=DIM
    )


async def sentence_from_another_model(graph: GraphRepo, tenant: str) -> None:
    target = SentenceTarget(
        sentence_hash=sentence_hash(S1), edges=(EdgeRef(source_url=url("/a"), position=0),)
    )
    await graph.write_surrounding_embeddings(
        tenant, [target], unit(S1)[None, :], model=OTHER_MODEL, dimensions=DIM
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("setup", "step"),
    [
        pytest.param(anchor_from_another_model, "anchor_models", id="anchors"),
        pytest.param(sentence_from_another_model, "sentence_models", id="sentences"),
    ],
)
async def test_ac6_model_guard_fails_before_any_voyage_call_or_write(
    graph: GraphRepo,
    tenant: str,
    setup: Callable[[GraphRepo, str], Awaitable[None]],
    step: str,
) -> None:
    await seed(graph, tenant)
    await setup(graph, tenant)
    before = (await stored_edges(graph, tenant), await stored_anchors(graph, tenant))
    expected = (
        f"tenant {tenant}: stored vectors use a different model than configured; "
        f"found {OTHER_MODEL} (1); configured {MODEL}. "
        "Vectors from different models are not comparable"
    )
    fake = fake_voyage()

    with (
        capture_logs() as logs,
        pytest.raises(EmbeddingModelMismatchError, match=f"^{re.escape(expected)}$"),
    ):
        await embed_links(graph, client(fake), tenant)

    assert fake.call_count == 0
    assert (await stored_edges(graph, tenant), await stored_anchors(graph, tenant)) == before
    assert not any(edge.has_key_property for edge in before[0].values())
    (failed,) = events(logs, "embedding.links.failed")
    assert (failed["step"], failed["flush"], failed["keys_written"]) == (step, 0, 0)
    assert failed["error_type"] == "EmbeddingModelMismatchError"
    assert events(logs, "embedding.links.start") == []


@pytest.mark.integration
async def test_another_tenants_anchors_and_models_do_not_count(
    graph: GraphRepo, tenant: str
) -> None:
    other = f"{tenant}-other"
    await seed(graph, other)
    await run(graph, other)
    await anchor_from_another_model(graph, other)
    await seed(graph, tenant)

    report, fake, _ = await run(graph, tenant)

    assert (report.anchors_cached, report.anchors_embedded) == (0, 4)
    assert (report.sentences_cached, report.sentences_embedded) == (0, 8)
    assert len(sent_texts(fake)) == 4 + 8


# ── AC7: a later flush failing keeps the earlier ones; the rerun does the rest


REJECTED = InvalidRequestError("input rejected", http_status=400)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("fail_call", "step", "anchors_done", "sentences_done"),
    [
        # flush 2: anchors are calls 1-2, sentences 3-6, one request per flush.
        pytest.param(2, "anchor", 2, 0, id="second-anchor-flush"),
        pytest.param(4, "sentence", 4, 2, id="second-sentence-flush"),
    ],
)
async def test_ac7_voyage_failure_keeps_earlier_flushes_and_a_rerun_embeds_only_the_rest(
    graph: GraphRepo,
    tenant: str,
    fail_call: int,
    step: str,
    anchors_done: int,
    sentences_done: int,
) -> None:
    await seed(graph, tenant)
    failing = fake_voyage(fail_on={fail_call: REJECTED})

    with capture_logs() as logs, pytest.raises(EmbeddingRequestError, match="input rejected"):
        await embed_links(graph, client(failing), tenant, flush_size=2)

    committed = {text for call in failing.calls[: fail_call - 1] for text in call.texts}
    anchors = await stored_anchors(graph, tenant)
    assert set(anchors) == committed & EMBEDDED_ANCHORS
    assert len(anchors) == anchors_done
    edges = await stored_edges(graph, tenant)
    with_vector = {edge for edge, stored in edges.items() if stored.vector is not None}
    assert with_vector == {e for e in edges if sentence_of(*e) in committed}
    assert len({sentence_of(*e) for e in with_vector}) == sentences_done
    # Every key was written during the scan, before the first anchor flush.
    assert {edge: stored.key for edge, stored in edges.items()} == KEYS
    (failed,) = events(logs, "embedding.links.failed")
    assert (failed["step"], failed["flush"], failed["error_type"]) == (
        step,
        2,
        "EmbeddingRequestError",
    )
    assert (failed["anchors_embedded"], failed["sentences_embedded"]) == (
        anchors_done,
        sentences_done,
    )
    assert events(logs, "embedding.links.done") == []

    report, healthy, _ = await run(graph, tenant, flush_size=2)

    assert sorted(sent_texts(healthy)) == sorted((EMBEDDED_ANCHORS | SENTENCES) - committed)
    assert (report.anchors_cached, report.anchors_embedded) == (anchors_done, 4 - anchors_done)
    assert (report.sentences_cached, report.sentences_embedded) == (
        sentences_done,
        8 - sentences_done,
    )
    assert report.keys_written == 0
    final = await stored_edges(graph, tenant)
    for edge, stored in final.items():
        if edge not in BLANK_SENTENCE_EDGES:
            assert_close(stored.vector, unit(sentence_of(*edge)))
    assert set(await stored_anchors(graph, tenant)) == EMBEDDED_ANCHORS


@pytest.mark.integration
async def test_a_graph_change_during_the_run_fails_that_flush_and_keeps_earlier_ones(
    graph: GraphRepo, tenant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await seed(graph, tenant)
    write = graph.write_surrounding_embeddings
    calls: list[list[tuple[str, int]]] = []

    async def racing_write(
        tenant_id: str, targets: Sequence[SentenceTarget], *args: object, **kwargs: object
    ) -> int:
        calls.append([(e.source_url, e.position) for t in targets for e in t.edges])
        if len(calls) == 2:
            # A concurrent re-ingest drops one edge of flush 2 after the scan.
            source, position = calls[-1][0]
            await graph._auto(
                "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO {position: $p}]->() DELETE r",
                t=tenant,
                u=source,
                p=position,
            )
        return await write(tenant_id, targets, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(graph, "write_surrounding_embeddings", racing_write)
    with capture_logs() as logs, pytest.raises(DatabaseWriteError, match="rolled back"):
        await embed_links(graph, client(fake_voyage()), tenant, flush_size=FLUSH)

    edges = await stored_edges(graph, tenant)
    first = {(s.removeprefix(BASE), p) for s, p in calls[0]}
    assert {edge for edge, stored in edges.items() if stored.vector is not None} == first
    (failed,) = events(logs, "embedding.links.failed")
    assert (failed["step"], failed["flush"], failed["error_type"]) == (
        "sentence",
        2,
        "DatabaseWriteError",
    )
    assert failed["surrounding_edges_written"] == len(first)


def dropped_last(batch: EmbeddingBatch) -> EmbeddingBatch:
    return rebuilt(batch, batch.embeddings[:-1])


def swapped_first_two(batch: EmbeddingBatch) -> EmbeddingBatch:
    return rebuilt(batch, [batch.embeddings[1], batch.embeddings[0], *batch.embeddings[2:]])


@pytest.mark.integration
@pytest.mark.parametrize(
    ("target", "tamper", "step", "message"),
    [
        # Flush size 4: the four anchors are one flush and one request.
        pytest.param(
            "c#", dropped_last, "anchor", "anchor flush 1: 3 vectors for 4 keys", id="anchor-short"
        ),
        pytest.param(
            sentence_hash(S1),
            swapped_first_two,
            "sentence",
            r"vector 0 is for [0-9a-f]{64}, expected [0-9a-f]{64}",
            id="sentence-out-of-order",
        ),
    ],
)
async def test_a_batch_that_does_not_match_its_texts_fails_before_writing(
    graph: GraphRepo,
    tenant: str,
    target: str,
    tamper: Callable[[EmbeddingBatch], EmbeddingBatch],
    step: str,
    message: str,
) -> None:
    await seed(graph, tenant)
    tampering = TamperingVoyage(client(fake_voyage()), target, tamper)

    with capture_logs() as logs, pytest.raises(EmbeddingResponseError, match=message):
        await embed_links(graph, tampering, tenant, flush_size=4)  # type: ignore[arg-type]

    edges = await stored_edges(graph, tenant)
    assert all(edges[e].vector is None for e in (("/a", 0), ("/a", 2), ("/a", 3)))
    if step == "anchor":
        assert await stored_anchors(graph, tenant) == {}
    (failed,) = events(logs, "embedding.links.failed")
    assert (failed["step"], failed["error_type"]) == (step, "EmbeddingResponseError")


# ── progress events and the empty tenant ───────────────────────────────────


@pytest.mark.integration
async def test_one_flush_event_per_flush_with_cumulative_tokens(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    report, fake, logs = await run(graph, tenant)

    flushes = events(logs, "embedding.links.flush")
    keys = ("kind", "flush", "flushes", "written")
    assert [{key: entry[key] for key in keys} for entry in flushes] == [
        dict(zip(keys, ("anchor", 1, 2, 3), strict=True)),
        dict(zip(keys, ("anchor", 2, 2, 1), strict=True)),
        dict(zip(keys, ("sentence", 1, 3, 3), strict=True)),
        dict(zip(keys, ("sentence", 2, 3, 3), strict=True)),
        dict(zip(keys, ("sentence", 3, 3, 2), strict=True)),
    ]
    assert sum(int(entry["edges_written"]) for entry in flushes[2:]) == 10  # type: ignore[call-overload]
    per_call = [sum(words(t) for t in call.texts) + API_TOKEN_DRIFT for call in fake.calls]
    assert [entry["api_tokens"] for entry in flushes] == [
        sum(per_call[: n + 1]) for n in range(len(per_call))
    ]
    assert flushes[-1]["api_tokens"] == report.api_tokens
    assert all(entry["stage"] == "embedding.links" for entry in flushes)
    (start,) = events(logs, "embedding.links.start")
    assert (start["edges"], start["anchors_pending"], start["sentences_pending"]) == (12, 4, 8)
    assert (start["anchor_flushes"], start["sentence_flushes"]) == (2, 3)


@pytest.mark.integration
async def test_a_tenant_without_links_makes_no_call_and_no_flush(
    graph: GraphRepo, tenant: str
) -> None:
    await graph.upsert_pages(tenant, [Page(url=url("/lonely"), status_code=200)])
    report, fake, logs = await run(graph, tenant)

    assert fake.call_count == 0
    assert (report.edges, report.unique_anchors, report.unique_sentences) == (0, 0, 0)
    assert (report.anchor_dedupe_ratio, report.sentence_dedupe_ratio) == (None, None)
    assert events(logs, "embedding.links.flush") == []
    assert len(events(logs, "embedding.links.done")) == 1


@pytest.mark.integration
async def test_a_sentence_blanked_after_embedding_loses_its_vector_on_the_next_run(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)
    await seed(graph, tenant, changed(CRAFTED, ("/a", 1), sentence="  \n "))

    report, fake, logs = await run(graph, tenant)

    assert fake.call_count == 0
    assert (report.surrounding_cleared, report.empty_sentences) == (1, 3)
    # "Compare every trail shoe we sell." was only on /a#1; 7 sentences remain, all cached.
    assert (report.unique_sentences, report.sentences_cached) == (7, 7)
    rows = await graph._auto(
        "MATCH (:Page {tenantId: $t, url: $u})-[r:LINKS_TO {position: 1}]->() RETURN keys(r) AS k",
        t=tenant,
        u=url("/a"),
    )
    assert not {
        "surroundingEmbedding",
        "surroundingEmbeddedHash",
        "surroundingEmbeddingModel",
    } & set(
        rows[0]["k"]  # type: ignore[call-overload]
    )
    (start,) = events(logs, "embedding.links.start")
    (done,) = events(logs, "embedding.links.done")
    assert (start["surrounding_cleared"], done["surrounding_cleared"]) == (1, 1)
    assert events(logs, "embedding.links.stale_surrounding") == []

    again, fake_again, _ = await run(graph, tenant)
    assert (again.surrounding_cleared, fake_again.call_count) == (0, 0)


# An anchor over the Anchor key limit: 6 ASCII bytes then 3-byte characters, so a plain
# 4096-byte cut would split a character. The second cut lands just after a space.
CJK = "\N{CJK UNIFIED IDEOGRAPH-65E5}"
LONG_CJK = "Trail " + CJK * 1400
LONG_SPACED = "b" * (ANCHOR_KEY_MAX_BYTES - 1) + " tail words"
LONG: tuple[Spec, ...] = (
    ("/l", 0, "/t1", LONG_CJK, "A very long anchor."),
    ("/l", 1, "/t2", LONG_SPACED, "Another very long anchor."),
    ("/l", 2, "/t3", "Trail Shoes", "A short anchor."),
)
LONG_KEYS = {0: "trail " + CJK * 1363, 1: "b" * (ANCHOR_KEY_MAX_BYTES - 1), 2: "trail shoes"}


@pytest.mark.integration
async def test_a1_an_oversize_anchor_key_is_cut_on_a_character_boundary_and_still_joins(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant, LONG)
    report, fake, logs = await run(graph, tenant)

    assert len(LONG_KEYS[0].encode()) == ANCHOR_KEY_MAX_BYTES - 1
    edges = await stored_edges(graph, tenant)
    assert {position: edges[("/l", position)].key for position in LONG_KEYS} == LONG_KEYS
    assert all(len(str(edges[("/l", p)].key).encode()) <= ANCHOR_KEY_MAX_BYTES for p in (0, 1))
    rows = await graph._auto(
        "MATCH (s:Page {tenantId: $t})-[r:LINKS_TO]->() "
        "MATCH (a:Anchor {tenantId: s.tenantId, text: r.anchorKey}) "
        "RETURN r.position AS position, a.text AS text, a.embedding AS vec",
        t=tenant,
    )
    joined = {row["position"]: row for row in rows}
    assert set(joined) == {0, 1, 2}
    for position, key in LONG_KEYS.items():
        assert joined[position]["text"] == key
        assert_close(joined[position]["vec"], unit(key))  # type: ignore[arg-type]
    assert report.anchors_embedded == 3
    (warning,) = events(logs, "embedding.links.long_anchors")
    assert (warning["edges"], warning["log_level"]) == (2, "warning")

    again, fake_again, _ = await run(graph, tenant)
    assert (fake_again.call_count, again.keys_written, again.anchors_cached) == (0, 0, 3)
    assert sorted(text for text in sent_texts(fake) if text in LONG_KEYS.values()) == sorted(
        LONG_KEYS.values()
    )


# ── guards that run before any I/O ──────────────────────────────────────────


class Untouchable:
    """Stands in for a store the code under test must not reach."""

    def __getattr__(self, name: str) -> NoReturn:
        raise AssertionError(f"{name} was used before the argument checks")


@pytest.mark.parametrize(
    ("tenant_id", "flush_size", "message"),
    [
        (" ", FLUSH_SIZE, "tenant_id must be a non-empty string"),
        ("acme", 0, "flush_size must be at least 1"),
        ("acme", -1, "flush_size must be at least 1"),
    ],
)
async def test_bad_arguments_raise_before_any_io(
    tenant_id: str, flush_size: int, message: str
) -> None:
    fake = fake_voyage()
    with pytest.raises(ValueError, match=message):
        await embed_links(Untouchable(), client(fake), tenant_id, flush_size=flush_size)  # type: ignore[arg-type]
    assert fake.call_count == 0


async def test_a_client_dimension_other_than_2048_fails_before_any_io() -> None:
    fake = FakeVoyage(dimension=16)
    with pytest.raises(SchemaError, match="embedding dimension 16 does not match the 2048d"):
        await embed_links(Untouchable(), client(fake), "acme")  # type: ignore[arg-type]
    assert fake.call_count == 0


# ── sentence normalisation and hash ─────────────────────────────────────────


def test_sentence_hash_is_sha256_of_the_whitespace_normalised_sentence() -> None:
    assert normalise_sentence("  Our  trail\n\tshoes. ") == "Our trail shoes."
    expected = hashlib.sha256(b"Our trail shoes.").hexdigest()
    assert sentence_hash("  Our  trail\n\tshoes. ") == sentence_hash("Our trail shoes.") == expected


def test_sentence_hash_keeps_case() -> None:
    assert sentence_hash("Our trail shoes.") != sentence_hash("our trail shoes.")


# ── amendments: reuse within a tenant, tenant isolation, tenant overrides ──


@pytest.mark.integration
async def test_a_recreated_edge_reuses_its_tenants_sentence_vector_without_voyage(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    await run(graph, tenant)
    source, position, *_ = CRAFTED[0]
    original = (await stored_edges(graph, tenant))[(source, position)]
    moved = tuple(
        (s, 99 if (s, p) == (source, position) else p, t, a, x) for s, p, t, a, x in CRAFTED
    )
    await seed(graph, tenant, moved)

    report, fake, _ = await run(graph, tenant)

    assert fake.call_count == 0
    assert report.sentences_reused == 1
    assert report.sentences_embedded == 0
    assert (await stored_edges(graph, tenant))[(source, 99)].vector == original.vector


@pytest.mark.integration
async def test_two_tenants_with_identical_text_share_nothing(graph: GraphRepo, tenant: str) -> None:
    other = f"{tenant}-other"
    await seed(graph, tenant)
    await seed(graph, other)
    first, first_fake, _ = await run(graph, tenant)

    second, second_fake, _ = await run(graph, other)

    assert second_fake.call_count == first_fake.call_count > 0
    assert sorted(sent_texts(second_fake)) == sorted(sent_texts(first_fake))
    assert (second.anchors_embedded, second.sentences_embedded) == (
        first.anchors_embedded,
        first.sentences_embedded,
    )
    assert second.sentences_reused == 0
    assert set(await stored_anchors(graph, other)) == set(await stored_anchors(graph, tenant))
    await graph.delete_tenant(other)


@pytest.mark.integration
async def test_tenant_overrides_change_which_anchors_are_generic(
    graph: GraphRepo, tenant: str
) -> None:
    await seed(graph, tenant)
    rules = AnchorRules(generic_add={"Trail Shoes"}, generic_remove={"click here"})
    with capture_logs():
        report = await embed_links(
            graph, client(fake_voyage()), tenant, flush_size=FLUSH, rules=rules
        )

    anchors = set(await stored_anchors(graph, tenant))
    assert "trail shoes" not in anchors
    assert "click here" in anchors
    assert report.generic_anchors == len(GENERIC_KEYS)
