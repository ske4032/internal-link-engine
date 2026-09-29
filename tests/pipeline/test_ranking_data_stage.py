"""held_out_rounds on real Neo4j and Mongo: labels are the round's hidden links only, every
link-derived value is recomputed on the round's view, the hidden links' anchors are gone from
their sources' spans and their targets' inbound anchors, nothing is written to either store or
for another tenant, and the round cache is keyed on what a round depends on."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, cast

import pyarrow.parquet as pq
import pytest
from ranking_seed import (
    NOISE_URLS,
    OFFSETS,
    ORPHANS,
    SIZE,
    TOPICS,
    URLS,
    body,
    body_links,
    inbound,
    keyword,
    linked,
    links,
    names,
    orphan_targets,
    page_url,
    phrase,
    seed_ranking,
    voyage,
)
from ranking_seed import path as page_path
from store_state import graph_state, mongo_state
from structlog.testing import capture_logs
from test_keyword_stage import page_record
from voyage_fakes import FakeVoyage, client
from voyageai.error import ServiceUnavailableError

from linking_engine.discovery.candidates import retrieve_candidates
from linking_engine.discovery.features import FEATURE_COLUMNS, KEY_COLUMNS
from linking_engine.ml.quality import hide_links
from linking_engine.models import FeatureWeight, HeldOutSettings, ScorerWeights
from linking_engine.pipeline.anchor_selection import _inbound_anchors, compute_anchor_choices
from linking_engine.pipeline.anchors import AnchorView, lexical_run
from linking_engine.pipeline.ranking_data import ROUND_SCHEMA, held_out_rounds, load_rounds
from linking_engine.pipeline.text_vectors import cache_path

if TYPE_CHECKING:
    from pathlib import Path

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TWO_ROUNDS = HeldOutSettings(rounds=2)
ONE_ROUND = HeldOutSettings(rounds=1)
PLACEMENT = ["context_relevance", "anchor_target_fit"]


def hidden_in(round_: int, share: float = 0.1) -> frozenset[tuple[str, str]]:
    return hide_links(body_links(), share=share, fold=round_)


def span(topic: str, source: int, target: int) -> tuple[int, int]:
    """Where page ``source``'s copy anchors its link to page ``target``."""
    text, sentence = body(topic, source), linked(topic, target)
    start = text.find(sentence) + sentence.find(phrase(topic, target))
    return start, start + len(phrase(topic, target))


def files(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}


@pytest.mark.integration
async def test_rounds_labels_are_hidden_links_and_views_recomputed(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_ranking(graph, mongo, tenant)

    with capture_logs() as logs:
        rounds = await held_out_rounds(
            graph, mongo, tenant, settings=TWO_ROUNDS, cache_dir=tmp_path, voyage=client(voyage())
        )

    assert (rounds.pages, rounds.body_links) == (len(URLS), len(body_links()))
    assert rounds.cache_hit == (False, False)
    assert hidden_in(0) == hide_links(body_links()), "round 0 is the quality evaluation's set"
    assert not hidden_in(0) & hidden_in(1), "rounds overlap"
    for path in rounds.paths:
        assert pq.read_schema(path).remove_metadata().equals(ROUND_SCHEMA)
    assert ROUND_SCHEMA.names == [
        "round",
        *KEY_COLUMNS,
        "label",
        "baseline_score",
        *FEATURE_COLUMNS,
    ]
    frame = load_rounds(rounds.paths)
    stored = inbound()
    outlinks = Counter(s for s, _ in body_links())
    visible_links = body_links()

    for summary in rounds.summaries:
        r = summary.round
        hidden = hidden_in(r)
        rows = frame[frame["round"] == r]
        pairs = set(zip(rows["source_url"], rows["target_url"], strict=True))
        positives = rows[rows["label"] == 1]
        found = set(zip(positives["source_url"], positives["target_url"], strict=True))

        assert found == hidden, f"round {r}: positives are not exactly its hidden links"
        assert set(rows["label"]) == {0, 1}
        assert not pairs & (visible_links - hidden), f"round {r}: a visible link is a candidate"
        # Every hidden link is a same-topic pair, and a topic's pages rank first for its targets.
        assert (summary.hidden, summary.recoverable, summary.positives) == (len(hidden),) * 3
        assert summary.pairs == len(rows) == len(pairs)
        assert summary.groups_with_positive == len({s for s, _ in hidden})

        into = Counter(t for _, t in hidden)
        out_of = Counter(s for s, _ in hidden)
        assert rows["target_inbound_count"].tolist() == [
            stored[t] - into[t] for t in rows["target_url"]
        ], f"round {r}: inbound counts are not recomputed without its hidden links"
        assert rows["source_outbound_count"].tolist() == [
            outlinks[s] - out_of[s] for s in rows["source_url"]
        ], f"round {r}: outbound counts are not recomputed without its hidden links"
        assert rows["baseline_score"].notna().all()
        # A hidden link's span is free on the view: the exact rung places its anchor there.
        assert positives[PLACEMENT].notna().all().all(), (
            f"round {r}: a hidden pair has no placement features"
        )
        named = [names(s, t) for s, t in zip(rows["source_url"], rows["target_url"], strict=True)]
        silent = rows[[not flag for flag in named]]
        assert len(silent) > 0
        assert silent["context_relevance"].isna().all(), (
            f"round {r}: a pair whose source never names its target has a placement feature"
        )
        assert rows[named]["context_relevance"].notna().all(), (
            f"round {r}: a pair whose source names its target has no placement feature"
        )
        negatives = rows[rows["label"] == 0]
        assert summary.positive_placement_share == 1.0
        assert summary.negative_placement_share == pytest.approx(
            negatives[PLACEMENT].notna().any(axis=1).mean()
        )
        assert summary.negative_placement_share < 0.5, "the planted gap the protocol creates"

    assert {line["tenant_id"] for line in logs if line["event"] == "ranker.round"} == {tenant}
    logged = " ".join(str(value) for entry in logs for value in entry.values())
    assert [u for u in URLS if u in logged] == [], "page urls in the log"


@pytest.mark.integration
async def test_orphan_targets_are_the_full_graph_pages_without_inbound_body_links(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    """Orphans come from the full graph, not a round's view: a page whose only inbound link is
    hidden in round 0 has none on that view, yet is a positive there, never an orphan."""
    await seed_ranking(graph, mongo, tenant)
    source, target = min(hidden_in(0))
    others = {s for s, t in body_links() if t == target and s != source}
    for other in sorted(others):
        await graph.replace_links(
            tenant,
            [other],
            [link for link in links() if link.source_url == other and link.target_url != target],
        )

    rounds = await held_out_rounds(
        graph, mongo, tenant, settings=TWO_ROUNDS, cache_dir=tmp_path, voyage=client(voyage())
    )

    assert rounds.orphan_targets == orphan_targets() == ORPHANS | NOISE_URLS
    assert target not in rounds.orphan_targets
    frame = load_rounds(rounds.paths)
    first = frame[frame["round"] == 0]
    assert (
        (first["source_url"] == source) & (first["target_url"] == target) & (first["label"] == 1)
    ).any()
    into = first[first["target_url"] == target]
    assert set(into["target_inbound_count"]) == {0}, "the view still counts the hidden link"
    # The round files keep the orphan rows; training leaves them out.
    kept = frame[frame["target_url"].isin(rounds.orphan_targets)]
    assert set(kept["target_url"]) == rounds.orphan_targets
    assert set(kept["label"]) == {0}


@pytest.mark.integration
async def test_a_view_frees_the_hidden_links_spans_and_inbound_anchors(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_ranking(graph, mongo, tenant)
    hidden = hidden_in(0)
    snapshot = await graph.link_graph(tenant)
    held = await retrieve_candidates(
        graph,
        tenant,
        links=[link for link in snapshot.links if link not in hidden],
        vectors=await graph.content_vectors(tenant),
    )
    view = AnchorView(held, hidden)

    production = await lexical_run(graph, mongo, tenant, cache_dir=tmp_path)
    run = await lexical_run(graph, mongo, tenant, cache_dir=tmp_path, view=view)

    candidates = {(s, entry.target_url) for entry in held.targets for s in entry.sources}
    assert {(p.source_url, p.target_url) for p in run.pairs} == candidates
    assert run.bridge_pairs == 0
    assert hidden <= candidates
    assert not hidden & {(p.source_url, p.target_url) for p in production.pairs}
    for topic in TOPICS:
        for i in range(SIZE):
            source = page_url(topic, i)
            every = {span(topic, i, (i + k) % SIZE) for k in OFFSETS}
            freed = {
                span(topic, i, (i + k) % SIZE)
                for k in OFFSETS
                if (source, page_url(topic, (i + k) % SIZE)) in hidden
            }
            assert set(production.existing[source]) == every
            assert set(run.existing[source]) == every - freed, f"{source}: a hidden span is kept"
    for source, target in hidden:
        topic, index = target.split("/")[1], URLS.index(target) % SIZE
        assert phrase(topic, index) in {m.phrase.casefold() for m in run.matches[(source, target)]}

    anchors = await _inbound_anchors(
        graph,
        tenant,
        {t for _, t in body_links()},
        generic_add=frozenset(),
        generic_remove=frozenset(),
        hidden=hidden,
    )
    for target in {t for _, t in body_links()}:
        linking = {s for s, t in body_links() if t == target}
        kept = {s for s, t in body_links() - hidden if t == target}
        assert {source for _, source in anchors.get(target, [])} == kept, (
            f"{target}: a hidden link's anchor still counts as an inbound anchor"
        )
        assert kept <= linking

    selection = await compute_anchor_choices(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage()), view=view
    )

    chosen = {(c.match.source_url, c.match.target_url): c for c in selection.choices if c.rank == 1}
    for source, target in hidden:
        choice = chosen[(source, target)]
        topic, index = target.split("/")[1], URLS.index(target) % SIZE
        assert choice.match.sentence == linked(topic, index)
        assert choice.context_relevance is not None
    assert selection.report.pairs == len(candidates)


@pytest.mark.integration
async def test_rounds_write_nothing_to_stores_and_other_tenant_untouched(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    other = f"{tenant}-other"
    await seed_ranking(graph, mongo, tenant)
    await seed_ranking(graph, mongo, other)
    stored = (await graph_state(graph, tenant), await graph_state(graph, other))
    documents = await mongo_state(mongo)
    fake = voyage()

    first = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(fake)
    )

    assert (await graph_state(graph, tenant), await graph_state(graph, other)) == stored
    assert await mongo_state(mongo) == documents, "the rounds wrote to Mongo"
    written = files(tmp_path)
    assert {name.split("/")[0] for name in written} == {tenant}, "a file outside the tenant"
    assert f"{tenant}/ranker/rounds/{first.key}.round-0.parquet" in written
    production = ("anchor_choices.parquet", "unanchored_pairs.parquet", "anchors.parquet")
    assert not [name for name in written if name.endswith(production)], (
        "held-out anchors reached a production file"
    )
    assert not [name for name in written if "features" in name], "held-out features were cached"
    embedded = fake.texts_seen
    assert embedded > 0

    again = await held_out_rounds(
        graph, mongo, other, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(fake)
    )

    assert again.cache_hit == (False,), "another tenant reused the tenant's round"
    assert again.key != first.key
    assert fake.texts_seen > embedded, "another tenant reused the tenant's vectors"
    assert again.summaries == first.summaries, "identical planted tenants, identical rounds"
    assert await mongo_state(mongo) == documents


@pytest.mark.integration
async def test_rounds_cache_hit_on_rerun_and_key_changes_with_share(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_ranking(graph, mongo, tenant)
    first = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(voyage())
    )
    [path] = first.paths
    written = path.read_bytes()
    fake = voyage()

    again = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(fake)
    )

    assert (first.cache_hit, again.cache_hit) == ((False,), (True,))
    assert (again.key, again.paths, again.summaries) == (first.key, first.paths, first.summaries)
    assert path.read_bytes() == written, "a cache hit rewrote the round"
    assert fake.call_count == 0, "a cache hit embedded text"

    theirs = tmp_path / f"{tenant}-other" / "ranker" / "rounds" / f"{first.key}.round-0.parquet"
    theirs.parent.mkdir(parents=True)
    theirs.write_bytes(written)
    model = path.parent.parent / "model-1.txt"
    model.write_text("tree")

    wider = await held_out_rounds(
        graph,
        mongo,
        tenant,
        settings=HeldOutSettings(rounds=1, share=0.2),
        cache_dir=tmp_path,
        voyage=client(voyage()),
    )
    assert wider.key != first.key
    assert wider.cache_hit == (False,)
    assert wider.summaries[0].hidden > first.summaries[0].hidden
    # Round files of an earlier key go; nothing else of the tenant's, nothing of another.
    assert not path.exists(), "a stale round file was kept"
    assert wider.paths[0].is_file()
    assert (theirs.read_bytes(), model.read_text()) == (written, "tree")

    # A stored input changes: the first page drops its last link.
    [dropped] = [link for link in links() if link.source_url == URLS[0] and link.position == 5]
    kept = [link for link in links() if link.source_url == URLS[0] and link != dropped]
    await graph.replace_links(tenant, [URLS[0]], kept)
    changed = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(voyage())
    )
    assert changed.key != first.key
    assert changed.cache_hit == (False,)
    assert changed.body_links == first.body_links - 1

    # A stored input that is not a link: one page gains GSC totals.
    await mongo._db["gsc_metrics"].insert_one(
        {
            "tenantId": tenant,
            "url": URLS[1],
            "impressions_28d": 400,
            "clicks_28d": 20,
            "avg_position": 4.0,
            "query_count": 1,
        }
    )
    measured = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(voyage())
    )
    assert measured.key != changed.key, "a GSC row does not key the rounds"
    assert measured.cache_hit == (False,)
    assert measured.body_links == changed.body_links

    keyless = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=None
    )
    assert keyless.key != measured.key, "the Voyage model does not key the rounds"
    assert keyless.cache_hit == (False,)


@pytest.mark.integration
async def test_a_round_after_a_voyage_outage_is_partial_and_never_a_cache_hit(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    await seed_ranking(graph, mongo, tenant)
    down = FakeVoyage(
        dimension=voyage().dimension,
        failures=[ServiceUnavailableError("down") for _ in range(100)],
    )

    broken = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(down)
    )

    [path] = broken.paths
    assert pq.read_schema(path).metadata[b"partial"] == b"true"
    assert broken.summaries[0].positives == broken.summaries[0].hidden, (
        "the lexical rungs still place the hidden anchors"
    )

    healed = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(voyage())
    )

    assert healed.cache_hit == (False,), "a partial round was reused"
    assert pq.read_schema(path).metadata[b"partial"] == b"false"

    path.write_bytes(b"not parquet")
    rebuilt = await held_out_rounds(
        graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(voyage())
    )
    assert rebuilt.cache_hit == (False,), "an unreadable round was reused"
    assert rebuilt.summaries == healed.summaries


@pytest.mark.integration
async def test_the_round_key_covers_every_stored_input_a_round_reads(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    """Each stored value a round's rows depend on, beyond the links, changes the key; writing a
    page again unchanged does not."""
    await seed_ranking(graph, mongo, tenant)
    source, noise = URLS[0], sorted(NOISE_URLS)[0]
    topic, index = source.split("/")[1], 0

    async def key() -> tuple[str, tuple[bool, ...]]:
        found = await held_out_rounds(
            graph, mongo, tenant, settings=ONE_ROUND, cache_dir=tmp_path, voyage=client(voyage())
        )
        return found.key, found.cache_hit

    async def edge(assignment: str) -> None:
        await graph._auto(
            "MATCH (:Page {tenantId: $t, url: $s})-[r:LINKS_TO {position: 0}]->() "
            f"SET {assignment}",
            t=tenant,
            s=source,
        )

    async def page(assignment: str) -> None:
        await graph._auto(
            f"MATCH (p:Page {{tenantId: $t, url: $u}}) SET {assignment}", t=tenant, u=noise
        )

    async def record(title: str, text: str) -> None:
        await mongo.write_pages(
            tenant,
            [page_record(page_path(topic, index), 200, title, keyword(topic, index), text, "en")],
            [],
        )

    changes = {
        "anchor text": lambda: edge("r.anchorText = 'trail kit'"),
        "surrounding text": lambda: edge("r.surroundingText = 'Pack it.'"),
        "generic flag": lambda: edge("r.anchorGeneric = true, r.anchorTargetFit = null"),
        "page title": lambda: record(f"{keyword(topic, index)} | Summit", body(topic, index)),
        "page body": lambda: record(
            f"{keyword(topic, index)} | Summit", f"{body(topic, index)} Updated."
        ),
        "indexable": lambda: page("p.isIndexable = false"),
        "canonical copy": lambda: page("p.isCanonical = false"),
    }
    last, hit = await key()
    assert hit == (False,)
    await record(f"{keyword(topic, index)} | Acme", body(topic, index))
    assert await key() == (last, (True,)), "writing a page unchanged changed the key"

    for name, change in changes.items():
        await change()
        found, hit = await key()
        assert found != last, f"the {name} does not key the rounds"
        assert hit == (False,), name
        last = found


@pytest.mark.integration
async def test_a_voyage_failure_after_the_semantic_rung_is_the_selections_skip_reason(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, tmp_path: Path
) -> None:
    """A view whose only candidates are its hidden links: the lexical rungs place every pair, so
    the semantic rung reads only cached keywords and phrases. With the sentence cache gone, the
    first Voyage call is for the chosen anchors' sentences, after the rung; that outage is the
    selection's skip reason, which a round reads to mark itself partial."""
    await seed_ranking(graph, mongo, tenant)
    hidden = hidden_in(0)
    snapshot = await graph.link_graph(tenant)
    held = await retrieve_candidates(
        graph,
        tenant,
        links=[link for link in snapshot.links if link not in hidden],
        vectors=await graph.content_vectors(tenant),
    )
    kept = []
    for entry in held.targets:
        pairs = [
            (source, similarity)
            for source, similarity in zip(entry.sources, entry.similarities, strict=True)
            if (source, entry.target_url) in hidden
        ]
        if pairs:
            sources, similarities = zip(*pairs, strict=True)
            kept.append(entry.model_copy(update={"sources": sources, "similarities": similarities}))
    view = AnchorView(held.model_copy(update={"targets": tuple(kept)}), hidden)
    warm = await compute_anchor_choices(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(voyage()), view=view
    )
    assert warm.skipped_reason is None
    cache_path(tmp_path, tenant, "sentences").unlink()
    down = voyage()
    down.fail_on = {n: ServiceUnavailableError("down", http_status=503) for n in range(1, 100)}

    selection = await compute_anchor_choices(
        graph, mongo, tenant, cache_dir=tmp_path, voyage=client(down), view=view
    )

    assert down.call_count > 0
    assert selection.report.semantic_skipped_reason is None, "the outage reached the rung"
    assert selection.skipped_reason == "Voyage unavailable after retries (ServiceUnavailableError)"
    chosen = [choice for choice in selection.choices if choice.rank == 1]
    assert {(c.match.source_url, c.match.target_url) for c in chosen} == hidden
    assert all(choice.context_relevance is None for choice in chosen), "a sentence vector"
    assert all(choice.anchor_target_fit is not None for choice in chosen), "a phrase vector lost"


class WeightsOnly:
    """A Mongo stand-in that only has scorer weights; any other read fails the test."""

    async def get_scorer_weights(self, tenant_id: str) -> ScorerWeights:
        return ScorerWeights(
            version="custom-1", features=(FeatureWeight(column="no_such_column", weight=1.0),)
        )


@pytest.mark.parametrize("tenant_id", [" ", "..", "../escape", "a/b"])
async def test_rounds_refuse_a_tenant_that_is_not_a_directory_name_before_any_read(
    tmp_path: Path, tenant_id: str
) -> None:
    unused = object()
    with pytest.raises(ValueError, match="tenant_id"):
        await held_out_rounds(
            cast("GraphRepo", unused),
            cast("MongoRepo", unused),
            tenant_id,
            settings=ONE_ROUND,
            cache_dir=tmp_path,
            voyage=None,
        )
    assert files(tmp_path) == set()


async def test_rounds_refuse_weights_naming_an_unknown_column_before_the_graph_is_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no_such_column"):
        await held_out_rounds(
            cast("GraphRepo", object()),
            cast("MongoRepo", WeightsOnly()),
            "test-weights",
            settings=ONE_ROUND,
            cache_dir=tmp_path,
            voyage=None,
        )


def test_the_planted_links_anchor_where_the_copy_names_their_targets() -> None:
    """The fixture itself: each link's surrounding sentence is in its source's copy and holds
    its anchor, so the anchor stages can locate every span."""
    for link in links():
        topic, index = link.source_url.split("/")[1], URLS.index(link.source_url) % SIZE
        text = body(topic, index)
        assert link.surrounding_text in text
        assert link.anchor_text in link.surrounding_text
    assert len(body_links()) == len(links()) == len(TOPICS) * SIZE * len(OFFSETS)
