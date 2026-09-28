"""Hub bridges: links mostly stay inside a hub, and every hub stays reachable from the others.

Per language, a hub pair is connected in one direction when at least ``FLOOR_SHARE`` of the
source hub's pages link into the other hub. The pairs held to that floor are the maximum spanning
tree over hub centroid similarity, each hub's nearest hubs and the widest bridge gaps; each
direction of them below the floor gets ranked page pairs to lift it. Noise pages and pages
without a hub are outside the guarantee.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import numpy as np

from linking_engine.discovery.signals import HUB_NOISE, jaccard
from linking_engine.models import BridgeLink, BridgeReason, BridgeReport, HubPair

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    import numpy.typing as npt

    from linking_engine.models import KeywordRung, PageStructure

STAGE: Final = "hub-bridges"
FLOOR_SHARE: Final = 0.05
NEAREST_HUBS: Final = 2
TOP_GAP_PAIRS: Final = 3
# Alternative targets per slot, at most 2: a bridge link has ranks 1 to 3.
ALTERNATIVES: Final = 2
# Sources whose best cosine into the other hub agrees to this many decimals are equally
# relevant; authority orders them.
RELEVANCE_DECIMALS: Final = 2
# Shared GSC queries kept on a hub pair as its evidence.
SHARED_QUERIES: Final = 10
# The bridge gap weighs centroid cosine and query Jaccard, less the scaled link density;
# without GSC data the cosine takes both weights.
COSINE_WEIGHT: Final = 0.4
JACCARD_WEIGHT: Final = 0.6
DENSITY_WEIGHT: Final = 50.0

HubKey = tuple[str | None, int]


@dataclass(frozen=True, slots=True)
class _HubIndex:
    # (language, hub) -> its pages' urls, ascending.
    members: dict[HubKey, tuple[str, ...]]
    pages: dict[str, PageStructure]
    # (language, hub, other hub) -> pages of the hub with a body link into the other.
    linking: dict[tuple[str | None, int, int], set[str]]
    # (language, lower hub, higher hub) -> distinct linked page pairs, both directions.
    pair_links: Counter[tuple[str | None, int, int]]


def floor_pages(size: int) -> int:
    """Pages of a hub of ``size`` that must link into another hub for it to count as connected."""
    if size < 1:
        raise ValueError("a hub has at least one page")
    # Rounded first, so 0.05 * 60 is 3 and not the float just above it.
    return max(1, math.ceil(round(FLOOR_SHARE * size, 9)))


def _language_key(language: str | None) -> tuple[bool, str]:
    return language is not None, language or ""


def _in_hub(hub_id: int | None) -> bool:
    return hub_id is not None and hub_id != HUB_NOISE


def _find(parent: dict[int, int], hub: int) -> int:
    while parent.get(hub, hub) != hub:
        parent[hub] = parent.get(parent[hub], parent[hub])
        hub = parent[hub]
    return hub


def _union(parent: dict[int, int], a: int, b: int) -> bool:
    root_a, root_b = _find(parent, a), _find(parent, b)
    if root_a == root_b:
        return False
    parent[max(root_a, root_b)] = min(root_a, root_b)
    return True


def _hub_index(pages: Sequence[PageStructure], links: Iterable[tuple[str, str]]) -> _HubIndex:
    by_url = {page.url: page for page in pages}
    if len(by_url) != len(pages):
        raise ValueError("duplicate page urls")
    grouped: defaultdict[HubKey, list[str]] = defaultdict(list)
    for page in pages:
        if _in_hub(page.hub_id) and page.hub_id is not None:
            grouped[(page.language, page.hub_id)].append(page.url)
    linking: defaultdict[tuple[str | None, int, int], set[str]] = defaultdict(set)
    pair_links: Counter[tuple[str | None, int, int]] = Counter()
    for source_url, target_url in set(links):
        source, target = by_url.get(source_url), by_url.get(target_url)
        if (
            source is None
            or target is None
            or source.language != target.language
            or not _in_hub(source.hub_id)
            or not _in_hub(target.hub_id)
            or source.hub_id is None
            or target.hub_id is None
            or source.hub_id == target.hub_id
        ):
            continue
        linking[(source.language, source.hub_id, target.hub_id)].add(source_url)
        low, high = sorted((source.hub_id, target.hub_id))
        pair_links[(source.language, low, high)] += 1
    return _HubIndex(
        members={key: tuple(sorted(urls)) for key, urls in grouped.items()},
        pages=by_url,
        linking=dict(linking),
        pair_links=pair_links,
    )


def _unit(vector: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    # A zero vector scores 0 against everything.
    return values / norm if norm else values


def hub_pairs(
    pages: Sequence[PageStructure],
    centroids: Mapping[int, npt.ArrayLike],
    links: Iterable[tuple[str, str]],
    queries: Mapping[str, frozenset[str]] | None = None,
) -> list[HubPair]:
    """Every pair of hubs with pages in one language, by language then hub ids; no reasons yet.

    ``links`` are body links as (source, target); ``queries`` each page's normalised GSC
    queries. Without them (None) the Jaccard term drops out and the cosine takes its weight.
    With them, a pair keeps up to ``SHARED_QUERIES`` of the queries both hubs have, those on
    the most pages first, then by text. A link counts once per page pair, and only between
    pages of the same language.
    """
    index = _hub_index(pages, links)
    hubs = sorted({hub for _, hub in index.members})
    missing = [hub for hub in hubs if hub not in centroids]
    if missing:
        raise ValueError(f"hubs {missing} have pages but no stored centroid")
    unit = {hub: _unit(centroids[hub]) for hub in hubs}
    if len({vector.shape for vector in unit.values()}) > 1:
        raise ValueError("hub centroids differ in length")

    found: list[HubPair] = []
    for language in sorted({language for language, _ in index.members}, key=_language_key):
        in_language = sorted(hub for lang, hub in index.members if lang == language)
        # Per hub, each query and how many of its pages have it.
        hub_queries: dict[int, Counter[str]] = {
            hub: Counter(
                query
                for url in index.members[(language, hub)]
                for query in (queries or {}).get(url, frozenset())
            )
            for hub in in_language
        }
        query_sets = {hub: frozenset(counts) for hub, counts in hub_queries.items()}
        for i, hub_a in enumerate(in_language):
            for hub_b in in_language[i + 1 :]:
                size_a = len(index.members[(language, hub_a)])
                size_b = len(index.members[(language, hub_b)])
                density = index.pair_links[(language, hub_a, hub_b)] / (size_a * size_b)
                cosine = float(np.clip(unit[hub_a] @ unit[hub_b], -1.0, 1.0))
                on_a, on_b = hub_queries[hub_a], hub_queries[hub_b]
                overlap = None if queries is None else jaccard(query_sets[hub_a], query_sets[hub_b])
                shared = sorted(on_a.keys() & on_b.keys(), key=lambda q: (-on_a[q] - on_b[q], q))
                topic = (
                    cosine if overlap is None else COSINE_WEIGHT * cosine + JACCARD_WEIGHT * overlap
                )
                found.append(
                    HubPair(
                        language=language,
                        hub_a=hub_a,
                        hub_b=hub_b,
                        size_a=size_a,
                        size_b=size_b,
                        pages_ab=len(index.linking.get((language, hub_a, hub_b), ())),
                        pages_ba=len(index.linking.get((language, hub_b, hub_a), ())),
                        link_density=density,
                        centroid_cosine=cosine,
                        query_jaccard=overlap,
                        shared_queries=tuple(shared[:SHARED_QUERIES]),
                        bridge_gap=topic - DENSITY_WEIGHT * density,
                    )
                )
    return found


def spanning_tree(pairs: Iterable[HubPair]) -> set[tuple[int, int]]:
    """The maximum spanning tree over centroid cosine of one language's hub pairs, as
    (hub_a, hub_b); ties go to the lower hub ids."""
    ordered = sorted(pairs, key=lambda pair: (-pair.centroid_cosine, pair.hub_a, pair.hub_b))
    if len({pair.language for pair in ordered}) > 1:
        raise ValueError("a spanning tree spans the hubs of one language")
    parent: dict[int, int] = {}
    return {(pair.hub_a, pair.hub_b) for pair in ordered if _union(parent, pair.hub_a, pair.hub_b)}


def covered_pairs(
    pairs: Iterable[HubPair], *, nearest: int = NEAREST_HUBS, top_gap: int = TOP_GAP_PAIRS
) -> list[HubPair]:
    """Every pair with the reasons it is held to the floor, per language: the spanning tree,
    each hub's ``nearest`` hubs by centroid cosine and the ``top_gap`` widest bridge gaps.
    Uncovered pairs keep no reasons. Ordered by language then hub ids."""
    if nearest < 0 or top_gap < 0:
        raise ValueError("nearest and top_gap cannot be negative")
    by_language: defaultdict[str | None, list[HubPair]] = defaultdict(list)
    for pair in pairs:
        by_language[pair.language].append(pair)
    covered: list[HubPair] = []
    for language in sorted(by_language, key=_language_key):
        group = sorted(by_language[language], key=lambda pair: (pair.hub_a, pair.hub_b))
        reasons: defaultdict[tuple[int, int], set[BridgeReason]] = defaultdict(set)
        for key in spanning_tree(group):
            reasons[key].add(BridgeReason.SPANNING_TREE)
        around: defaultdict[int, list[tuple[float, int, HubPair]]] = defaultdict(list)
        for pair in group:
            around[pair.hub_a].append((-pair.centroid_cosine, pair.hub_b, pair))
            around[pair.hub_b].append((-pair.centroid_cosine, pair.hub_a, pair))
        for neighbours in around.values():
            for *_, pair in sorted(neighbours, key=lambda item: item[:2])[:nearest]:
                reasons[(pair.hub_a, pair.hub_b)].add(BridgeReason.NEAREST_HUB)
        widest = sorted(group, key=lambda pair: (-pair.bridge_gap, pair.hub_a, pair.hub_b))
        for pair in widest[:top_gap]:
            reasons[(pair.hub_a, pair.hub_b)].add(BridgeReason.BRIDGE_GAP)
        covered.extend(
            pair.model_copy(
                update={
                    "reasons": tuple(
                        reason
                        for reason in BridgeReason
                        if reason in reasons[(pair.hub_a, pair.hub_b)]
                    )
                }
            )
            for pair in group
        )
    return covered


def _directions(pair: HubPair) -> Iterator[tuple[int, int, int, int]]:
    """(hub from, hub to, size of the hub from, its pages already linking) both ways."""
    yield pair.hub_a, pair.hub_b, pair.size_a, pair.pages_ab
    yield pair.hub_b, pair.hub_a, pair.size_b, pair.pages_ba


def _source_order(relevance: float, page: PageStructure) -> tuple[float, bool, float, int, str]:
    # Relevance to the other hub first: a link counts by its context. Then the most authority
    # and the fewest outbound links, for more equity per link.
    rank = page.page_rank_percentile
    return (
        -round(relevance, RELEVANCE_DECIMALS),
        rank is None,
        -(rank or 0.0),
        page.outbound,
        page.url,
    )


def bridge_links(
    pairs: Iterable[HubPair],
    pages: Sequence[PageStructure],
    links: Iterable[tuple[str, str]],
    vectors: Mapping[str, npt.ArrayLike],
    targets: Iterable[str],
    non_canonical: Iterable[str],
    keywords: Mapping[str, tuple[str, KeywordRung]],
    *,
    alternatives: int = ALTERNATIVES,
) -> list[BridgeLink]:
    """Ranked page pairs for every direction of a covered pair below the floor, one distinct
    source page per missing page, best first.

    Sources are the hub's pages with a vector, not a non-canonical copy and not already
    linking into the other hub, the most relevant to it first: the best cosine to one of its
    targets, then PageRank percentile, fewer outbound links and url. ``targets`` are the
    candidate targets (indexable, with a vector); a source's targets are the other hub's, not
    non-canonical, by cosine to the source page. Rank 1 is the proposal and up to
    ``alternatives`` more follow it; a target may serve several sources. ``keywords`` gives the
    target's anchor, a page url to (text, rung).
    """
    if not 0 <= alternatives <= ALTERNATIVES:
        raise ValueError(f"alternatives must be between 0 and {ALTERNATIVES}")
    index = _hub_index(pages, links)
    indexable = set(targets)
    copies = set(non_canonical)
    # float32 like the stored vectors: a large tenant touches most of its pages.
    units: dict[str, npt.NDArray[np.float32]] = {}

    def unit(url: str) -> npt.NDArray[np.float32]:
        if url not in units:
            units[url] = _unit(vectors[url]).astype(np.float32)
        return units[url]

    proposed: list[BridgeLink] = []
    for pair in pairs:
        if not pair.reasons:
            continue
        language = pair.language
        for hub_from, hub_to, size, have in _directions(pair):
            slots = floor_pages(size) - have
            if slots <= 0:
                continue
            linked = index.linking.get((language, hub_from, hub_to), set())
            eligible = [
                url
                for url in index.members.get((language, hub_from), ())
                if url in vectors and url not in copies and url not in linked
            ]
            candidates = [
                url
                for url in index.members.get((language, hub_to), ())
                if url in indexable and url in vectors and url not in copies
            ]
            if not eligible or not candidates:
                continue
            matrix = np.stack([unit(url) for url in candidates])
            # Row per eligible source, column per candidate target.
            cosines = np.clip(np.stack([unit(url) for url in eligible]) @ matrix.T, -1.0, 1.0)
            relevance = cosines.max(axis=1)
            chosen = sorted(
                range(len(eligible)),
                key=lambda row: _source_order(float(relevance[row]), index.pages[eligible[row]]),
            )[:slots]
            for slot, row in enumerate(chosen, start=1):
                source_url = eligible[row]
                similarities = cosines[row]
                # Most similar first, url ascending on ties: candidates are in url order.
                best = np.lexsort((np.arange(len(candidates)), -similarities))
                for rank, column in enumerate(best[: 1 + alternatives].tolist(), start=1):
                    target_url = candidates[column]
                    anchor = keywords.get(target_url)
                    proposed.append(
                        BridgeLink(
                            language=language,
                            hub_from=hub_from,
                            hub_to=hub_to,
                            slot=slot,
                            rank=rank,
                            source_url=source_url,
                            target_url=target_url,
                            similarity=float(similarities[column]),
                            source_page_rank_percentile=index.pages[
                                source_url
                            ].page_rank_percentile,
                            anchor_keyword=anchor[0] if anchor else None,
                            anchor_rung=anchor[1] if anchor else None,
                            reasons=pair.reasons,
                        )
                    )
    return sorted(
        proposed,
        key=lambda link: (
            _language_key(link.language),
            link.hub_from,
            link.hub_to,
            link.slot,
            link.rank,
        ),
    )


def components(
    pages: Sequence[PageStructure], pairs: Iterable[HubPair], links: Iterable[BridgeLink] = ()
) -> int:
    """Connected components of the hub graphs, summed over languages. Two hubs are joined when
    their pair meets the floor both ways, counting each proposed link (rank 1) as made."""
    hubs: defaultdict[str | None, set[int]] = defaultdict(set)
    for page in pages:
        if _in_hub(page.hub_id) and page.hub_id is not None:
            hubs[page.language].add(page.hub_id)
    added = Counter((link.language, link.hub_from, link.hub_to) for link in links if link.rank == 1)
    parents: defaultdict[str | None, dict[int, int]] = defaultdict(dict)
    for pair in pairs:
        if all(
            have + added[(pair.language, hub_from, hub_to)] >= floor_pages(size)
            for hub_from, hub_to, size, have in _directions(pair)
        ):
            _union(parents[pair.language], pair.hub_a, pair.hub_b)
    return sum(
        len({_find(parents[language], hub) for hub in in_language})
        for language, in_language in hubs.items()
    )


def bridge_report(
    tenant_id: str,
    pages: Sequence[PageStructure],
    pairs: Sequence[HubPair],
    links: Sequence[BridgeLink],
    *,
    gsc_used: bool,
    started: float,
) -> BridgeReport:
    """``pairs`` carry their reasons and ``links`` are `bridge_links` of them; ``started`` is
    the run's ``time.perf_counter()`` start."""
    made = Counter((link.language, link.hub_from, link.hub_to) for link in links if link.rank == 1)
    below = needed = short = 0
    for pair in pairs:
        if not pair.reasons:
            continue
        for hub_from, hub_to, size, have in _directions(pair):
            slots = floor_pages(size) - have
            if slots > 0:
                below += 1
                needed += slots
                short += made[(pair.language, hub_from, hub_to)] < slots
    return BridgeReport(
        tenant_id=tenant_id,
        floor_share=FLOOR_SHARE,
        hubs=len({page.hub_id for page in pages if _in_hub(page.hub_id)}),
        noise_pages=sum(page.hub_id == HUB_NOISE for page in pages),
        hub_pairs=len(pairs),
        components_before=components(pages, pairs),
        components_after=components(pages, pairs, links),
        directions_below_floor=below,
        links_needed=needed,
        bridge_links=sum(made.values()),
        alternatives=sum(link.rank > 1 for link in links),
        directions_short=short,
        by_reason={
            reason: sum(reason in pair.reasons for pair in pairs) for reason in BridgeReason
        },
        gsc_used=gsc_used,
        seconds=round(time.perf_counter() - started, 3),
        finished_at=datetime.now(UTC),
    )


def summarise_bridges(report: BridgeReport) -> str:
    """A short prose record of one hub-bridge run, for the MLflow run description; hub ids and
    counts only, never page urls."""
    topic = (
        "centroid cosine and GSC query overlap"
        if report.gsc_used
        else "centroid cosine only (no GSC data)"
    )
    lines = [
        f"Hub bridges for tenant {report.tenant_id}: {report.hubs} hubs, {report.noise_pages} "
        f"noise pages outside the guarantee, {report.hub_pairs} hub pairs scored on {topic}.",
        f"Connected means at least {report.floor_share:.0%} of a hub's pages link into the other "
        "hub, both ways.",
        "Covered pairs: "
        + ", ".join(f"{reason.value.lower()} {count}" for reason, count in report.by_reason.items())
        + ".",
        f"{report.directions_below_floor} covered directions below the floor need "
        f"{report.links_needed} links: {report.bridge_links} proposed with {report.alternatives} "
        f"alternatives; {report.directions_short} directions short of eligible pages.",
        f"Components (summed over languages): {report.components_before} before, "
        f"{report.components_after} after the proposed bridges.",
        f"{report.seconds:.1f} s.",
    ]
    return "\n".join(lines)
