"""A planted tenant for the #99 link audit: twenty topic pages linking to each other, an archive
that lists them by title, a long kit-list guide linking to all of them, a sitemap and a paginated
journal page, and at least one link per audit case.

Every link carries what the audit should find on it with #16's scores stored (A2) and without
them (A1): flags, verdict, proposed anchor and fix target; healthy links carry nothing, and links
from or into the sitemap or the paginated page are not audited at all. Context relevance and
anchor-target fit are stored in two tight modes, on topic near 0.8 and off topic near 0.3, and
every keyword's words are unique to it. The data separate cleanly by construction: recovering the
truth proves the plumbing, not the method.

Page i's base links go to pages i+1 .. i+OUT[i]; the first two are anchored with the target's
keyword, the rest with a partial match. Planted links are appended after them, or put first when
their case needs the most equity. The archive is dense with links (a listing) and has the most of
them (index-like); the guide has nearly as many in long copy (index-like, not a listing).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from test_keyword_stage import page_record, url
from voyage_fakes import MODEL, FakeVoyage

from linking_engine.anchor.extraction import Stems
from linking_engine.anchor.generic import is_generic, normalise_anchor
from linking_engine.anchor.scoring import stem_jaccard
from linking_engine.audit.links import AnchorFacts, Proposal
from linking_engine.models import (
    ActionType,
    AuditEdge,
    IssueFlag,
    KeywordRung,
    KeywordSource,
    KeywordTarget,
    LanguageRules,
    Link,
    Page,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from linking_engine.graph.repo import GraphRepo
    from linking_engine.ingest.mongo_repo import MongoRepo

TOPIC_KEYWORDS: Final = (
    "trail shoes", "rain jacket", "camp stove", "dome tent", "sleeping bag",
    "head torch", "water filter", "trekking poles", "sun hat", "wool socks",
    "map case", "ice axe", "fleece vest", "cook pot", "folding chair",
    "climbing rope", "hand warmer", "insect net", "snow shovel", "pocket compass",
)  # fmt: skip
TOPICS: Final = len(TOPIC_KEYWORDS)
# Base links per topic page.
OUT: Final = (3, 4, 5, 3, 3, 4, 5, 4, 3, 4, 4, 5, 3, 4, 4, 3, 4, 5, 3, 4)
# Five identical exact-match anchors into page 7 from sources that are not listings, four into 9.
OVER_OPTIMISED_TARGET: Final = 7
CONTROL_TARGET: Final = 9
SATURATED_SOURCES: Final = (9, 11)

ARCHIVE: Final = "/archive"
GUIDE: Final = "/guides/kit-list"
SITEMAP: Final = "/sitemap"
PAGINATED: Final = "/journal/page/2"
NOISE: Final = "/misc/offcuts"
NOISE_KEYWORDED: Final = "/misc/buckles"
NOINDEX: Final = "/gear/clearance"
BROKEN: Final = "/gear/retired"
REDIRECTED: Final = "/gear/moved"
CANONICAL: Final = "/gear/dry-sack"
COPY: Final = "/gear/dry-sack-print"
PLACEHOLDER: Final = "/gear/unlisted"
DUPLICATE_GROUP: Final = "dup-dry-sack"
DIMENSION: Final = 2048
TENANT_SUFFIX: Final = "Acme"
# Long copy for the guide, in words no keyword uses.
GUIDE_FILLER: Final = " ".join(["Plan the route and check the weather before you set out."] * 40)

ON_TOPIC: Final = (0.74, 0.86)
OFF_TOPIC: Final = (0.24, 0.34)

GENERIC, MISALIGNED, OFF, OVER, WASTED = (
    IssueFlag.GENERIC,
    IssueFlag.MISALIGNED,
    IssueFlag.OFF_TOPIC,
    IssueFlag.OVER_OPTIMISED,
    IssueFlag.WASTED_EQUITY,
)
FIX, REANCHOR, REMOVE = ActionType.FIX, ActionType.REANCHOR, ActionType.REMOVE


def topic(index: int) -> str:
    return "/gear/" + TOPIC_KEYWORDS[index].replace(" ", "-")


@dataclass(frozen=True, slots=True)
class PlantedPage:
    path: str
    percentile: float
    keyword: str | None = None
    status: int = 200
    indexable: bool = True
    hub: int | None = None
    duplicate_group: str | None = None
    canonical: bool | None = None
    body: str = ""

    @property
    def crawled_ok(self) -> bool:
        return self.status == 200

    @property
    def words(self) -> int:
        return len(self.body.split())


@dataclass(frozen=True, slots=True)
class Expected:
    flags: frozenset[IssueFlag] = frozenset()
    verdict: ActionType | None = None
    proposal: str | None = None


HEALTHY: Final = Expected()


@dataclass(frozen=True, slots=True)
class PlantedLink:
    case: str
    source: str
    target: str
    position: int
    anchor: str
    sentence: str
    context: float | None
    fit: float | None
    a2: Expected
    a1: Expected
    follow: bool = True
    fix_target: str | None = None
    # Why the audit leaves the link out ("sitemap" or "paginated"); None when it is audited.
    skipped: str | None = None

    @property
    def generic(self) -> bool:
        return is_generic(self.anchor)

    @property
    def unverified(self) -> bool:
        return self.target == PLACEHOLDER

    def expected(self, *, a2: bool) -> Expected:
        return self.a2 if a2 else self.a1


@dataclass(frozen=True, slots=True)
class _Spec:
    case: str
    target: str
    anchor: str
    sentence: str
    context: str | None
    fit: str | None
    a2: Expected
    a1: Expected
    follow: bool = True
    first: bool = False
    fix_target: str | None = None
    skipped: str | None = None


def _exact(target: int) -> Expected:
    if target == OVER_OPTIMISED_TARGET:
        return Expected(frozenset({OVER}), REANCHOR)
    return HEALTHY


def _base(source: int) -> list[_Spec]:
    specs = []
    for step in range(1, OUT[source] + 1):
        target = (source + step) % TOPICS
        keyword = TOPIC_KEYWORDS[target]
        anchor, sentence = (
            (keyword, f"Our {keyword} passed every field test."),
            (keyword, f"We restocked the {keyword} this week."),
            (f"best {keyword}", f"Compare the best {keyword} side by side."),
            (f"{keyword} range", f"The {keyword} range grew again."),
            (f"{keyword} reviews", f"Owners left {keyword} reviews for us."),
        )[step - 1]
        expected = _exact(target) if step <= 2 else HEALTHY
        specs.append(
            _Spec("healthy", topic(target), anchor, sentence, "on", "on", expected, expected)
        )
    return specs


def _same(expected: Expected) -> dict[str, Expected]:
    return {"a2": expected, "a1": expected}


# (source page, specs): the planted cases, each with its A2 and A1 truth.
_PLANTED: Final[tuple[tuple[int, _Spec], ...]] = (
    # Generic anchors: REANCHOR, proposing the target's keyword where the copy writes it.
    (2, _Spec("generic", topic(10), "click here", "To keep charts flat, click here for options.",
              "on", None, **_same(Expected(frozenset({GENERIC}), REANCHOR, "map case")))),
    (3, _Spec("generic", topic(12), "read more", "For cold evenings, read more about layers.",
              "on", None, **_same(Expected(frozenset({GENERIC}), REANCHOR)))),
    (4, _Spec("generic", topic(13), "learn more", "Before the first dinner outdoors, learn more.",
              "on", None, **_same(Expected(frozenset({GENERIC}), REANCHOR, "cook pot")))),
    # A synonym that shares no word with the keyword but fits the target: only A1, which has no
    # fit, calls it misaligned.
    (6, _Spec("synonym", topic(14), "portable seat", "Every portable seat we sell packs small.",
              "on", "on", a2=HEALTHY,
              a1=Expected(frozenset({MISALIGNED}), REANCHOR, "folding chair"))),
    # Misaligned in both: no shared word and, with A2, a weak fit.
    (5, _Spec("misaligned", topic(11), "winter tools", "Pack the right winter tools for the ridge.",
              "on", "off", **_same(Expected(frozenset({MISALIGNED}), REANCHOR, "ice axe")))),
    (7, _Spec("misaligned", topic(15), "static line", "Tie in with a static line before the traverse.",
              "on", "off", **_same(Expected(frozenset({MISALIGNED}))))),
    # A partial anchor with a weak fit: A2 reanchors it to the keyword the copy writes.
    (0, _Spec("weak-fit", topic(18), "snow gear", "Keep some snow gear by the door.",
              "on", "off", a2=Expected(frozenset(), REANCHOR, "snow shovel"), a1=HEALTHY)),
    # Off topic alone, on a source neither saturated nor spending much equity: no verdict.
    (8, _Spec("off-topic", topic(16), "garden gnomes", "Our neighbours also sell garden gnomes.",
              "off", "off", a2=Expected(frozenset({OFF, MISALIGNED})),
              a1=Expected(frozenset({MISALIGNED})))),
    # Off topic and misaligned on a saturated source: REMOVE.
    (9, _Spec("healthy", topic(0), "trail shoes guide", "Read our trail shoes guide first.",
              "on", "on", **_same(HEALTHY))),
    (9, _Spec("healthy", topic(1), "rain jacket care", "Wash with our rain jacket care tips.",
              "on", "on", **_same(HEALTHY))),
    (9, _Spec("off-topic", topic(17), "birthday cards", "The shop next door has birthday cards.",
              "off", "off", a2=Expected(frozenset({OFF, MISALIGNED}), REMOVE),
              a1=Expected(frozenset({MISALIGNED})))),
    # Off topic and misaligned, first on a high-equity source, into a page that needs none of
    # it: REMOVE.
    (19, _Spec("off-topic", topic(8), "kitchen scales", "We weigh parcels on kitchen scales.",
               "off", "off", first=True,
               a2=Expected(frozenset({OFF, MISALIGNED, WASTED}), REMOVE),
               a1=Expected(frozenset({MISALIGNED})))),
    # Links into noise pages: wasted when first on a high-equity source, fine otherwise.
    (18, _Spec("noise", NOISE_KEYWORDED, "spare buckles", "Loose spare buckles sit in the bin.",
               "on", "on", first=True, **_same(Expected(frozenset({WASTED}))))),
    (1, _Spec("noise", NOISE, "offcut bin", "Scraps go to the offcut bin.",
              "on", "on", **_same(HEALTHY))),
    # Technical health: FIX.
    (17, _Spec("noindex", NOINDEX, "clearance deals", "End of season clearance deals are live.",
               "on", "on", first=True,
               **_same(Expected(frozenset({IssueFlag.NOINDEX_TARGET, WASTED}), FIX)))),
    (12, _Spec("broken", BROKEN, "retired boots", "The retired boots page is gone.",
               None, None, **_same(Expected(frozenset({IssueFlag.BROKEN}), FIX)))),
    (13, _Spec("redirected", REDIRECTED, "moved kettles", "See the moved kettles page.",
               None, None, **_same(Expected(frozenset({IssueFlag.REDIRECTED}), FIX)))),
    (14, _Spec("nofollow", topic(4), "sleeping bag", "A sponsored sleeping bag review.",
               "on", "on", follow=False, **_same(Expected(frozenset({IssueFlag.NOFOLLOW}), FIX)))),
    (15, _Spec("non-canonical", COPY, "dry sack", "The dry sack keeps kit dry.",
               "on", "on", fix_target=CANONICAL, **_same(Expected(frozenset(), FIX)))),
    (16, _Spec("canonical", CANONICAL, "dry sack", "Every dry sack is seam taped.",
               "on", "on", **_same(HEALTHY))),
    # A target never crawled: unverified, no scores, flags or verdict.
    (16, _Spec("placeholder", PLACEHOLDER, "unlisted gear", "The unlisted gear page is new.",
               None, None, **_same(HEALTHY))),
    # Three more identical exact-match anchors make five into the over-optimised target; two
    # more make four into the control.
    (10, _Spec("over-optimised", topic(OVER_OPTIMISED_TARGET), "trekking poles",
               "Light trekking poles save knees.", "on", "on",
               **_same(Expected(frozenset({OVER}), REANCHOR)))),
    (11, _Spec("over-optimised", topic(OVER_OPTIMISED_TARGET), "trekking poles",
               "Carbon trekking poles weigh little.", "on", "on",
               **_same(Expected(frozenset({OVER}), REANCHOR)))),
    (12, _Spec("over-optimised", topic(OVER_OPTIMISED_TARGET), "trekking poles",
               "Folding trekking poles pack flat.", "on", "on",
               **_same(Expected(frozenset({OVER}), REANCHOR)))),
    (13, _Spec("control", topic(CONTROL_TARGET), "wool socks", "Merino wool socks stay warm.",
               "on", "on", **_same(HEALTHY))),
    (15, _Spec("control", topic(CONTROL_TARGET), "wool socks", "Thick wool socks prevent blisters.",
               "on", "on", **_same(HEALTHY))),
    # Off topic on a saturated source, but the anchor shares a stem with the keyword: a flag,
    # never REMOVE.
    (11, _Spec("off-topic-aligned", topic(16), "warmer weather deals",
               "Sign up for warmer weather deals.", "off", "off",
               a2=Expected(frozenset({OFF})), a1=HEALTHY)),
    # Last on its source, so little equity reaches the non-indexable page.
    (11, _Spec("noindex", NOINDEX, "clearance deals", "More clearance deals arrive on Fridays.",
               "on", "on", **_same(Expected(frozenset({IssueFlag.NOINDEX_TARGET}), FIX)))),
    # Links into a sitemap or a paginated page are not audited.
    (1, _Spec("sitemap", SITEMAP, "site map", "Browse the site map for everything.",
              "on", "on", **_same(HEALTHY), skipped="sitemap")),
    (3, _Spec("paginated", PAGINATED, "older journal entries", "Find older journal entries here.",
              "on", "on", **_same(HEALTHY), skipped="paginated")),
)  # fmt: skip

# Sentences a source's copy writes outside any link, so the ladder can find the target's keyword.
PROPOSAL_SENTENCES: Final = {
    2: "A map case keeps charts dry.",
    4: "A cook pot with a lid boils faster.",
    6: "A folding chair saves your back.",
    5: "An ice axe is essential on steep slopes.",
    0: "A snow shovel clears the path.",
}


def _archive() -> list[_Spec]:
    """Every topic page by title, one per line, after an off-topic link that wastes equity, then
    a generic link and one into a broken page. A listing: flags without REANCHOR or REMOVE, and
    its exact titles never count as over-optimisation; FIX still applies."""
    specs = [
        _Spec("listing", topic(8), "car wax", "Old car wax from the catalogue.", "off", "off",
              a2=Expected(frozenset({OFF, MISALIGNED, WASTED})),
              a1=Expected(frozenset({MISALIGNED}))),
    ]  # fmt: skip
    for target, keyword in enumerate(TOPIC_KEYWORDS):
        title = keyword.title()
        specs.append(
            _Spec("healthy", topic(target), title, f"{title}.", "on", "on", **_same(HEALTHY))
        )
    specs += [
        _Spec("listing", topic(19), "See more", "See more.", "on", None,
              **_same(Expected(frozenset({GENERIC})))),
        _Spec("listing", BROKEN, "Retired Boots", "Retired Boots.", None, None,
              **_same(Expected(frozenset({IssueFlag.BROKEN}), FIX))),
    ]  # fmt: skip
    return specs


def _guide() -> list[_Spec]:
    """An off-topic link first, then a checklist link to every topic page, in long copy: as many
    links as the archive gives an index-like page, which is never REMOVE; not a listing."""
    specs = [
        _Spec("index-like", topic(8), "tyre levers", "Pack tyre levers for the bike.", "off",
              "off", a2=Expected(frozenset({OFF, MISALIGNED, WASTED})),
              a1=Expected(frozenset({MISALIGNED}))),
    ]  # fmt: skip
    for target, keyword in enumerate(TOPIC_KEYWORDS):
        specs.append(
            _Spec("healthy", topic(target), f"{keyword} checklist",
                  f"Tick the {keyword} checklist item.", "on", "on", **_same(HEALTHY))
        )  # fmt: skip
    return specs


def _hub(targets: range, skipped: str) -> list[_Spec]:
    return [
        _Spec(skipped, topic(t), TOPIC_KEYWORDS[t].title(), f"{TOPIC_KEYWORDS[t].title()} page.",
              "on", "on", **_same(HEALTHY), skipped=skipped)
        for t in targets
    ]  # fmt: skip


def _percentile(index: int) -> float:
    return {8: 0.06, 17: 0.85, 18: 0.9, 19: 0.95}.get(index, round(0.1 + 0.035 * index, 3))


def _build() -> tuple[tuple[PlantedPage, ...], tuple[PlantedLink, ...]]:
    specs: dict[str, list[_Spec]] = {topic(i): _base(i) for i in range(TOPICS)}
    for source, spec in _PLANTED:
        planted = specs[topic(source)]
        if spec.first:
            planted.insert(0, spec)
        else:
            planted.append(spec)
    specs[ARCHIVE] = _archive()
    specs[GUIDE] = _guide()
    specs[SITEMAP] = _hub(range(5), "sitemap")
    specs[PAGINATED] = _hub(range(5, 8), "paginated")

    rng = np.random.default_rng(99)

    def draw(kind: str | None) -> float | None:
        if kind is None:
            return None
        low, high = ON_TOPIC if kind == "on" else OFF_TOPIC
        return round(float(rng.uniform(low, high)), 4)

    links = tuple(
        PlantedLink(
            case=spec.case,
            source=source,
            target=spec.target,
            position=position,
            anchor=spec.anchor,
            sentence=spec.sentence,
            context=draw(spec.context),
            fit=draw(spec.fit),
            a2=spec.a2,
            a1=spec.a1,
            follow=spec.follow,
            fix_target=spec.fix_target,
            skipped=spec.skipped,
        )
        for source, source_specs in specs.items()
        for position, spec in enumerate(source_specs)
    )

    def body(path: str, opening: str, *extra: str) -> str:
        sentences = [link.sentence for link in links if link.source == path]
        index = next((i for i in range(TOPICS) if topic(i) == path), None)
        proposal = [PROPOSAL_SENTENCES[index]] if index in PROPOSAL_SENTENCES else []
        return " ".join([opening, *sentences, *proposal, *extra])

    pages = [
        PlantedPage(
            topic(i),
            _percentile(i),
            keyword,
            hub=i // 5,
            body=body(topic(i), f"Notes from the {keyword} workshop."),
        )
        for i, keyword in enumerate(TOPIC_KEYWORDS)
    ]
    pages += [
        PlantedPage(ARCHIVE, 0.92, hub=-1, body=body(ARCHIVE, "Everything we listed.")),
        PlantedPage(GUIDE, 0.88, hub=0, body=body(GUIDE, "The full kit list.", GUIDE_FILLER)),
        PlantedPage(SITEMAP, 0.5, hub=-1, body=body(SITEMAP, "All pages.")),
        PlantedPage(PAGINATED, 0.2, hub=-1, body=body(PAGINATED, "Journal, older entries.")),
        PlantedPage(NOISE, 0.02, hub=-1, body="Odd parts and offcuts."),
        PlantedPage(NOISE_KEYWORDED, 0.03, "spare buckles", hub=-1, body="Buckles for straps."),
        PlantedPage(NOINDEX, 0.04, "clearance deals", indexable=False, hub=1,
                    body="Deals change weekly."),
        PlantedPage(BROKEN, 0.01, status=404),
        PlantedPage(REDIRECTED, 0.015, status=301),
        PlantedPage(CANONICAL, 0.3, "dry sack", hub=2, duplicate_group=DUPLICATE_GROUP,
                    canonical=True, body="Sizes and seams of every sack."),
        PlantedPage(COPY, 0.05, hub=2, duplicate_group=DUPLICATE_GROUP, canonical=False,
                    body="Sizes and seams of every sack."),
    ]  # fmt: skip
    return tuple(pages), links


PAGES, LINKS = _build()
PAGE_BY_PATH: Final = {page.path: page for page in PAGES}
# The links the audit scores, in the order it reads them.
AUDITED: Final = tuple(link for link in LINKS if link.skipped is None)


def link_key(link: PlantedLink) -> tuple[str, int]:
    """(source url, position): how the audit identifies an edge."""
    return url(link.source), link.position


def audit_edges(*, a2: bool = True) -> list[AuditEdge]:
    """Every planted link as the audit reads it from the graph, skipped ones included; without
    #16's scores for A1."""
    edges = []
    for link in LINKS:
        source = PAGE_BY_PATH[link.source]
        target = PAGE_BY_PATH.get(link.target)
        edges.append(
            AuditEdge(
                source_url=url(link.source),
                position=link.position,
                target_url=url(link.target),
                anchor_text=link.anchor,
                is_follow=link.follow,
                context_relevance=link.context if a2 else None,
                anchor_target_fit=link.fit if a2 else None,
                source_language="en",
                source_page_rank_percentile=source.percentile,
                source_word_count=source.words,
                target_placeholder=target is None,
                target_status_code=None if target is None else target.status,
                target_indexable=None if target is None else target.indexable,
                target_hub_id=None if target is None else target.hub,
                target_page_rank_percentile=None if target is None else target.percentile,
                target_canonical_url=url(CANONICAL) if link.target == COPY else None,
            )
        )
    return edges


def anchor_facts(links: Sequence[PlantedLink] = AUDITED) -> list[AnchorFacts]:
    """Each planted anchor measured as the pipeline does, against its target's one keyword."""
    stems = Stems("en")
    facts = []
    for link in links:
        target = PAGE_BY_PATH.get(link.target)
        key = normalise_anchor(link.anchor) or None
        if target is None or target.keyword is None or key is None:
            facts.append(AnchorFacts(key, link.generic, False, None))
            continue
        jaccard = stem_jaccard(link.anchor, target.keyword, stems)
        facts.append(AnchorFacts(key, link.generic, key == target.keyword, jaccard))
    return facts


def planted_proposals() -> dict[tuple[str, str], list[Proposal]]:
    """What the ladder finds in the planted copy: the target's keyword, where it is written."""
    stems = Stems("en")
    found: dict[tuple[str, str], list[Proposal]] = {}
    for link in AUDITED:
        phrase = link.a2.proposal or link.a1.proposal
        keyword = PAGE_BY_PATH[link.target].keyword if link.target in PAGE_BY_PATH else None
        if phrase is None or keyword is None:
            continue
        found[(url(link.source), url(link.target))] = [
            Proposal(
                phrase, normalise_anchor(phrase), stem_jaccard(phrase, keyword, stems), None, None
            )
        ]
    return found


def _key(text: str) -> str:
    return " ".join(text.casefold().split())


def hashed(text: str) -> list[float]:
    """A vector drawn from the case-folded text: case variants of one phrase embed alike and
    any two other texts are nearly orthogonal."""
    seed = int.from_bytes(hashlib.sha256(_key(text).encode()).digest()[:8], "big")
    vector: list[float] = np.random.default_rng(seed).normal(size=DIMENSION).tolist()
    return vector


def respond(texts: Sequence[str]) -> list[list[float]]:
    return [hashed(text) for text in texts]


def voyage() -> FakeVoyage:
    return FakeVoyage(dimension=DIMENSION, respond=respond)


def _title(page: PlantedPage) -> str | None:
    if page.keyword is None:
        return None
    return f"{page.keyword.title()} | {TENANT_SUFFIX}"


async def seed_link_audit(
    graph: GraphRepo, mongo: MongoRepo, tenant: str, *, scored: bool = True
) -> None:
    """The planted tenant in both stores: crawl, graph analytics and ranked keywords, and with
    ``scored`` also what embed-links and score-links leave on the edges (A2)."""
    await mongo.set_language_rules(tenant, LanguageRules(default_language="en"))
    await mongo.write_pages(
        tenant,
        [page_record(p.path, p.status, _title(p), None, p.body, "en") for p in PAGES],
        [],
    )
    await graph.upsert_pages(
        tenant,
        [
            Page(
                url=url(p.path),
                status_code=p.status,
                is_indexable=p.indexable,
                language="en",
                word_count=p.words,
            )
            for p in PAGES
        ],
    )
    await graph.upsert_placeholders(tenant, [url(PLACEHOLDER)])
    sources = sorted({link.source for link in LINKS})
    await graph.replace_links(
        tenant,
        [url(source) for source in sources],
        [
            Link(
                source_url=url(link.source),
                target_url=url(link.target),
                position=link.position,
                anchor_text=link.anchor,
                surrounding_text=link.sentence,
                is_follow=link.follow,
            )
            for link in LINKS
        ],
    )
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.pageRank = row.percentile, p.pageRankPercentile = row.percentile, "
        "p.hubId = row.hub, p.duplicateGroup = row.group, p.isCanonical = row.canonical",
        t=tenant,
        rows=[
            {
                "url": url(p.path),
                "percentile": p.percentile,
                "hub": p.hub,
                "group": p.duplicate_group,
                "canonical": p.canonical,
            }
            for p in PAGES
        ],
    )
    strategic = KeywordSource.CLIENT_STRATEGIC
    await graph.replace_keyword_targets(
        tenant,
        strategic,
        [
            KeywordTarget(
                url=url(p.path),
                text=p.keyword,
                language="en",
                source=strategic,
                rung=KeywordRung.STRATEGIC,
            )
            for p in PAGES
            if p.keyword is not None
        ],
    )
    if not scored:
        return
    await graph._auto(
        "UNWIND $rows AS row MATCH (p:Page {tenantId: $t, url: row.url}) "
        "SET p.content_embedding = row.vec, p.embeddingModel = $model",
        t=tenant,
        model=MODEL,
        rows=[{"url": url(p.path), "vec": hashed(p.body)} for p in PAGES if p.crawled_ok],
    )
    await graph._auto(
        "UNWIND $rows AS row "
        "MATCH (:Page {tenantId: $t, url: row.source})-[r:LINKS_TO {position: row.position}]->() "
        "SET r.anchorKey = row.key, r.anchorGeneric = row.generic, "
        "r.contextRelevance = row.context, r.anchorTargetFit = row.fit",
        t=tenant,
        rows=[
            {
                "source": url(link.source),
                "position": link.position,
                "key": normalise_anchor(link.anchor),
                "generic": link.generic,
                "context": link.context,
                "fit": link.fit,
            }
            for link in LINKS
        ],
    )
