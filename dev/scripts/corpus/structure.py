"""
Deterministic site structure: pages, body links, GSC metrics.

No LLM, no network. Given a seed this produces the identical corpus every time,
which is what makes the assertion suite meaningful.

Content is generated separately — see content.py. This module decides *what*
each page is about and whether it should contain its own head term; the content
layer only has to honour that.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from corpus.taxonomy import (
    BRIDGE_GAPS, FOOTER_TARGETS, GENERIC_ANCHORS, NAV_TARGETS,
    NOISE_SUBJECTS, NO_HEAD_TERM_RATE, PAGE_TYPE_IMPRESSION_BASE,
    PAGE_TYPE_POSITION_SCALE, PILLAR_MISMATCH, P_CROSS_BASELINE,
    P_CROSS_BRIDGE_GAP, P_CROSS_WELL_CONNECTED, TOPICS, WELL_CONNECTED, ctr_at,
)


@dataclass
class Page:
    url: str
    topic: str | None
    subtopic: str | None
    page_type: str
    title: str
    h1: str
    word_count: int
    crawl_depth: int
    include_head_term: bool = True      # drives the extraction ladder
    html: str = ""                      # filled by the content layer
    body_text: str = ""                 # html stripped, what gets embedded
    embedding: list[float] = field(default_factory=list)
    http_status: int = 200
    is_indexable: bool = True
    published_at: str = ""
    lifecycle_stage: str = "ESTABLISHED"
    content_hash: str = ""
    planted: list[str] = field(default_factory=list)

    @property
    def content_key(self) -> str:
        """Cache key for generated content. Deliberately excludes url, so two
        pages with identical spec share a cache entry."""
        parts = [self.topic or "noise", self.subtopic or "-", self.page_type,
                 str(self.include_head_term), self.title]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


@dataclass
class Link:
    source: str
    target: str
    anchor_text: str
    anchor_type: str
    link_position: str = "body"   # body links only — nav/footer never captured
    is_follow: bool = True
    target_http_status: int = 200
    surrounding_text: str = ""
    planted: list[str] = field(default_factory=list)


def _slug(text: str) -> str:
    return text.replace(" ", "-").replace("/", "-").lower()


# ── pages ───────────────────────────────────────────────────────────────────

def build_pages(n_pages: int, rng: random.Random) -> tuple[list[Page], dict[str, list[Page]]]:
    topics = list(TOPICS)
    pages: list[Page] = []
    by_topic: dict[str, list[Page]] = {t: [] for t in topics}

    # pillars — two deliberately displaced from their cluster centroid
    for t in topics:
        sub = next(iter(TOPICS[t]["subtopics"]))
        planted = ["DECLARED_PILLAR"]
        if t in PILLAR_MISMATCH:
            planted.append("PILLAR_MISMATCH")
        p = Page(
            url=f"/{t}", topic=t, subtopic=sub, page_type="PILLAR",
            title=TOPICS[t]["head"].title(), h1=TOPICS[t]["head"].title(),
            word_count=rng.randint(2200, 3400), crawl_depth=1,
            published_at=str(date(2024, 1, 1) + timedelta(days=rng.randint(0, 180))),
            planted=planted,
        )
        pages.append(p)
        by_topic[t].append(p)

    n_noise = max(6, int(n_pages * 0.04))
    n_spokes = n_pages - len(topics) - n_noise - 5

    for i in range(n_spokes):
        t = rng.choice(topics)
        sub = rng.choice(list(TOPICS[t]["subtopics"]))
        term = rng.choice(TOPICS[t]["subtopics"][sub])
        page_type = (
            "PRODUCT" if TOPICS[t]["spread"] < 0.25 and rng.random() < 0.5
            else rng.choices(["ARTICLE", "PRODUCT", "CATEGORY"], weights=[7, 2, 1])[0]
        )
        include_head = rng.random() > NO_HEAD_TERM_RATE
        p = Page(
            url=f"/{t}/{sub}/{_slug(term)}-{i}",
            topic=t, subtopic=sub, page_type=page_type,
            title=f"{term.title()} Guide", h1=term.title(),
            word_count=rng.randint(700, 2600), crawl_depth=rng.randint(2, 5),
            include_head_term=include_head,
            published_at=str(date(2024, 1, 1) + timedelta(days=rng.randint(0, 720))),
            planted=[] if include_head else ["NO_HEAD_TERM"],
        )
        pages.append(p)
        by_topic[t].append(p)

    # noise: no topic, but DOES receive body links
    for slug, _ in (NOISE_SUBJECTS * 4)[:n_noise]:
        pages.append(Page(
            url=f"/{slug}-{rng.randint(100, 999)}", topic=None, subtopic=None,
            page_type="ARTICLE", title=slug.replace("-", " ").title(),
            h1=slug.replace("-", " ").title(),
            word_count=rng.randint(150, 500), crawl_depth=rng.randint(2, 6),
            published_at=str(date(2023, 6, 1) + timedelta(days=rng.randint(0, 900))),
            planted=["TRUE_NOISE"],
        ))

    # utility pages — reachable via nav on a real site, orphans in the body graph
    for slug, title, _ in NAV_TARGETS:
        pages.append(Page(
            url=f"/{slug}", topic=None, subtopic=None, page_type="CATEGORY",
            title=title, h1=title, word_count=rng.randint(200, 700), crawl_depth=1,
            published_at="2023-01-15", planted=["NAV_TARGET", "TRUE_NOISE"],
        ))
    for slug, title in FOOTER_TARGETS:
        pages.append(Page(
            url=f"/{slug}", topic=None, subtopic=None, page_type="ARTICLE",
            title=title, h1=title, word_count=rng.randint(150, 400), crawl_depth=1,
            published_at="2023-01-15", planted=["FOOTER_TARGET", "TRUE_NOISE"],
        ))

    # orphan NEW pages with priority-5 keywords — must surface as ADD_LINK
    for t in topics[:3]:
        sub = rng.choice(list(TOPICS[t]["subtopics"]))
        p = Page(
            url=f"/{t}/new-{t}-2026", topic=t, subtopic=sub, page_type="ARTICLE",
            title=f"New {TOPICS[t]['head'].title()} Guide 2026",
            h1=f"New {TOPICS[t]['head'].title()} Guide",
            word_count=1600, crawl_depth=3, published_at=str(date(2026, 7, 1)),
            lifecycle_stage="NEW", planted=["ORPHAN_NEW_STRATEGIC"],
        )
        pages.append(p)
        by_topic[t].append(p)

    # cannibalisation: two pages sharing a primary keyword with their pillar
    for t in ("hydraulic", "pressbrake"):
        sub = next(iter(TOPICS[t]["subtopics"]))
        p = Page(
            url=f"/{t}/duplicate-intent", topic=t, subtopic=sub, page_type="ARTICLE",
            title=f"{TOPICS[t]['head'].title()} Overview",
            h1=f"{TOPICS[t]['head'].title()} Overview",
            word_count=1400, crawl_depth=3, published_at="2025-04-02",
            planted=["CANNIBALISATION"],
        )
        pages.append(p)
        by_topic[t].append(p)

    return pages, by_topic


# ── links ───────────────────────────────────────────────────────────────────

def build_links(
    pages: list[Page], by_topic: dict[str, list[Page]], rng: random.Random
) -> tuple[list[Link], dict[str, int]]:
    topics = list(TOPICS)
    links: list[Link] = []
    orphans = {p.url for p in pages if "ORPHAN_NEW_STRATEGIC" in p.planted}
    topical = [p for p in pages if p.topic]
    counts = dict.fromkeys(
        ("generic", "off_topic", "nofollow", "over_optimised",
         "healthy", "bridge_present", "noise_inbound"), 0)

    gap_pairs = {tuple(sorted(x)) for x in BRIDGE_GAPS}
    conn_pairs = {tuple(sorted(x)) for x in WELL_CONNECTED}

    def cross_allowed(a: str, b: str) -> bool:
        pair = tuple(sorted((a, b)))
        if pair in gap_pairs:
            return rng.random() < P_CROSS_BRIDGE_GAP
        if pair in conn_pairs:
            return rng.random() < P_CROSS_WELL_CONNECTED
        return rng.random() < P_CROSS_BASELINE

    for src in topical:
        same = [p for p in by_topic[src.topic] if p.url != src.url]
        k = rng.randint(8, 16) if src.page_type == "PILLAR" else rng.randint(2, 6)
        targets = rng.sample(same, k=min(len(same), k))

        for other in [t for t in topics if t != src.topic]:
            if cross_allowed(src.topic, other):
                targets.append(rng.choice(by_topic[other]))
                if tuple(sorted((src.topic, other))) in gap_pairs:
                    counts["bridge_present"] += 1

        src_head = TOPICS[src.topic]["head"].lower() if src.topic else ""

        for tgt in targets:
            if tgt.url in orphans:
                continue
            roll = rng.random()
            head = TOPICS[tgt.topic]["head"]

            # The anchor is rendered into the SOURCE page's body. If that page
            # is planted as NO_HEAD_TERM, an anchor containing its own head term
            # would silently break the fixture — reroll into a safe bucket.
            head_unsafe = (not src.include_head_term
                           and src_head in head.lower())
            if head_unsafe and 0.26 <= roll < 0.62:
                roll = 0.7   # force NATURAL, which never contains the head term

            if roll < 0.18:
                anchor, atype, tag = rng.choice(GENERIC_ANCHORS), "GENERIC", "GENERIC_ANCHOR"
                counts["generic"] += 1
            elif roll < 0.26:
                # off-topic body link: no topical justification, dilutes equity
                other_t = rng.choice([t for t in topics if t != tgt.topic])
                other_sub = rng.choice(list(TOPICS[other_t]["subtopics"]))
                anchor = rng.choice(TOPICS[other_t]["subtopics"][other_sub])
                atype, tag = "PARTIAL", "OFF_TOPIC"
                counts["off_topic"] += 1
            elif roll < 0.30:
                anchor, atype, tag = head, "EXACT", "NOFOLLOW"
                counts["nofollow"] += 1
            elif roll < 0.44:
                anchor, atype, tag = head, "EXACT", "HEALTHY"
                counts["healthy"] += 1
            elif roll < 0.62:
                anchor = rng.choice(TOPICS[tgt.topic]["subtopics"][tgt.subtopic])
                atype, tag = "PARTIAL", "HEALTHY"
                counts["healthy"] += 1
            else:
                anchor = f"{rng.choice(TOPICS[tgt.topic]['verbs'])} guidance"
                atype, tag = "NATURAL", "HEALTHY"
                counts["healthy"] += 1

            links.append(Link(
                source=src.url, target=tgt.url, anchor_text=anchor,
                anchor_type=atype, is_follow=(tag != "NOFOLLOW"),
                target_http_status=tgt.http_status, planted=[tag],
            ))

    # noise pages receive genuine body links — someone did reference them.
    # NAV_TARGETS and FOOTER_TARGETS deliberately receive none.
    for p in pages:
        if "TRUE_NOISE" not in p.planted:
            continue
        if any(k in p.planted for k in ("NAV_TARGET", "FOOTER_TARGET")):
            continue
        for src in rng.sample(topical, k=rng.randint(3, 6)):
            links.append(Link(
                source=src.url, target=p.url, anchor_text=p.title.lower(),
                anchor_type="BRANDED", planted=["NOISE_INBOUND"],
            ))
            counts["noise_inbound"] += 1

    # over-optimised target: many identical exact-match inbound anchors
    victim = by_topic["pressbrake"][1]
    for src in rng.sample(by_topic["pressbrake"], k=12):
        if src.url == victim.url:
            continue
        # same reason as the reroll above: this anchor is the head term, and it
        # renders into the source page's body
        if not src.include_head_term:
            continue
        links.append(Link(
            source=src.url, target=victim.url,
            anchor_text=TOPICS["pressbrake"]["head"], anchor_type="EXACT",
            planted=["OVER_OPTIMISED"],
        ))
        counts["over_optimised"] += 1

    return links, counts


# ── GSC ─────────────────────────────────────────────────────────────────────

def build_queries(
    pages: list[Page], by_topic: dict[str, list[Page]], rng: random.Random
) -> list[dict]:
    """
    Lognormal impressions and exponential position — closer to real GSC
    distributions than uniform scaling, and the heavy tail matters because
    opportunity_value = impressions × (CTR@1 − CTR@current) is dominated by it.
    """
    queries: list[dict] = []
    topical = [p for p in pages if p.topic]

    for p in topical:
        if "ORPHAN_NEW_STRATEGIC" in p.planted:
            continue  # new pages have no GSC history
        imp_base = PAGE_TYPE_IMPRESSION_BASE[p.page_type]
        pos_scale = PAGE_TYPE_POSITION_SCALE[p.page_type]

        terms = TOPICS[p.topic]["subtopics"][p.subtopic] + [TOPICS[p.topic]["head"]]
        for term in terms:
            position = round(min(100.0, float(np.random.exponential(pos_scale)) + 1.0), 1)
            impressions = max(20, int(np.random.lognormal(mean=imp_base, sigma=1.2)))
            impressions = int(impressions / max(1, p.crawl_depth - 1))
            ctr = float(np.clip(ctr_at(position) + np.random.normal(0, 0.003), 0.0005, 0.45))
            queries.append({
                "query": term, "url": p.url, "impressions": impressions,
                "clicks": int(round(impressions * ctr)),
                "position": position, "ctr": round(ctr, 4),
                "search_volume": int(impressions * rng.uniform(1.2, 3.0)),
                "topic": p.topic, "subtopic": p.subtopic,
            })

    # Shared demand across bridge-gap pairs: queries overlap, links do not.
    for a, b in BRIDGE_GAPS:
        shared = [
            f"{TOPICS[a]['head'].split()[0]} {TOPICS[b]['head'].split()[-1]}",
            f"{TOPICS[b]['head'].split()[0]} for {TOPICS[a]['head'].split()[-1]}",
        ]
        for term in shared:
            for topic in (a, b):
                for p in rng.sample(by_topic[topic], k=min(6, len(by_topic[topic]))):
                    position = round(rng.uniform(4, 18), 1)
                    impressions = rng.randint(400, 2500)
                    ctr = ctr_at(position)
                    queries.append({
                        "query": term, "url": p.url, "impressions": impressions,
                        "clicks": int(impressions * ctr), "position": position,
                        "ctr": round(ctr, 4),
                        "search_volume": int(impressions * 1.8),
                        "topic": topic, "subtopic": p.subtopic,
                        "planted": "BRIDGE_QUERY",
                    })
    return queries


# ── ground truth ────────────────────────────────────────────────────────────

def ground_truth(pages, links, queries, counts) -> dict:
    n_sub = sum(len(t["subtopics"]) for t in TOPICS.values())
    return {
        "pages": len(pages), "links": len(links), "queries": len(queries),
        "topics": len(TOPICS), "subtopics": n_sub,
        "cluster_spreads": {k: v["spread"] for k, v in TOPICS.items()},
        "planted_pages": {
            "true_noise": sum(1 for p in pages if "TRUE_NOISE" in p.planted),
            "utility_orphans": sum(
                1 for p in pages
                if any(k in p.planted for k in ("NAV_TARGET", "FOOTER_TARGET"))),
            "pillar_mismatch": sum(1 for p in pages if "PILLAR_MISMATCH" in p.planted),
            "cannibalisation": sum(1 for p in pages if "CANNIBALISATION" in p.planted),
            "orphan_new_strategic": sum(
                1 for p in pages if "ORPHAN_NEW_STRATEGIC" in p.planted),
            "no_head_term": sum(1 for p in pages if "NO_HEAD_TERM" in p.planted),
        },
        "planted_links": counts,
        "bridge_gaps": [list(x) for x in BRIDGE_GAPS],
        "well_connected": [list(x) for x in WELL_CONNECTED],
        "expected_verdicts": {
            "REANCHOR": counts["generic"],
            "REMOVE": counts["off_topic"],
            "FIX": counts["nofollow"],
            "ADD_LINK": sum(1 for p in pages if "ORPHAN_NEW_STRATEGIC" in p.planted),
        },
        "what_this_tests": {
            "hierarchy": f"{len(TOPICS)} topics / {n_sub} subtopics. HDBSCAN's "
                         "condensed tree should expose both levels.",
            "density": "spread 0.18 (pressbrake) to 0.55 (maintenance). Leiden's "
                       "resolution is global; HDBSCAN adapts per cluster.",
            "noise": "TRUE_NOISE pages have inbound body links, so Leiden must "
                     "assign them a community. HDBSCAN should label them -1.",
            "bridges": "BRIDGE_GAPS pairs share query vocabulary with ~1% link "
                       "density. Hub-to-hub scoring should rank them top.",
            "pillar": "PILLAR_MISMATCH topics have a declared pillar far from the "
                      "cluster centroid; centroid-nearest should differ.",
            "extraction": f"~{int(NO_HEAD_TERM_RATE*100)}% of pages omit their own "
                          "head term, pushing the ladder past rung 1.",
            "utility": "nav/footer pages exist but receive no body links, so they "
                       "are orphans. The crawler extracts body links only.",
        },
    }
