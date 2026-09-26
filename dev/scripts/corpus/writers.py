"""
Persistence. Neo4j gets the graph and vectors; Mongo gets content, metrics and
the `_planted` ground-truth fields the assertion suite reads.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from corpus.structure import Link, Page
from corpus.taxonomy import TOPICS


def write_neo4j(uri: str, user: str, pwd: str,
                pages: list[Page], links: list[Link]) -> None:
    from neo4j import GraphDatabase

    drv = GraphDatabase.driver(uri, auth=(user, pwd))
    with drv.session() as s:
        s.run("MATCH (n) DETACH DELETE n")

        rows = []
        for p in pages:
            d = asdict(p)
            for k in ("html", "body_text", "planted"):
                d.pop(k, None)
            rows.append(d)

        s.run("""
        UNWIND $rows AS r
        CREATE (p:Page {
          url: r.url, topic: r.topic, subtopic: r.subtopic,
          pageType: r.page_type, isIndexable: r.is_indexable,
          httpStatus: r.http_status, wordCount: r.word_count,
          crawlDepth: r.crawl_depth, language: 'en',
          publishedAt: date(r.published_at), lifecycleStage: r.lifecycle_stage,
          contentHash: r.content_hash, content_embedding: r.embedding
        })""", rows=rows)

        s.run("""UNWIND $rows AS r
                 MERGE (k:Keyword {text: r.text, language: 'en'})
                 SET k.isStrategic = true""",
              rows=[{"text": t["head"]} for t in TOPICS.values()])

        for topic, t in TOPICS.items():
            s.run("""
            MATCH (p:Page {topic: $topic}), (k:Keyword {text: $head, language: 'en'})
            MERGE (p)-[r:TARGETS_KEYWORD]->(k)
            SET r.source = 'CLIENT_STRATEGIC', r.isPrimary = true,
                r.priority = CASE WHEN p.lifecycleStage = 'NEW' THEN 5 ELSE 4 END
            """, topic=topic, head=t["head"])

        lrows = [{k: v for k, v in asdict(l).items() if k != "planted"}
                 for l in links]
        s.run("""
        UNWIND $rows AS r
        MATCH (s:Page {url: r.source}), (t:Page {url: r.target})
        CREATE (s)-[:LINKS_TO {
          anchorText: r.anchor_text, anchorType: r.anchor_type,
          linkPosition: r.link_position, isFollow: r.is_follow,
          targetHttpStatus: r.target_http_status,
          surroundingText: r.surrounding_text
        }]->(t)""", rows=lrows)
    drv.close()


def write_mongo(uri: str, pages: list[Page], queries: list[dict]) -> None:
    from pymongo import MongoClient

    db = MongoClient(uri).get_database("linking_engine")
    for c in ("pages", "gsc_queries", "gsc_metrics",
              "strategic_keywords", "ctr_curves"):
        db[c].delete_many({"tenantId": "demo"})

    db.pages.insert_many([{
        "tenantId": "demo", "url": p.url, "title": p.title, "h1": p.h1,
        "html": p.html, "bodyText": p.body_text, "wordCount": p.word_count,
        "language": "en", "contentHash": p.content_hash,
        "crawledAt": datetime.utcnow(),
        "_topic": p.topic, "_subtopic": p.subtopic,
        "_includeHeadTerm": p.include_head_term, "_planted": p.planted,
    } for p in pages])

    db.gsc_queries.insert_many([{"tenantId": "demo", **q} for q in queries])

    # per-url rollup, what the ranker reads
    by_url: dict[str, list[dict]] = {}
    for q in queries:
        by_url.setdefault(q["url"], []).append(q)
    db.gsc_metrics.insert_many([{
        "tenantId": "demo", "url": url,
        "impressions_28d": sum(q["impressions"] for q in qs),
        "clicks_28d": sum(q["clicks"] for q in qs),
        "avg_position": round(sum(q["position"] for q in qs) / len(qs), 2),
        "query_count": len(qs),
    } for url, qs in by_url.items()])

    db.strategic_keywords.insert_many([{
        "tenantId": "demo", "url": p.url, "keyword": TOPICS[p.topic]["head"],
        "priority": 5 if p.lifecycle_stage == "NEW" else 4,
        "isPrimary": True, "language": "en",
    } for p in pages if p.topic])

    from corpus.taxonomy import CTR_CURVE
    db.ctr_curves.replace_one(
        {"tenantId": "demo"},
        {"tenantId": "demo", "curve": {str(k): v for k, v in CTR_CURVE.items()},
         "derivedAt": datetime.utcnow(), "source": "SYNTHETIC"},
        upsert=True)
