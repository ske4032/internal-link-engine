"""
Synthetic corpus generator.

    # instant, offline — plumbing work
    uv run python scripts/generate_corpus.py --pages 600

    # real prose + real embeddings — before trusting any retrieval number
    uv run python scripts/generate_corpus.py --pages 600 \
        --content llm --embeddings voyage

    # ground truth only, touches nothing
    uv run python scripts/generate_corpus.py --report

Backends are independent. `--content llm --embeddings synthetic` is a valid
combination for checking HTML extraction without spending on Voyage.

What each backend proves:

  content=template     nothing about text quality. Structurally regular prose.
  content=llm          real heading hierarchy, real anchors in real sentences.
  embeddings=synthetic clustering and retrieval plumbing. Not quality.
  embeddings=voyage    the only configuration whose retrieval numbers mean
                       anything.

Ground truth survives every combination — that is why LLM output is verified
against the planted constraints rather than trusted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from corpus import content as content_mod
from corpus import embeddings as emb_mod
from corpus import writers
from corpus.structure import build_links, build_pages, build_queries, ground_truth


def build(n_pages: int, seed: int):
    rng = random.Random(seed)
    pages, by_topic = build_pages(n_pages, rng)
    links, counts = build_links(pages, by_topic, rng)
    queries = build_queries(pages, by_topic, rng)
    return pages, links, queries, ground_truth(pages, links, queries, counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--content", choices=["template", "llm"], default="template")
    ap.add_argument("--embeddings", choices=["synthetic", "voyage"], default="synthetic")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="print ground truth and exit, no generation")
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="localdevpassword")
    ap.add_argument("--mongo-uri",
                    default="mongodb://localhost:27017/?directConnection=true")
    a = ap.parse_args()

    pages, links, queries, truth = build(a.pages, a.seed)

    if a.report:
        print(json.dumps(truth, indent=2))
        return

    print(f"{len(pages)} pages · {len(links)} links · {len(queries)} queries")
    print(f"content={a.content}  embeddings={a.embeddings}")

    links_by_source: dict[str, list] = {}
    for l in links:
        links_by_source.setdefault(l.source, []).append(l)

    print("generating content...")
    stats = asyncio.run(content_mod.generate_all(
        pages, links_by_source, a.content, a.seed,
        concurrency=a.concurrency, use_cache=not a.no_cache))
    print(f"  {stats}")
    if stats.get("fallback"):
        print(f"  ⚠ {stats['fallback']} pages fell back to template "
              f"({stats['violations']} constraint violations) — "
              f"tagged CONTENT_FALLBACK in _planted")

    print("embedding...")
    if a.embeddings == "voyage":
        print(f"  {emb_mod.apply_voyage(pages, queries)}")
    else:
        emb_mod.apply_synthetic(pages, a.seed)
        rng = np.random.default_rng(a.seed + 2)
        space = emb_mod.build_space(a.seed)
        for q in queries:
            q["embedding"] = emb_mod.embed_query_synthetic(
                rng, space, q["topic"], q.get("subtopic")).tolist()
        print(f"  synthetic vectors for {len(pages)} pages, {len(queries)} queries")

    writers.write_neo4j(a.neo4j_uri, a.neo4j_user, a.neo4j_password, pages, links)
    print("  neo4j written")
    writers.write_mongo(a.mongo_uri, pages, queries)
    print("  mongo written\n")

    truth["generation"] = {"content": a.content, "embeddings": a.embeddings, **stats}
    print(json.dumps(truth, indent=2))


if __name__ == "__main__":
    main()
