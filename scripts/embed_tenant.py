"""Embed one tenant's new and changed pages, anchors and sentences with Voyage into Neo4j.

Pages and links run as separate Prefect flows (embed-pages, embed-links), so a failure
names its stage; with --stage all a failed page flow does not stop the link flow.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Final

from linking_engine.errors import DatabaseError, EmbeddingError
from linking_engine.pipeline.embed import FLUSH_SIZE
from linking_engine.pipeline.flows import embed_links_flow, embed_pages_flow

USAGE: Final = "uv run --env-file .env python scripts/embed_tenant.py --tenant <tenant> [--stage pages|links|all]"
EXIT_FAILED: Final = 1
# Exit status when the graph and Mongo disagree on some pages.
EXIT_STALE_GRAPH: Final = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--stage", choices=("pages", "links", "all"), default="all")
    ap.add_argument("--flush-size", type=int, default=FLUSH_SIZE, help="items per Neo4j write")
    args = ap.parse_args()
    if args.flush_size < 1:
        ap.error("--flush-size must be at least 1")

    failures: list[str] = []
    stale = 0
    if args.stage in ("pages", "all"):
        try:
            pages = asyncio.run(embed_pages_flow(args.tenant, args.flush_size))
            print(f"pages:\n{pages.model_dump_json(indent=2)}")
            stale = pages.skipped_missing + pages.skipped_hash_mismatch
        except (EmbeddingError, DatabaseError) as error:
            failures.append(f"embed-pages failed: {type(error).__name__}: {error}")
    if args.stage in ("links", "all"):
        try:
            links = asyncio.run(embed_links_flow(args.tenant, args.flush_size))
            print(f"links:\n{links.model_dump_json(indent=2)}")
        except (EmbeddingError, DatabaseError) as error:
            failures.append(f"embed-links failed: {type(error).__name__}: {error}")

    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        sys.exit(EXIT_FAILED)
    if stale:
        print(
            f"graph and Mongo disagree for {stale} pages: "
            "run scripts/load_graph.py after scripts/prepare_corpus.py, then embed again",
            file=sys.stderr,
        )
        sys.exit(EXIT_STALE_GRAPH)


if __name__ == "__main__":
    main()
