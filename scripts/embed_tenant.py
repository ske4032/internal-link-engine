"""Embed one tenant's new and changed pages with Voyage and store the vectors in Neo4j."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Final

from linking_engine.errors import DatabaseError, EmbeddingError
from linking_engine.pipeline.embed import FLUSH_SIZE
from linking_engine.pipeline.flows import embed_tenant_flow

USAGE: Final = "uv run --env-file .env python scripts/embed_tenant.py --tenant action1"
# Exit status when the graph and Mongo disagree on some pages.
EXIT_STALE_GRAPH: Final = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, epilog=USAGE)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--flush-size", type=int, default=FLUSH_SIZE, help="pages per Neo4j write")
    args = ap.parse_args()
    if args.flush_size < 1:
        ap.error("--flush-size must be at least 1")

    try:
        report = asyncio.run(embed_tenant_flow(args.tenant, args.flush_size))
    except (EmbeddingError, DatabaseError) as error:
        sys.exit(f"embedding run failed: {type(error).__name__}: {error}")
    print(report.model_dump_json(indent=2))
    stale = report.skipped_missing + report.skipped_hash_mismatch
    if stale:
        print(
            f"graph and Mongo disagree for {stale} pages: "
            "run scripts/load_graph.py after scripts/prepare_corpus.py, then embed again",
            file=sys.stderr,
        )
        sys.exit(EXIT_STALE_GRAPH)


if __name__ == "__main__":
    main()
