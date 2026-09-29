"""Show or change the paths a tenant keeps out of the pipeline, and list the pages the latest
preparation excluded with their reasons.

Sitemap pages and pages that are mostly link text are excluded without a setting; this adds a
tenant's own paths, each covering the pages below it.

    uv run --env-file .env python scripts/excluded_paths.py --tenant <tenant>
    uv run --env-file .env python scripts/excluded_paths.py --tenant <tenant> --add /site-index --remove /old

Re-run prepare_corpus.py and load_graph.py after a change.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.urls import normalise_path


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--add", action="append", default=[], help="path to exclude")
    ap.add_argument("--remove", action="append", default=[], help="path to stop excluding")
    args = ap.parse_args()

    async with await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]) as repo:
        paths = await repo.get_excluded_paths(args.tenant)
        if args.add or args.remove:
            removed = {normalise_path(path) for path in args.remove}
            try:
                added = {normalise_path(path) for path in args.add}
                paths = await repo.set_excluded_paths(args.tenant, (paths | added) - removed)
            except ValueError as error:
                ap.error(str(error))
        excluded = await repo.excluded_pages(args.tenant)
    print(f"tenant {args.tenant}: excluded paths {sorted(paths)}")
    print(f"excluded by the latest preparation: {len(excluded)}")
    for page in excluded:
        print(
            f"  {page.url}  {page.label}  "
            f"({page.link_words} of {page.words} words in {page.links} links)"
        )


if __name__ == "__main__":
    asyncio.run(main())
