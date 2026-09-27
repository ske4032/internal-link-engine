"""Show or change a tenant's query parameter overrides for URL normalisation.

    uv run --env-file .env python scripts/url_rules.py --tenant <tenant>
    uv run --env-file .env python scripts/url_rules.py --tenant <tenant> --keep announcement_pg --drop p

Re-run prepare_corpus.py, load_graph.py and embed_tenant.py after a change.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.urls import DOCUMENT_ID_PARAMS, OFFSET_PARAMS, PAGE_NUMBER_PARAMS, UrlRules


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--keep", action="append", default=[], help="query parameter to keep")
    ap.add_argument("--drop", action="append", default=[], help="query parameter to strip")
    ap.add_argument(
        "--unset", action="append", default=[], help="remove a parameter from both lists"
    )
    args = ap.parse_args()

    async with await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]) as repo:
        rules = await repo.get_url_rules(args.tenant)
        if args.keep or args.drop or args.unset:
            unset = {name.lower() for name in args.unset}
            keep = (rules.keep_params | {n.lower() for n in args.keep}) - unset
            drop = (rules.drop_params | {n.lower() for n in args.drop}) - unset
            if keep & drop:
                ap.error(f"a parameter cannot be both kept and dropped: {sorted(keep & drop)}")
            rules = UrlRules(keep_params=keep, drop_params=drop)
            await repo.set_url_rules(args.tenant, rules)
    print(f"built-in page numbers: {sorted(PAGE_NUMBER_PARAMS)}")
    print(f"built-in offsets:      {sorted(OFFSET_PARAMS)}")
    print(f"built-in document ids: {sorted(DOCUMENT_ID_PARAMS)}")
    print(
        f"tenant {args.tenant}: keep {sorted(rules.keep_params)}, drop {sorted(rules.drop_params)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
