"""Show or change a tenant's overrides of the generic-anchor dictionary.

    uv run --env-file .env python scripts/anchor_rules.py --tenant action1
    uv run --env-file .env python scripts/anchor_rules.py --tenant action1 --add "download now" --remove details

Re-run embed_tenant.py after a change.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from linking_engine.anchor.generic import GENERIC_ANCHORS_BY_LANGUAGE
from linking_engine.ingest.mongo_repo import MongoRepo
from linking_engine.models import AnchorRules


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--add", action="append", default=[], help="phrase to treat as generic")
    ap.add_argument("--remove", action="append", default=[], help="phrase never treated as generic")
    ap.add_argument("--unset", action="append", default=[], help="drop a phrase from both lists")
    args = ap.parse_args()

    async with await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]) as repo:
        rules = await repo.get_anchor_rules(args.tenant)
        if args.add or args.remove or args.unset:
            unset = {phrase.strip().lower() for phrase in args.unset}
            add = (rules.generic_add | {p.strip().lower() for p in args.add}) - unset
            remove = (rules.generic_remove | {p.strip().lower() for p in args.remove}) - unset
            if add & remove:
                ap.error(f"a phrase cannot be both added and removed: {sorted(add & remove)}")
            rules = AnchorRules(generic_add=add, generic_remove=remove)
            await repo.set_anchor_rules(args.tenant, rules)
    for language, phrases in sorted(GENERIC_ANCHORS_BY_LANGUAGE.items()):
        print(f"built-in {language}: {len(phrases)} phrases")
    print(
        f"tenant {args.tenant}: add {sorted(rules.generic_add)}, remove {sorted(rules.generic_remove)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
