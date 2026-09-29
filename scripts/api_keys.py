"""Issue, revoke or list a tenant's output API keys.

A key is printed once, when it is issued, and cannot be shown again: only its hash is stored.

    uv run --env-file .env python scripts/api_keys.py --tenant <tenant> --issue --label "Acme dashboard"
    uv run --env-file .env python scripts/api_keys.py --tenant <tenant> --revoke <key id>
    uv run --env-file .env python scripts/api_keys.py --tenant <tenant> --list
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from pymongo import AsyncMongoClient

from linking_engine.output.keys import KeyStore


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tenant", required=True)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--issue", action="store_true", help="issue a new key")
    action.add_argument("--revoke", metavar="KEY_ID", help="revoke the key with this id")
    action.add_argument("--list", action="store_true", help="list the tenant's keys")
    ap.add_argument("--label", help="what the key is for; with --issue")
    args = ap.parse_args()
    if args.label is not None and not args.issue:
        ap.error("--label goes with --issue")

    client: AsyncMongoClient[dict[str, object]] = AsyncMongoClient(
        os.environ["MONGO_URI"], tz_aware=True
    )
    try:
        keys = KeyStore(client[os.environ["MONGO_DB"]])
        if args.issue:
            await keys.ensure_indexes()
            key, info = await keys.issue(args.tenant, args.label)
            print(f"tenant {args.tenant}: issued key {info.key_id}")
            print(key)
            print(
                "Store this key now: it is not stored and cannot be shown again.", file=sys.stderr
            )
        elif args.revoke:
            if not await keys.revoke(args.tenant, args.revoke):
                sys.exit(f"tenant {args.tenant}: no live key {args.revoke}")
            print(f"tenant {args.tenant}: revoked key {args.revoke}")
        else:
            listed = await keys.keys(args.tenant)
            print(f"tenant {args.tenant}: {len(listed)} keys")
            for info in listed:
                revoked = (
                    "live"
                    if info.revoked_at is None
                    else f"revoked {info.revoked_at:%Y-%m-%d %H:%M}"
                )
                print(
                    f"  {info.key_id}  {info.label or '-'}  "
                    f"created {info.created_at:%Y-%m-%d %H:%M}  {revoked}"
                )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
