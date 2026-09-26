"""Prepare a scraped corpus for the link engine: clean the markdown, extract links.

Dry run by default: reports what cleaning produces and writes nothing.

    uv run --env-file .env python scripts/prepare_corpus.py
    uv run --env-file .env python scripts/prepare_corpus.py --tenant <id> --write

With ``--write`` each prepared page is upserted into the project database's
``pages`` collection, keyed on ``(tenantId, url)`` so re-runs converge
(ADR-013). Only pages with HTTP 200 and ``usable`` set are prepared.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
from collections import Counter
from datetime import UTC, datetime

from pymongo import MongoClient, UpdateOne

from linking_engine.ingest.markdown_clean import clean_page, find_boilerplate, line_shares

# Anything that should never survive cleaning.
RESIDUE = {
    "image or link markup": re.compile(r"!\[|\]\("),
    "url": re.compile(r"https?://\S|www\.\S"),
    "block marker": re.compile(r"^\s*(?:#{1,6}\s|>|\|)", re.M),
    "rule": re.compile(r"^\s*[=\-]{3,}\s*$", re.M),
    "emphasis marker": re.compile(r"\*\*|__"),
    "html tag": re.compile(r"</?[a-zA-Z][^<>]*>"),
    "html entity": re.compile(r"&[a-zA-Z]+;|&#\d+;"),
}


def pct(values: list[int], q: float) -> int:
    return sorted(values)[min(len(values) - 1, int(len(values) * q))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source-db", default="action1")
    ap.add_argument("--source-collection", default="pages_v2")
    ap.add_argument("--boilerplate-share", type=float, default=0.2)
    ap.add_argument("--nav-share", type=float, default=0.02)
    ap.add_argument("--tenant", help="tenant id the pages are written under")
    ap.add_argument("--write", action="store_true", help="upsert into the project pages collection")
    ap.add_argument("--sample", help="url to show before and after (default: median-length page)")
    args = ap.parse_args()
    if args.write and not args.tenant:
        ap.error("--write needs --tenant")

    client = MongoClient(os.environ["MONGO_URI"])
    source = client[args.source_db][args.source_collection]
    docs = list(
        source.find(
            {},
            {
                "url": 1,
                "title": 1,
                "description": 1,
                "content": 1,
                "statusCode": 1,
                "usable": 1,
                "contentHash": 1,
                "scrapedAt": 1,
            },
        )
    )
    skipped: Counter[str] = Counter()
    keep = []
    for doc in docs:
        if doc.get("statusCode") != 200:
            skipped[f"status {doc.get('statusCode')}"] += 1
        elif not doc.get("usable"):
            skipped["not usable"] += 1
        elif not doc.get("content"):
            skipped["no content"] += 1
        else:
            keep.append(doc)

    shares = line_shares(doc["content"] for doc in keep)
    boilerplate = find_boilerplate(
        (doc["content"] for doc in keep),
        min_share=args.boilerplate_share,
        nav_min_share=args.nav_share,
    )
    pages = [
        (
            doc,
            clean_page(doc["content"], doc["url"], title=doc.get("title"), boilerplate=boilerplate),
        )
        for doc in keep
    ]

    print(
        f"source {args.source_db}.{args.source_collection}: {len(docs)} documents, "
        f"{len(keep)} prepared, skipped {dict(skipped) or 'none'}\n"
    )

    template = [(line, share) for line, share in shares if line in boilerplate]
    navigation = [(line, share) for line, share in template if share < args.boilerplate_share]
    print(
        f"template lines removed: {len(template)} "
        f"({len(template) - len(navigation)} on >= {args.boilerplate_share:.0%} of pages, "
        f"{len(navigation)} navigation lines on >= {args.nav_share:.0%})"
    )
    for line, share in template[:6]:
        print(f"  {share:5.0%}  {line[:100]}")
    # Repeated lines that survive: the ones a person should look at. Rules and
    # images are removed by the cleaner anyway, so they are not listed.
    removed_anyway = re.compile(r"^\s*[=\-]{3,}\s*$|^!\[|^\[!\[")
    kept = [
        (line, share)
        for line, share in shares
        if share >= 0.05 and line not in boilerplate and not removed_anyway.search(line)
    ]
    print(f"repeated on >= 5% of pages but kept, review these: {len(kept)}")
    for line, share in kept[:12]:
        print(f"  {share:5.0%}  {line[:100]}")

    totals: Counter[str] = Counter()
    for _, page in pages:
        totals.update(dict(page.removed))
    print("\nremoved, by kind:")
    for kind, n in totals.most_common():
        print(f"  {kind:20} {n:7}")

    links = [link for _, page in pages for link in page.links]
    internal = [link for link in links if link.is_internal]
    crawled = {str(page.url) for _, page in pages}
    crawled_loose = {u.rstrip("/").replace("http://", "https://") for u in crawled}
    hits = sum(1 for link in internal if str(link.target_url) in crawled)
    hits_loose = sum(
        1
        for link in internal
        if str(link.target_url).rstrip("/").replace("http://", "https://") in crawled_loose
    )
    print(
        f"\nlinks extracted: {len(links)} ({len(internal)} internal, {len(links) - len(internal)} external)"
    )
    if internal:
        print(
            f"  internal targets that are prepared pages: {hits / len(internal):.0%} exact, "
            f"{hits_loose / len(internal):.0%} ignoring trailing slash and scheme"
        )

    before = [len(doc["content"]) for doc, _ in pages]
    after = [len(page.body_text) for _, page in pages]
    print(
        f"\ncharacters per page  before: median {statistics.median(before):,.0f}, "
        f"p90 {pct(before, 0.9):,}, max {max(before):,}"
    )
    print(
        f"                     after:  median {statistics.median(after):,.0f}, "
        f"p90 {pct(after, 0.9):,}, max {max(after):,}"
    )
    print(
        f"  total after: {sum(after):,} characters, roughly {sum(after) // 4:,} tokens "
        f"(4 characters per token, an estimate)"
    )
    print(f"  pages with empty body after cleaning: {sum(1 for n in after if n == 0)}")
    print(f"  pages with an h1: {sum(1 for _, page in pages if page.h1)}")

    residue: Counter[str] = Counter()
    example: dict[str, str] = {}
    for _, page in pages:
        for kind, pattern in RESIDUE.items():
            match = pattern.search(page.body_text)
            if match:
                residue[kind] += 1
                example.setdefault(
                    kind, page.body_text[max(0, match.start() - 40) : match.end() + 40]
                )
    print(f"\nresidue check: {sum(residue.values())} hits across {len(pages)} pages")
    for kind, n in residue.most_common():
        print(f"  {kind:20} {n:4} pages  e.g. ...{example[kind]!r}...")

    sample = next((p for d, p in pages if d["url"] == args.sample), None) if args.sample else None
    if sample is None:
        sample = sorted(pages, key=lambda dp: len(dp[1].body_text))[len(pages) // 2][1]
    raw = next(doc["content"] for doc, page in pages if page is sample)
    print(f"\nsample {sample.url}  (h1: {sample.h1!r})")
    print("---- before (first 500 chars) ----")
    print(raw[:500])
    print("---- after (first 500 chars) ----")
    print(sample.body_text[:500])

    if not args.write:
        print("\ndry run: nothing written")
        return
    target = client[os.environ["MONGO_DB"]]["pages"]
    now = datetime.now(UTC)
    ops = [
        UpdateOne(
            {"tenantId": args.tenant, "url": str(page.url)},
            {
                "$set": {
                    "tenantId": args.tenant,
                    "url": str(page.url),
                    "title": page.title,
                    "h1": page.h1,
                    "meta": doc.get("description"),
                    "bodyText": page.body_text,
                    "links": [link.model_dump(mode="json") for link in page.links],
                    "contentHash": doc.get("contentHash"),
                    "crawledAt": doc.get("scrapedAt"),
                    "source": f"{args.source_db}.{args.source_collection}",
                    "preparedAt": now,
                }
            },
            upsert=True,
        )
        for doc, page in pages
    ]
    result = target.bulk_write(ops, ordered=False)
    print(
        f"\nwritten to {os.environ['MONGO_DB']}.pages under tenant {args.tenant!r}: "
        f"{result.upserted_count} inserted, {result.modified_count} updated"
    )


if __name__ == "__main__":
    main()
