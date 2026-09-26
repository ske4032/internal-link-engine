"""Transform a scraped crawl (the pages_v2 shape) into the link engine schema.

Dry run by default: reports what cleaning produces and writes nothing.

    uv run --env-file .env python scripts/prepare_corpus.py
    uv run --env-file .env python scripts/prepare_corpus.py --tenant action1 --write

The source is read through CrawlSource, which has no write methods. With
--write, pages and links are upserted into MONGO_DB under --tenant.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import statistics
from collections import Counter

from linking_engine.ingest.markdown_clean import (
    clean_meta,
    clean_page,
    find_boilerplate,
    line_shares,
)
from linking_engine.ingest.mongo_repo import CrawlSource, MongoRepo
from linking_engine.models import CrawlPage, Heading, LinkRecord, PageRecord

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


async def main() -> None:
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
    if args.source_db == os.environ["MONGO_DB"]:
        ap.error("source and target database must differ: the source is read-only")

    uri = os.environ["MONGO_URI"]
    async with await CrawlSource.connect(uri, args.source_db, args.source_collection) as source:
        docs: list[CrawlPage] = [doc async for batch in source.iter_pages() for doc in batch]
    skipped: Counter[str] = Counter()
    keep: list[CrawlPage] = []
    for doc in docs:
        if doc.status_code != 200:
            skipped[f"status {doc.status_code}"] += 1
        elif not doc.usable:
            skipped["not usable"] += 1
        elif not doc.content:
            skipped["no content"] += 1
        else:
            keep.append(doc)

    shares = line_shares(doc.content or "" for doc in keep)
    boilerplate = find_boilerplate(
        (doc.content or "" for doc in keep),
        min_share=args.boilerplate_share,
        nav_min_share=args.nav_share,
    )
    pages = [
        (
            doc,
            clean_page(doc.content or "", str(doc.url), title=doc.title, boilerplate=boilerplate),
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

    before = [len(doc.content or "") for doc, _ in pages]
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

    sample = next((p for d, p in pages if str(d.url) == args.sample), None) if args.sample else None
    if sample is None:
        sample = sorted(pages, key=lambda dp: len(dp[1].body_text))[len(pages) // 2][1]
    raw = next(doc.content or "" for doc, page in pages if page is sample)
    print(f"\nsample {sample.url}  (h1: {sample.h1!r})")
    print("---- before (first 500 chars) ----")
    print(raw[:500])
    print("---- after (first 500 chars) ----")
    print(sample.body_text[:500])

    if not args.write:
        print("\ndry run: nothing written")
        return
    records: list[PageRecord] = []
    link_records: list[LinkRecord] = []
    for doc in docs:
        page = clean_page(doc.content or "", str(doc.url), title=doc.title, boilerplate=boilerplate)
        records.append(
            PageRecord(
                url=page.url,
                status_code=doc.status_code,
                usable=doc.usable,
                meta_title=page.title,
                meta_description=clean_meta(doc.description),
                h1=page.h1,
                headings=tuple(Heading(level=lvl, text=txt) for lvl, txt in page.headings),
                body_text=page.body_text,
                word_count=len(page.body_text.split()),
                link_count=len(page.links),
                content_hash=doc.content_hash,
                scraped_at=doc.scraped_at,
                source=f"{args.source_db}.{args.source_collection}",
            )
        )
        link_records.extend(
            LinkRecord(
                source_url=page.url,
                position=position,
                target_url=link.target_url,
                anchor_text=link.anchor_text,
                surrounding_text=link.surrounding_text,
                is_internal=link.is_internal,
            )
            for position, link in enumerate(page.links)
        )
    async with await MongoRepo.connect(uri, os.environ["MONGO_DB"]) as repo:
        await repo.ensure_indexes()
        written_pages, written_links, deleted = await repo.write_pages(
            args.tenant, records, link_records
        )
    print(
        f"\nwritten to {os.environ['MONGO_DB']} under tenant {args.tenant!r}: "
        f"{written_pages} pages, {written_links} links, {deleted} stale links deleted"
    )


if __name__ == "__main__":
    asyncio.run(main())
