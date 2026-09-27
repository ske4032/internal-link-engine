"""Transform a scraped crawl into the link engine schema.

Dry run by default: reports what cleaning produces and writes nothing.

    uv run --env-file .env python scripts/prepare_corpus.py
    uv run --env-file .env python scripts/prepare_corpus.py --tenant <tenant> --write

The source is read through CrawlSource, which has no write methods. With
--write, pages and links are upserted into MONGO_DB under --tenant, each page
with its menu and footer inlinks: distinct other crawled pages linking to it
from template lines.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import statistics
from collections import Counter
from urllib.parse import urlsplit

from linking_engine.ingest.markdown_clean import (
    body_hash,
    clean_meta,
    clean_page,
    find_boilerplate,
    line_shares,
)
from linking_engine.ingest.mongo_repo import CrawlSource, MongoRepo
from linking_engine.ingest.template_links import count_template_inlinks
from linking_engine.ingest.url_params import query_param_evidence
from linking_engine.models import CleanedPage, CrawlPage, Heading, LinkRecord, PageRecord
from linking_engine.urls import UrlRules, normalise_url, url_rules

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


def _rank(doc: CrawlPage) -> tuple[bool, bool, int, str]:
    """Which of several crawled urls with one key to keep: 200, then https, then shortest."""
    url = str(doc.url)
    return (doc.status_code != 200, urlsplit(url).scheme != "https", len(url), url)


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source-db", required=True, help="crawl database (read-only)")
    ap.add_argument("--source-collection", required=True, help="crawl collection")
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

    rules = UrlRules()
    if args.tenant:
        async with await MongoRepo.connect(os.environ["MONGO_URI"], os.environ["MONGO_DB"]) as repo:
            rules = await repo.get_url_rules(args.tenant)
    print(
        f"url rules: keep {sorted(rules.keep_params) or 'built-in only'}, "
        f"drop {sorted(rules.drop_params) or 'none'}"
    )
    with url_rules(rules):
        await prepare(args)


async def prepare(args: argparse.Namespace) -> None:

    uri = os.environ["MONGO_URI"]
    async with await CrawlSource.connect(uri, args.source_db, args.source_collection) as source:
        docs: list[CrawlPage] = [doc async for batch in source.iter_pages() for doc in batch]
    evidence = query_param_evidence((str(doc.url), doc.content_hash) for doc in docs)
    if evidence:
        print("query parameters in crawled urls (pairs differing only in that parameter):")
        for item in evidence:
            if item.content_changed and not item.kept:
                verdict = f"changes content: consider scripts/url_rules.py --keep {item.name}"
            elif item.content_same and not item.content_changed and item.kept:
                verdict = f"never changed content: consider scripts/url_rules.py --drop {item.name}"
            else:
                verdict = "kept" if item.kept else "stripped"
            print(
                f"  {item.name:24} urls {item.urls:4}  content changed {item.content_changed:4}  "
                f"same {item.content_same:4}  {verdict}"
            )
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
    crawled = {normalise_url(str(page.url)) for _, page in pages}
    hits = sum(1 for link in internal if normalise_url(str(link.target_url)) in crawled)
    print(
        f"\nlinks extracted: {len(links)} ({len(internal)} internal, {len(links) - len(internal)} external)"
    )
    if internal:
        print(f"  internal targets that are prepared pages: {hits / len(internal):.0%}")

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

    chosen: dict[str, tuple[CrawlPage, CleanedPage]] = {}
    merged: list[str] = []
    for doc in docs:
        page = clean_page(doc.content or "", str(doc.url), title=doc.title, boilerplate=boilerplate)
        key = normalise_url(str(doc.url))
        current = chosen.get(key)
        if current is None or _rank(doc) < _rank(current[0]):
            if current is not None:
                merged.append(str(current[0].url))
            chosen[key] = (doc, page)
        else:
            merged.append(str(doc.url))
    if merged:
        print(
            f"\n{len(merged)} crawled urls share a normalised key with a kept page, e.g. {merged[:5]}"
        )

    inlinks = {
        item.url: item
        for item in count_template_inlinks((key, page) for key, (_, page) in chosen.items())
        if item.url in chosen
    }
    menu = sum(1 for item in inlinks.values() if item.menu_inlinks)
    footer = sum(1 for item in inlinks.values() if item.footer_inlinks)
    both = sum(1 for item in inlinks.values() if item.menu_inlinks and item.footer_inlinks)
    print(
        f"\npages linked from template lines of other pages: {menu} from menus, "
        f"{footer} from footers, {both} from both, of {len(chosen)}"
    )

    if not args.write:
        print("\ndry run: nothing written")
        return
    records: list[PageRecord] = []
    link_records: list[LinkRecord] = []
    for key, (doc, page) in chosen.items():
        found = inlinks.get(key)
        records.append(
            PageRecord(
                url=key,
                crawl_url=str(doc.url),
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
                body_hash=body_hash(page.body_text),
                scraped_at=doc.scraped_at,
                source=f"{args.source_db}.{args.source_collection}",
                menu_inlinks=found.menu_inlinks if found else 0,
                footer_inlinks=found.footer_inlinks if found else 0,
            )
        )
        link_records.extend(
            LinkRecord(
                source_url=key,
                position=position,
                target_url=str(link.target_url),
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
