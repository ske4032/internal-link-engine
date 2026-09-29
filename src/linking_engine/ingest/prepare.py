"""Prepare a tenant's crawl: clean every page and build its page and link records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

import structlog

from linking_engine.ingest.depth import crawl_depths
from linking_engine.ingest.markdown_clean import (
    body_hash,
    clean_meta,
    clean_page,
    find_boilerplate,
    line_shares,
)
from linking_engine.ingest.template_links import count_template_inlinks
from linking_engine.models import (
    EXCLUSION_LABELS,
    CleanedPage,
    CrawlPage,
    ExcludedPage,
    ExclusionReason,
    Heading,
    LanguageRules,
    LinkRecord,
    PageRecord,
    PrepareReport,
    TemplateInlinks,
)
from linking_engine.urls import is_sitemap, normalise_url, under_paths, url_rules

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from linking_engine.ingest.mongo_repo import CrawlSource, MongoRepo

log = structlog.get_logger(__name__)

# Both shares were measured on the first real crawl; see find_boilerplate.
BOILERPLATE_SHARE: Final = 0.2
NAV_SHARE: Final = 0.02
# A page whose body words are at least this share link text, over at least this many links, is a
# list of links rather than copy: kept out of the pipeline and labelled for review.
LINK_ONLY_SHARE: Final = 0.8
LINK_ONLY_MIN_LINKS: Final = 5


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    """Records for every crawled url key, and what the dry-run review reports on."""

    records: tuple[PageRecord, ...]
    links: tuple[LinkRecord, ...]
    # 200, usable pages with content, cleaned: the set template detection runs over.
    cleaned: tuple[tuple[CrawlPage, CleanedPage], ...]
    shares: tuple[tuple[str, float], ...]
    boilerplate: frozenset[str]
    skipped: dict[str, int]
    merged: tuple[str, ...]
    inlinks: dict[str, TemplateInlinks]
    # Crawled pages kept out of the pipeline: no record, and none of their links.
    excluded: tuple[ExcludedPage, ...] = ()


def keep_rank(doc: CrawlPage) -> tuple[bool, bool, int, str]:
    """Which of several crawled urls with one key to keep: 200, then https, then shortest."""
    url = str(doc.url)
    return (doc.status_code != 200, urlsplit(url).scheme != "https", len(url), url)


def page_language(url: str, rules: LanguageRules) -> str:
    """The language of the longest ``rules`` prefix matching the url key's path in whole
    segments (``/de/`` matches ``/de`` and ``/de/x``, not ``/design``), else the default."""
    path = _path_segments(url)
    matched, language = -1, rules.default_language
    for prefix, prefix_language in rules.prefixes:
        segments = _prefix_segments(prefix)
        if len(segments) > matched and path[: len(segments)] == segments:
            matched, language = len(segments), prefix_language
    return language


def _path_segments(url: str) -> list[str]:
    return url.split("?", 1)[0].partition("/")[2].split("/")


def _prefix_segments(prefix: str) -> list[str]:
    return [segment.lower() for segment in prefix.split("/") if segment]


def _is_root(url: str, rules: LanguageRules) -> bool:
    """A bare host, or a language home such as ``example.com/en``: a site whose root only
    redirects to its language homes still gets depths."""
    if "?" in url:
        return False
    if "/" not in url:
        return True
    path = _path_segments(url)
    return any(path == _prefix_segments(prefix) for prefix, _ in rules.prefixes)


def _followed_links(pages: Mapping[str, CleanedPage]) -> dict[str, set[str]]:
    """Every url key each page links to from its body or from its menu and footer lines."""
    edges: dict[str, set[str]] = {}
    for url, page in pages.items():
        targets = edges.setdefault(url, set())
        linked = [str(link.target_url) for link in page.links if link.is_internal]
        linked.extend(str(link.target_url) for link in page.template_links)
        for target in linked:
            try:
                targets.add(normalise_url(target))
            except ValueError:
                continue
    return edges


def exclusion(key: str, page: CleanedPage, excluded_paths: Collection[str]) -> ExcludedPage | None:
    """Why a crawled page stays out of the pipeline, or None: a sitemap by its path, a path the
    tenant excludes, or a body that is mostly link text."""
    words = len(page.body_text.split())
    link_words = sum(len(link.anchor_text.split()) for link in page.links)
    links = len(page.links)
    if is_sitemap(key):
        reason = ExclusionReason.SITEMAP
    elif under_paths(key, excluded_paths):
        reason = ExclusionReason.TENANT_EXCLUDED
    elif words and links >= LINK_ONLY_MIN_LINKS and link_words >= LINK_ONLY_SHARE * words:
        reason = ExclusionReason.INSUFFICIENT_CONTENT
    else:
        return None
    return ExcludedPage(
        url=key,
        reason=reason,
        label=EXCLUSION_LABELS[reason],
        words=words,
        link_words=link_words,
        links=links,
    )


def prepare_corpus(
    docs: Sequence[CrawlPage],
    *,
    source: str,
    boilerplate_share: float = BOILERPLATE_SHARE,
    nav_share: float = NAV_SHARE,
    language_rules: LanguageRules = LanguageRules(),
    excluded_paths: Collection[str] = frozenset(),
) -> PreparedCorpus:
    """Pure: runs under whatever url rules are active, so call it inside the tenant's.
    Excluded pages get no record and their links none; a link into one stays, like a link into
    any page that was not crawled."""
    skipped: Counter[str] = Counter()
    usable: list[CrawlPage] = []
    for doc in docs:
        if doc.status_code != 200:
            skipped[f"status {doc.status_code}"] += 1
        elif not doc.usable:
            skipped["not usable"] += 1
        elif not doc.content:
            skipped["no content"] += 1
        else:
            usable.append(doc)
    contents = [doc.content or "" for doc in usable]
    boilerplate = find_boilerplate(contents, min_share=boilerplate_share, nav_min_share=nav_share)

    chosen: dict[str, tuple[CrawlPage, CleanedPage]] = {}
    merged: list[str] = []
    cleaned_by_doc: dict[int, CleanedPage] = {}
    for doc in docs:
        page = clean_page(doc.content or "", str(doc.url), title=doc.title, boilerplate=boilerplate)
        cleaned_by_doc[id(doc)] = page
        key = normalise_url(str(doc.url))
        current = chosen.get(key)
        if current is None or keep_rank(doc) < keep_rank(current[0]):
            if current is not None:
                merged.append(str(current[0].url))
            chosen[key] = (doc, page)
        else:
            merged.append(str(doc.url))
    excluded = {
        key: kept_out
        for key, (_, page) in chosen.items()
        if (kept_out := exclusion(key, page, excluded_paths)) is not None
    }
    chosen = {key: pair for key, pair in chosen.items() if key not in excluded}

    inlinks = {
        item.url: item
        for item in count_template_inlinks((key, page) for key, (_, page) in chosen.items())
        if item.url in chosen
    }
    # Depth follows what a visitor can click, so template links count here though never as edges.
    depths = crawl_depths(
        _followed_links({key: page for key, (_, page) in chosen.items()}),
        (key for key in chosen if _is_root(key, language_rules)),
    )
    records: list[PageRecord] = []
    links: list[LinkRecord] = []
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
                source=source,
                menu_inlinks=found.menu_inlinks if found else 0,
                footer_inlinks=found.footer_inlinks if found else 0,
                language=page_language(key, language_rules),
                crawl_depth=depths.get(key),
            )
        )
        links.extend(
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
    return PreparedCorpus(
        records=tuple(records),
        links=tuple(links),
        cleaned=tuple((doc, cleaned_by_doc[id(doc)]) for doc in usable),
        shares=tuple(line_shares(contents)),
        boilerplate=boilerplate,
        skipped=dict(skipped),
        merged=tuple(merged),
        inlinks=inlinks,
        excluded=tuple(excluded[key] for key in sorted(excluded)),
    )


async def prepare_tenant(
    source: CrawlSource,
    mongo: MongoRepo,
    tenant_id: str,
    *,
    source_name: str,
    write: bool = True,
    boilerplate_share: float = BOILERPLATE_SHARE,
    nav_share: float = NAV_SHARE,
) -> tuple[PreparedCorpus, PrepareReport]:
    """Read the crawl, prepare it under the tenant's url and language rules and, with
    ``write``, upsert it."""
    if not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string")
    rules = await mongo.get_url_rules(tenant_id)
    language_rules = await mongo.get_language_rules(tenant_id)
    excluded_paths = await mongo.get_excluded_paths(tenant_id)
    docs = [doc async for batch in source.iter_pages() for doc in batch]
    written = (0, 0, 0)
    with url_rules(rules):
        corpus = prepare_corpus(
            docs,
            source=source_name,
            boilerplate_share=boilerplate_share,
            nav_share=nav_share,
            language_rules=language_rules,
            excluded_paths=excluded_paths,
        )
        if write:
            await mongo.ensure_indexes()
            written = await mongo.write_pages(tenant_id, list(corpus.records), list(corpus.links))
            await mongo.delete_pages(tenant_id, [page.url for page in corpus.excluded])
            await mongo.replace_excluded_pages(tenant_id, corpus.excluded)
    counts = corpus.inlinks.values()
    report = PrepareReport(
        tenant_id=tenant_id,
        source=source_name,
        documents=len(docs),
        cleaned=len(corpus.cleaned),
        skipped=corpus.skipped,
        template_lines=len(corpus.boilerplate),
        pages=len(corpus.records),
        links=len(corpus.links),
        merged_urls=len(corpus.merged),
        menu_inlink_pages=sum(1 for c in counts if c.menu_inlinks),
        footer_inlink_pages=sum(1 for c in counts if c.footer_inlinks),
        both_inlink_pages=sum(1 for c in counts if c.menu_inlinks and c.footer_inlinks),
        pages_written=written[0],
        links_written=written[1],
        stale_links_deleted=written[2],
        pages_with_depth=sum(1 for record in corpus.records if record.crawl_depth is not None),
        excluded=dict(Counter(page.reason for page in corpus.excluded)),
        finished_at=datetime.now(UTC),
    )
    log.info("ingest.prepare", **report.model_dump(exclude={"skipped", "excluded", "finished_at"}))
    if corpus.excluded:
        log.info(
            "ingest.pages_excluded",
            tenant_id=tenant_id,
            by_reason={reason.value: n for reason, n in report.excluded.items()},
        )
    return corpus, report
