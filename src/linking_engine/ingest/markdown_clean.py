"""Turn scraped markdown into clean body text for storage and embedding.

Removed: heading underlines and horizontal rules, heading, list and blockquote
markers, images (including linked images and inline SVG data), bare and
angle-bracket URLs, emphasis and code markers, table pipes and rules, HTML
fragments, backslash escapes, HTML entities, breadcrumbs, and site template:
lines repeated across many pages (banners, calls to action) and navigation
(link-only, list or heading lines repeated across a section, such as a
documentation sidebar).

Links are extracted before their markup goes. Each link's anchor words stay in
the body text, and its target, anchor and surrounding sentence are returned
alongside: that is the only record of them the audit and anchor stages get.
Boilerplate lines are dropped before extraction, so breadcrumb and banner links
never become body links (ADR-004).

Everything here is a pure function over strings. Reading a source collection
and writing the result belongs to the caller.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit, urlunsplit

from pydantic import HttpUrl, ValidationError

from linking_engine.models.corpus import CleanedPage, ExtractedLink

if TYPE_CHECKING:
    from collections.abc import Iterable

# A link or image destination, "( url "optional title" )", allowing one level of
# balanced parentheses inside the url.
_DEST = r"\((?P<{name}>(?:[^()\s]|\([^()\s]*\))*)(?:\s+\"[^\"]*\")?\s*\)"

_LINKED_IMAGE = re.compile(
    r"\[\s*!\[[^\]]*\]" + _DEST.format(name="img") + r"\s*\]" + _DEST.format(name="dest")
)
_IMAGE = re.compile(r"!\[[^\]]*\]" + _DEST.format(name="img"))
_LINK = re.compile(r"\[(?P<text>[^\[\]]*)\]" + _DEST.format(name="dest"))
_AUTOLINK = re.compile(r"<(?:https?|mailto):[^>\s]+>")
_BARE_URL = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]+")

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_STRONG = re.compile(r"(?<!\\)(\*\*|__)(?=\S)(.+?)(?<=[^\s\\])\1")
_EM_STAR = re.compile(r"(?<![\w*\\])\*(?=\S)([^*\n]+?)(?<=[^\s\\])\*(?![\w*])")
_EM_UNDERSCORE = re.compile(r"(?<![\w\\])_(?=\S)([^_\n]+?)(?<=[^\s\\])_(?!\w)")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
_SPACES = re.compile(r"[ \t\u00a0]+")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")

_RULE = re.compile(r"^\s*(?:[=\-]{3,}|(?:\*\s*){3,}|(?:_\s*){3,})\s*$")
_ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_BLOCKQUOTE = re.compile(r"^\s*>\s?")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_BREADCRUMB_SEPARATOR = re.compile(
    r"[\u203a\u00bb]"
)  # single and double right-pointing angle quotes
_ANGLE_SEPARATOR = re.compile(r"^[\u203a\u00bb]\s")
# A breadcrumb continuation: an angle quote, or the "5" an icon font renders as a
# separator glyph. Only trusted directly after a breadcrumb or template line, so
# content such as "5 steps to patch" survives.
_BREADCRUMB_CONTINUATION = re.compile(r"^(?:[\u203a\u00bb]|5)\s+\S")
# Link text wrapped over up to three extra lines, joined before per-line work.
_WRAPPED_LINK_TEXT = re.compile(r"\[[^\[\]\n]*(?:\n[^\[\]\n]*){1,3}\]\(")
_ORPHAN_LINK_TAIL = re.compile(r"\]\(\s*[^)\s]*\s*\)")
_ORPHAN_STRONG = re.compile(r"\*\*")
# Icon-font glyphs a scraper renders in place of bullets, at line start.
_GLYPH_BULLET = re.compile(
    r"^\s*[\^\u2022\u00b7\u25aa\u25e6\u2610\u2611\u2713\u2714\u2013\u2014\u2192]\s+"
)
_REPLACEMENT_CHAR = re.compile(r"\ufffd")
# A fragment the source wrapped right after a link: ", identifying ...". The
# space keeps ".NET" and ".action1.com" from counting.
_CONTINUATION = re.compile(r"^[,.;:](?:\s|$)")
# What an image-only table row or a stripped separator leaves behind: ";".
_PUNCTUATION_ONLY = re.compile(r"[^\w]+")
# A link or image target, blanked so template lines that differ only in their
# urls (per-post "Previous / Next" links) compare equal.
_LINK_TARGET = re.compile(r"\]" + _DEST.format(name="target"))
# Navigation candidates: a line that is only a link (optionally behind list or
# breadcrumb markers), or a list item or heading.
_LINK_ONLY = re.compile(
    r"^(?:(?:[-*+]|\d{1,3}[.)])\s+)*(?:(?:[\u203a\u00bb]|5)\s+)?\[[^\[\]]+\]\([^)]*\)\s*$"
)
_STRUCTURAL = re.compile(r"^(?:[-*+]\s|\d{1,3}[.)]\s|#{1,6}\s)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")

_NON_WEB_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "sms:", "ftp:")


def line_shares(documents: Iterable[str], min_length: int = 20) -> list[tuple[str, float]]:
    """Every line of at least ``min_length`` characters, as its template key,
    with the share of documents it appears in, most widespread first."""
    counts: Counter[str] = Counter()
    total = 0
    for document in documents:
        total += 1
        counts.update(
            {
                template_key(line)
                for line in document.splitlines()
                if len(line.strip()) >= min_length
            }
        )
    if total == 0:
        return []
    return sorted(((line, n / total) for line, n in counts.items()), key=lambda x: -x[1])


def find_boilerplate(
    documents: Iterable[str],
    min_share: float = 0.2,
    nav_min_share: float = 0.02,
    min_length: int = 20,
) -> frozenset[str]:
    """Site template, found by repetition across documents, as template keys
    (see ``template_key``: lines differing only in link targets are one line).

    Two tiers, because one threshold cannot separate template from content:

    * any line repeated on at least ``min_share`` of documents
      (banners, calls to action);
    * a navigation line (link-only, list item or heading) repeated on at least
      ``nav_min_share`` (a section sidebar, tag lists, bylines).

    Real content can repeat too, for example one advisory summarised on a hundred
    pages, which is why plain sentences need the higher share. Both defaults come
    from measuring the first real crawl; callers should report the lines just
    below them.
    """
    for name, value in (("min_share", min_share), ("nav_min_share", nav_min_share)):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1], got {value}")
    return frozenset(
        line
        for line, share in line_shares(documents, min_length)
        if share >= min_share or (share >= nav_min_share and is_navigation(line))
    )


def template_key(line: str) -> str:
    """A line with link and image targets blanked: the identity used to compare
    template lines across pages."""
    return _LINK_TARGET.sub("]()", line.strip())


def clean_meta(text: str | None) -> str | None:
    """Clean a plain-text meta field such as a title or description."""
    if text is None:
        return None
    text = _REPLACEMENT_CHAR.sub("", _ZERO_WIDTH.sub("", text))
    return _SPACES.sub(" ", html.unescape(text)).strip() or None


def is_navigation(line: str) -> bool:
    """True for a link-only line, a list item or a heading."""
    return bool(_LINK_ONLY.match(line) or _STRUCTURAL.match(line))


def clean_page(
    markdown: str,
    url: str,
    *,
    title: str | None = None,
    boilerplate: frozenset[str] = frozenset(),
) -> CleanedPage:
    """Clean one page's markdown and extract its links."""
    page_url = _normalise(url)
    if page_url is None:
        raise ValueError(f"page url is not an absolute http(s) url: {url!r}")
    host = _host(page_url)
    removed: Counter[str] = Counter()
    links: list[ExtractedLink] = []
    out: list[str] = []
    headings: list[tuple[int, str]] = []
    h1: str | None = None
    # The cleaned text of the line directly above, for setext underlines. Reset
    # by anything that is not a kept text line.
    last_line: str | None = None

    text = _ZERO_WIDTH.sub("", markdown.replace("\r\n", "\n").replace("\r", "\n"))
    text = _WRAPPED_LINK_TEXT.sub(lambda m: m.group(0).replace("\n", " "), text)
    template = frozenset(template_key(line) for line in boilerplate)
    in_breadcrumb = False
    for raw in text.split("\n"):
        stripped = raw.strip()
        underlined, last_line = last_line, None
        if not stripped:
            in_breadcrumb = False
            out.append("")
            continue
        if template_key(stripped) in template:
            removed["boilerplate_line"] += 1
            in_breadcrumb = True
            continue
        if (
            _ANGLE_SEPARATOR.match(stripped)
            or (in_breadcrumb and _BREADCRUMB_CONTINUATION.match(stripped))
            or (_BREADCRUMB_SEPARATOR.search(stripped) and _LINK.search(stripped))
        ):
            removed["breadcrumb_line"] += 1
            in_breadcrumb = True
            continue
        in_breadcrumb = False
        if _RULE.match(raw):
            # Directly under a text line, "===" and "---" are setext h1 and h2
            # underlines; anywhere else they are horizontal rules.
            if underlined and stripped[0] in "=-":
                level = 1 if stripped[0] == "=" else 2
                headings.append((level, underlined))
                if level == 1 and h1 is None:
                    h1 = underlined
            removed["rule_or_underline"] += 1
            continue
        if _TABLE_RULE.match(raw):
            removed["table_rule"] += 1
            continue

        line = raw
        if _TABLE_ROW.match(line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if not cells:
                removed["empty_table_row"] += 1
                continue
            line = "; ".join(cells)
            removed["table_row"] += 1

        line = _strip_block_markers(line, removed)
        level = 0
        heading = _ATX.match(line)
        if heading:
            line = heading.group(2)
            level = len(heading.group(1))
            removed["heading_marker"] += 1

        cleaned, found = _clean_inline(line, page_url, host, removed)
        # A glyph can surface only once inline markup is gone: "** \u2013 text".
        unglyphed = _GLYPH_BULLET.sub("", cleaned, count=1)
        if unglyphed != cleaned:
            removed["glyph_bullet"] += 1
            cleaned = unglyphed
        if not found and _PUNCTUATION_ONLY.fullmatch(cleaned):
            removed["punctuation_line"] += 1
            continue
        if not cleaned:
            continue
        if level:
            headings.append((level, cleaned))
            if level == 1 and h1 is None:
                h1 = cleaned
        else:
            last_line = cleaned
        for anchor, target, internal in found:
            try:
                links.append(
                    ExtractedLink(
                        target_url=HttpUrl(target),
                        anchor_text=anchor,
                        surrounding_text=_sentence_containing(cleaned, anchor),
                        is_internal=internal,
                    )
                )
            except ValidationError:
                removed["invalid_link"] += 1
        out.append(cleaned)

    return CleanedPage(
        url=HttpUrl(page_url),
        title=clean_meta(title),
        h1=h1,
        headings=tuple(headings),
        body_text=_join(out),
        links=tuple(links),
        removed=tuple(sorted(removed.items())),
    )


def _strip_block_markers(line: str, removed: Counter[str]) -> str:
    """Strip blockquote, list and glyph-bullet markers, which nest: ``> * ^ text``."""
    markers = (
        (_BLOCKQUOTE, "blockquote_marker"),
        (_LIST_MARKER, "list_marker"),
        (_GLYPH_BULLET, "glyph_bullet"),
    )
    while True:
        new = line
        for pattern, kind in markers:
            stripped = pattern.sub("", new, count=1)
            if stripped != new:
                removed[kind] += 1
                new = stripped
        if new == line:
            return line
        line = new


def _clean_inline(
    line: str, page_url: str, host: str, removed: Counter[str]
) -> tuple[str, list[tuple[str, str, bool]]]:
    found: list[tuple[str, str, bool]] = []

    def drop(kind: str) -> str:
        removed[kind] += 1
        return " "

    def link(match: re.Match[str]) -> str:
        anchor = _clean_text(match.group("text"))
        target = _resolve(match.group("dest"), page_url)
        if target is None:
            removed["non_web_link"] += 1
            return anchor
        if target == page_url:
            removed["same_page_link"] += 1
            return anchor
        removed["link_markup"] += 1
        if not anchor:
            removed["empty_anchor_link"] += 1
            return " "
        found.append((anchor, target, _host(target) == host))
        # A url used as its own anchor is still a link, but the url is noise in
        # the body text.
        return " " if _BARE_URL.fullmatch(anchor) else anchor

    line = _LINKED_IMAGE.sub(lambda _: drop("linked_image"), line)
    line = _IMAGE.sub(lambda _: drop("image"), line)
    line = _LINK.sub(link, line)
    line = _ORPHAN_LINK_TAIL.sub(lambda _: drop("orphan_link_markup"), line)
    line = _AUTOLINK.sub(lambda _: drop("url"), line)
    line = _BARE_URL.sub(lambda _: drop("url"), line)
    return _clean_text(line), found


def _clean_text(text: str) -> str:
    text = _REPLACEMENT_CHAR.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _STRONG.sub(r"\2", text)
    text = _EM_STAR.sub(r"\1", text)
    text = _EM_UNDERSCORE.sub(r"\1", text)
    text = _ORPHAN_STRONG.sub("", text)  # a "**" whose partner is on another line
    text = _HTML_TAG.sub(" ", text)
    text = _ESCAPE.sub(r"\1", text)
    text = html.unescape(text)
    return _SPACES.sub(" ", text).strip()


def _resolve(destination: str, page_url: str) -> str | None:
    destination = destination.strip().strip("<>")
    if not destination or destination.lower().startswith(_NON_WEB_SCHEMES):
        return None
    return _normalise(urljoin(page_url, destination))


def _normalise(url: str) -> str | None:
    """Absolute http(s) url with lower-cased scheme and host and no fragment."""
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, "")
    )


def _host(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host.removeprefix("www.")


def _sentence_containing(text: str, anchor: str) -> str:
    for sentence in _SENTENCE_END.split(text):
        if anchor in sentence:
            return sentence
    return text


def _join(lines: list[str]) -> str:
    """Join cleaned lines, keeping one blank line between paragraphs and gluing
    a punctuation fragment back onto the line it continues."""
    result: list[str] = []
    for line in lines:
        if not line and (not result or not result[-1]):
            continue
        if line and _CONTINUATION.match(line) and result and result[-1]:
            result[-1] += line
            continue
        result.append(line)
    while result and not result[-1]:
        result.pop()
    return "\n".join(result)
