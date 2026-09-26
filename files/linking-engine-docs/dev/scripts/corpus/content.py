"""
Page content: real HTML prose.

Two backends.

  template   deterministic, offline, instant. Structurally regular, which makes
             it useless for judging embedding quality but fine for plumbing.

  llm        Gemini via google-genai. Real prose, real heading hierarchy, real
             anchors embedded in text. This is what you need before trusting any
             retrieval number, because synthetic vectors only ever prove the
             wiring works.

The critical constraint is `include_head_term`. The structure layer decides
whether a page contains its own head term verbatim; that flag drives ~22% of
pages past extraction rung 1 and into variants, Jaccard, semantic and finally
CONTENT_GAP. An LLM will happily ignore a negative instruction, so every
generation is verified and regenerated on failure.

Anchors are verified too. Told to embed "click here", a model will helpfully
rewrite it into something descriptive — which silently destroys the planted
REANCHOR fixtures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from pathlib import Path

from corpus.structure import Link, Page
from corpus.taxonomy import TOPICS

CACHE_DIR = Path(".cache/content")
MODEL_ID = "gemini-2.5-flash"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    """What gets embedded. Voyage should see prose, not markup — but the HTML is
    kept, because the extractor needs it and offsets differ between the two."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# ── template backend ────────────────────────────────────────────────────────

def template_html(page: Page, links: list[Link], rng: random.Random) -> str:
    if page.topic is None:
        body = " ".join(rng.sample([
            "Details are available on request.",
            "This page is reviewed periodically.",
            "Contact the office for anything not covered here.",
            "We publish updates when circumstances change.",
        ], k=3))
        return f"<body><h1>{page.h1}</h1><p>{body}</p></body>"

    t = TOPICS[page.topic]
    terms = t["subtopics"][page.subtopic]
    paragraphs = max(4, page.word_count // 220)
    link_queue = list(links)
    out = [f"<body><h1>{page.h1}</h1>"]

    for i in range(paragraphs):
        if i % 3 == 0:
            out.append(f"<h2>{rng.choice(terms).title()}</h2>")
        elif i % 3 == 1:
            out.append(f"<h3>Working with {rng.choice(terms)}</h3>")

        picks = rng.sample(terms, k=min(2, len(terms)))
        sentences = [
            f"When {rng.choice(t['verbs'])} at volume, {picks[0]} governs the result.",
            f"A correctly specified {picks[-1]} reduces rework across the line.",
            "Teams tend to overlook this until throughput starts slipping.",
        ]
        if page.include_head_term and i == max(1, paragraphs // 2):
            sentences.insert(
                1, f"Selecting the right {t['head']} depends on volume and tolerance.")
        if link_queue:
            l = link_queue.pop(0)
            sentences.append(
                f'For more on this, see <a href="{l.target}">{l.anchor_text}</a>.')
        rng.shuffle(sentences)
        out.append(f"<p>{' '.join(sentences)}</p>")

    # any links that didn't fit go in a final paragraph — every planted link
    # must appear in the HTML or the audit fixtures break
    if link_queue:
        tail = " ".join(
            f'See <a href="{l.target}">{l.anchor_text}</a>.' for l in link_queue)
        out.append(f"<p>{tail}</p>")

    out.append("</body>")
    return "".join(out)


# ── llm backend ─────────────────────────────────────────────────────────────

def build_prompt(page: Page, links: list[Link]) -> str:
    if page.topic is None:
        return f"""Write a short HTML page (120-250 words) inside <body> tags.

Title: {page.h1}
This is a utility page — company information, policy, or admin. It is NOT about
any technical product topic. Keep it plainly administrative.

Use <h1> then <h2>. Output raw HTML only, no markdown fences."""

    t = TOPICS[page.topic]
    terms = t["subtopics"][page.subtopic]
    head = t["head"]

    if page.include_head_term:
        constraint = (
            f'Use the exact phrase "{head}" at least twice in the prose, '
            f"naturally, inside sentences."
        )
    else:
        constraint = (
            f'CRITICAL CONSTRAINT: never write the phrase "{head}", and never '
            f'write any close variant of it (no singular/plural forms, no '
            f'reordering of those words). Discuss the subject using only these '
            f'terms: {", ".join(terms)}. This constraint matters more than style.'
        )

    style = (
        "Commercial product page. Conversion-focused, specifications, "
        "enterprise benefits, integration notes."
        if page.page_type in ("PRODUCT", "CATEGORY")
        else "In-depth technical article. Educational, analytical, "
             "problem breakdown then implementation steps."
    )

    link_spec = json.dumps(
        [{"href": l.target, "anchor": l.anchor_text} for l in links], indent=2)

    return f"""Write an HTML article of roughly {page.word_count} words inside <body> tags.

Title (H1): {page.h1}
Subject area: {page.topic} / {page.subtopic}
Terms to use naturally: {", ".join(terms)}

Style: {style}

{constraint}

Embed these internal links contextually inside prose sentences:
{link_spec}

LINK RULES — these are strict:
- Use the anchor text EXACTLY as given. Do not rephrase, expand, capitalise
  differently, or improve it. If an anchor reads poorly, use it anyway.
- Each link appears exactly once, inside a sentence, never in a list or heading.
- Format: <a href="HREF">ANCHOR</a>

FORMATTING:
- <h1> once, then logical <h2>, <h3>, <h4>. Never <h5> or <h6>.
- Output raw HTML inside <body>...</body> only. No markdown code fences.
"""


def verify(page: Page, links: list[Link], html: str) -> list[str]:
    """Returns a list of violations. Empty means the generation is usable."""
    problems: list[str] = []
    text = strip_html(html).lower()

    if page.topic is not None:
        head = TOPICS[page.topic]["head"].lower()
        present = head in text
        if present != page.include_head_term:
            problems.append(
                f"head_term_{'present' if present else 'absent'}_"
                f"expected_{'present' if page.include_head_term else 'absent'}")

    for l in links:
        if f'href="{l.target}"' not in html:
            problems.append(f"missing_href:{l.target}")
        elif l.anchor_text not in html:
            problems.append(f"anchor_rewritten:{l.anchor_text}")

    if "<h1" not in html:
        problems.append("no_h1")
    if re.search(r"<h[56]", html, re.I):
        problems.append("h5_or_h6_used")

    words = len(text.split())
    if page.topic is not None and words < 300:
        problems.append(f"too_short:{words}")

    return problems


async def generate_llm(
    page: Page, links: list[Link], client, semaphore: asyncio.Semaphore,
    max_attempts: int = 3, use_cache: bool = True,
) -> tuple[str, list[str]]:
    """Returns (html, unresolved_problems). Retries on constraint violation,
    since a negative instruction is the one models most often ignore."""
    from google.genai import types

    key = hashlib.sha256(
        f"{page.content_key}|{json.dumps([l.anchor_text for l in links])}".encode()
    ).hexdigest()[:24]
    cache_file = CACHE_DIR / f"{key}.html"

    if use_cache and cache_file.exists():
        html = cache_file.read_text()
        return html, verify(page, links, html)

    prompt = build_prompt(page, links)
    last_problems: list[str] = ["not_attempted"]

    async with semaphore:
        for attempt in range(max_attempts):
            try:
                resp = await client.aio.models.generate_content(
                    model=MODEL_ID, contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7 if attempt == 0 else 0.4),
                )
                html = resp.text.strip()
                html = re.sub(r"^```(?:html)?\s*|\s*```$", "", html)

                last_problems = verify(page, links, html)
                if not last_problems:
                    if use_cache:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        cache_file.write_text(html)
                    return html, []

                # sharpen the prompt for the retry
                prompt = build_prompt(page, links) + (
                    f"\n\nPREVIOUS ATTEMPT FAILED: {', '.join(last_problems)}. "
                    f"Fix these specifically."
                )
            except Exception as exc:  # noqa: BLE001 — surfaced to the caller
                last_problems = [f"api_error:{type(exc).__name__}"]
                await asyncio.sleep(2 ** attempt)

    return "", last_problems


async def generate_all(
    pages: list[Page], links_by_source: dict[str, list[Link]],
    backend: str, seed: int, concurrency: int = 5, use_cache: bool = True,
) -> dict[str, int]:
    """Fills page.html and page.body_text in place. Returns a failure summary."""
    stats = {"generated": 0, "cached": 0, "fallback": 0, "violations": 0}

    if backend == "template":
        rng = random.Random(seed)
        for p in pages:
            p.html = template_html(p, links_by_source.get(p.url, []), rng)
            p.body_text = strip_html(p.html)
            p.content_hash = hashlib.sha256(p.body_text.encode()).hexdigest()[:16]
            stats["generated"] += 1
        return stats

    from google import genai

    client = genai.Client()
    semaphore = asyncio.Semaphore(concurrency)
    rng = random.Random(seed)

    async def one(p: Page) -> None:
        links = links_by_source.get(p.url, [])
        html, problems = await generate_llm(p, links, client, semaphore,
                                            use_cache=use_cache)
        if html and not problems:
            p.html = html
            stats["generated"] += 1
        else:
            # A failed page must not silently become a page with no planted
            # structure — fall back to the template so the fixture survives.
            p.html = template_html(p, links, rng)
            stats["fallback"] += 1
            if problems:
                stats["violations"] += 1
                p.planted.append(f"CONTENT_FALLBACK:{problems[0]}")
        p.body_text = strip_html(p.html)
        p.content_hash = hashlib.sha256(p.body_text.encode()).hexdigest()[:16]

    await asyncio.gather(*(one(p) for p in pages))
    return stats
