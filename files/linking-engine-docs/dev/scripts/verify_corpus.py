"""
Corpus integrity checks.

Runs against the in-memory corpus without touching Neo4j or Mongo, so it is
cheap enough to run after every generator change.

    uv run python scripts/verify_corpus.py
    uv run python scripts/verify_corpus.py --content llm    # verify real prose

These assert that the planted ground truth actually survived generation. The
LLM backend is the reason this exists: a model told "never write this phrase"
will sometimes write it anyway, and a model told to embed "click here" will
helpfully improve it — either of which silently destroys a fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from corpus import content as content_mod
from corpus.taxonomy import BRIDGE_GAPS, TOPICS
from generate_corpus import build


def run_checks(pages, links, links_by_source, truth) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    topical = [p for p in pages if p.topic]

    viol = [p.url for p in topical
            if (TOPICS[p.topic]["head"].lower() in p.body_text.lower())
            != p.include_head_term]
    checks.append(("head-term constraint", not viol,
                   f"{len(viol)} violations" + (f" e.g. {viol[0]}" if viol else "")))

    missing = rewritten = 0
    for p in pages:
        for l in links_by_source.get(p.url, []):
            if f'href="{l.target}"' not in p.html:
                missing += 1
            elif l.anchor_text not in p.html:
                rewritten += 1
    checks.append(("links embedded exactly", missing + rewritten == 0,
                   f"missing={missing} anchor_rewritten={rewritten}"))

    no_h1 = sum(1 for p in pages if "<h1" not in p.html)
    deep = sum(1 for p in pages if re.search(r"<h[56]", p.html, re.I))
    checks.append(("heading hierarchy", no_h1 + deep == 0,
                   f"no_h1={no_h1} h5_h6={deep}"))

    leak = sum(1 for p in pages if "<" in p.body_text)
    checks.append(("body_text stripped", leak == 0, f"tag_leak={leak}"))

    inbound = {l.target for l in links}
    util = [p for p in pages
            if any(k in p.planted for k in ("NAV_TARGET", "FOOTER_TARGET"))]
    orphaned = sum(1 for p in util if p.url not in inbound)
    checks.append(("utility pages orphaned", orphaned == len(util),
                   f"{orphaned}/{len(util)}"))

    lab = {p.url: p.topic for p in pages}
    ct: collections.Counter = collections.Counter()
    for l in links:
        a, b = lab.get(l.source), lab.get(l.target)
        if a and b and a != b:
            ct[tuple(sorted((a, b)))] += 1
    gaps = {tuple(sorted(x)) for x in BRIDGE_GAPS}
    gv = [ct.get(g, 0) for g in gaps]
    ov = [v for k, v in ct.items() if k not in gaps]
    checks.append(("bridge gaps separated", max(gv) < min(ov),
                   f"gaps={gv} min_non_gap={min(ov)}"))

    positions = {l.link_position for l in links}
    checks.append(("body links only", positions == {"body"}, str(positions)))

    rate = len([p for p in pages if "NO_HEAD_TERM" in p.planted]) / len(topical)
    checks.append(("no-head-term rate", 0.15 < rate < 0.30, f"{rate:.1%}"))

    fallbacks = [p.url for p in pages
                 if any(x.startswith("CONTENT_FALLBACK") for x in p.planted)]
    checks.append(("no content fallbacks", not fallbacks,
                   f"{len(fallbacks)} pages fell back to template"))

    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--content", choices=["template", "llm"], default="template")
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()

    pages, links, queries, truth = build(a.pages, a.seed)
    links_by_source: dict[str, list] = {}
    for l in links:
        links_by_source.setdefault(l.source, []).append(l)

    stats = asyncio.run(content_mod.generate_all(
        pages, links_by_source, a.content, a.seed, concurrency=a.concurrency))

    print(f"corpus: {len(pages)} pages · {len(links)} links · "
          f"{len(queries)} queries · content={a.content}")
    print(f"generation: {stats}\n")

    checks = run_checks(pages, links, links_by_source, truth)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:24s} {detail}")

    failed = [c for c in checks if not c[1]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
