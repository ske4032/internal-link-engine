# Fixed in this pass

## Diagrams — 5-action-type / body-links-only staleness

All six jsx/html diagram files predated the decision removing REPOSITION as an
action type (the crawler extracts body links only, so nav/footer links are
never captured and there's nothing to move a link out of). Fixed:

- architecture-diagram.jsx — node label and rationale panel body both listed
  REPOSITION; panel now explains why it was dropped instead
- system-design.jsx — node subtitle said "(6 types)", now "(5 types)"
- linking-engine-tech.html — code block listed REPOSITION; prose said
  "Six action types" and "four of the six action types" (two separate spots)
- linking-engine-pitch.html — had a full interactive "Move it" tab and data
  card built on REPOSITION; tab, data object, and the "Six things" heading
  all removed/corrected to five
- pipeline-walkthrough.html — verdict list had a REPOSITION entry; a separate
  mechanics block also said "six action types" (found on a second sweep,
  missed on the first)
- ml-stack.html — already clean, no changes

Verified with a case-insensitive sweep for REPOSITION / "six action" /
"(6 types)" / "four of the six" across all six files after editing. Two
intentional mentions remain (architecture-diagram.jsx and
linking-engine-tech.html), both explaining *why* REPOSITION doesn't exist —
correct, not stale.

## Roadmap

- wiki/PR-Roadmap.md #7 — "structlog -> Splunk HEC" was stale (Splunk deferred
  to production per ADR-009). Now points at Promtail/Loki.

Everything else (wiki/*, ADRs.md, github-import/*, config/*, dev/*) was
grep-verified clean: Prefect deployed and kept, observability deferred to
production, 2048d embeddings throughout, 5 action types, body-links-only graph.
