# Internal Linking Intelligence Engine

Multi-tenant SEO internal linking engine. Audits a site's existing internal links, discovers the ones it's missing, and derives anchor text from phrases already present in the copy.

**Spec version:** v6 · **Stack:** Python 3.13 / FastAPI / Neo4j / igraph / MLflow

---

## Start here

| Page | What's in it |
|---|---|
| [[Architecture]] | The eleven stages, what each produces |
| [[Stack]] | Every dependency and why it's there |
| [[ML-Components]] | The nine models: input, output, task type |
| [[Data-Model]] | Neo4j schema, MongoDB collections, Pydantic contracts |
| [[Development]] | Local setup, running the pipeline, testing |
| [[Setup]] | **Start here.** Install, tooling, quality gates, repo layout |
| [[Core-Build-Plan]] | Embedding → validated intelligence core, ~28 days |
| [[PR-Roadmap]] | Full production scope, 68 PRs across M0–M8 |
| [[Decisions]] | What was rejected and why |
| [[Measurement-Backlog]] | Every unresolved assumption, with how to test it |

---

## Diagrams

Rendered walkthroughs of the same system at four altitudes.

| Diagram | What it covers |
|---|---|
| [The case for it](https://ske4032.github.io/internal-link-engine/files/linking-engine-docs/diagrams/linking-engine-pitch.html) | Why the system exists, without the machinery |
| [Technical walkthrough](https://ske4032.github.io/internal-link-engine/files/linking-engine-docs/diagrams/linking-engine-tech.html) | The eleven stages — what each computes and runs on |
| [Nine models deep](https://ske4032.github.io/internal-link-engine/files/linking-engine-docs/diagrams/ml-stack.html) | Every ML layer, in the order text passes through it |
| [From vector to recommendation](https://ske4032.github.io/internal-link-engine/files/linking-engine-docs/diagrams/pipeline-walkthrough.html) | Embedding to ranked, anchored recommendation |

`architecture-diagram.jsx` and `system-design.jsx` ship as React source and need
a build step to view.

---

## Scope note

[[Core-Build-Plan]] is the current build: Voyage embedding through to a validated
intelligence core, run against synthetic ground truth. Roughly 25 days, or 18–22
with LLM assistance.

[[PR-Roadmap]] is the full production scope — 158 days including crawler, GSC,
REST API, and multi-tenancy. Most of that is deferred or already built elsewhere.

## The two stages

**Stage 1 — Audit.** Scores every existing `LINKS_TO` edge. Emits `REANCHOR`, `REMOVE`, `FIX`. No trained model required.

**Stage 2 — Discovery.** Finds pairs that should be linked but aren't. Emits `ADD_LINK`. Requires GraphSAGE, cross-attention, LambdaMART.

Both share anchor resolution and feed one ranked queue.

**Stage 1 ships first.** It needs no model, and it produces the acceptance feedback Stage 2's models train on. Building discovery first means months of models with nothing to learn from.

---

## Five action types

```
ADD_LINK      no link exists, should
REANCHOR      link exists, anchor text is generic or misaligned
REMOVE        link dilutes equity, no topical justification
FIX           broken, redirected, or nofollowed
CONTENT_GAP   no linkable phrase exists — routes to a writer, not a link queue
```

---

## Non-negotiables

- **Anchor text is extracted, never generated.** The keyword is the specification; find the phrase already in the copy. If none exists, emit `CONTENT_GAP`.
- **Eligibility is not priority.** No hardcoded thresholds upstream of the ranker. Everything the old GSC filter decided is a feature now.
- **Every model artefact is versioned and gated.** A retrained ranker ships only if it beats production NDCG@10 on holdout.
- **Pydantic at every boundary.** Crawler output, API payloads, model features, tenant config. No dicts crossing module lines.
- **Jaccard where set sizes are comparable, cosine where they are not.** A 3-token anchor against a 2,000-token page is a cosine problem; two 40-query sets are a Jaccard problem.
- **Graph algorithms run outside the database.** igraph and `leidenalg`, not GDS. Neo4j stores edges and vectors.
- **Body links only.** Nav, header, footer and sidebar links are never captured. Template links are an SEO concern audited per template, not per page.
