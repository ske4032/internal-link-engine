# GitHub issue import

33 issues covering the Core Build Plan: Voyage embedding through to a validated
intelligence core. 28.5 engineer-days, 20–24 with LLM assistance.

Each issue is written to be worked by someone unfamiliar with the project:
**Why** it exists, **Steps a–f** with the actual code and queries, **Acceptance**
criteria, and **Gotchas** — the mistakes that fail silently rather than loudly.

## Files

| File | Use |
|---|---|
| `issues.json` | Source of truth. Includes `depends_on` and `estimate_days` |
| `issues.csv` | For CSV importers, or paste into a project board |
| `create_issues.sh` | `gh` CLI script. Creates labels, milestone, all issues |

## Import

```bash
gh auth login
cd path/to/repo
./create_issues.sh            # dry run
./create_issues.sh --apply    # create for real
```

Creates 11 labels, a "Core Engine" milestone, and 33 issues in dependency order,
so issue numbers match the `[NN]` prefixes and cross-references resolve.

## Phases

```
0  Foundation          2.0 d   scaffold, Pydantic models, schema @2048d
1  Embedding           3.0 d   Voyage client, token batching, write path
2  Graph analytics     3.0 d   igraph, Leiden ×2, HDBSCAN     ← GATE
3  Retrieval           3.0 d   HNSW, Jaccard signals, clusters
4  Features + scoring  4.0 d   ~30 features, heuristic baseline, hub bridges
5  Anchor resolution   4.0 d   keyword chain, ladder rungs 1–3
6a Ranking, synthetic  4.0 d   proxy labels, LambdaMART, MLflow
6b Ranking, real       2.0 d   real crawl, blind labelling      ← GATE
7  Validation          2.0 d   10 pass/fail checks
```

## Five gates

Issues labelled `gate` are decision points. Two of them can delete work:

| Gate | Decides |
|---|---|
| **#11** | Whether HDBSCAN (#10) and betweenness (#8) survive at all |
| **#14** | Whether anything is still filtering candidates it shouldn't |
| **#18** | Whether hub-bridge scoring finds the planted gaps |
| **#23** | Whether extraction hits the expected ~78% / ~22% split |
| **#30** | **Does the intelligence core work** — the real one |

## Three issues to read first

**#11 — Phase 2 gate.** Run before writing Phase 3. Two outcomes delete later
work: if `spearman(betweenness, pagerank) > 0.8`, issue #8 comes out of the
pipeline; if `ARI(leiden, hdbscan) > 0.85`, issue #10 was a duplicate feature.

**#24 — proxy labels.** Explains what synthetic ranking validation can and cannot
prove. A model that learns planted labels perfectly has learned your priors.

**#30 — real labels.** The gate that actually answers "does the intelligence core
work." Everything before it validates plumbing.
