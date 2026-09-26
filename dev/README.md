# Dev environment — intelligence layer only

Neo4j and MongoDB. Nothing else. MinIO and MLflow stay on your server; point
`.env` at them when you need model tracking.

No crawler, no GSC client. A generator produces a synthetic corpus with
**known ground truth**, so you can build the audit, clustering, retrieval and
extraction logic and assert against expected counts.

## Start

```bash
make up        # neo4j + mongo, schema applied
make seed      # 614 pages, template content, synthetic vectors — instant
make verify    # assert planted ground truth survived
make sanity    # confirm it landed in the databases
```

Browser at http://localhost:7474, `neo4j` / `localdevpassword`.

## Two backends, chosen independently

```bash
make seed         # template content + synthetic vectors   instant, offline
make seed-prose   # LLM prose  + synthetic vectors         ~10 min, Gemini only
make seed-real    # LLM prose  + Voyage embeddings         ~15 min, costs money
```

| Backend | What it proves |
|---|---|
| `content=template` | Nothing about text quality. Structurally regular prose, useful only for plumbing |
| `content=llm` | Real `<h1>`–`<h4>` hierarchy, real anchors inside real sentences. Exercises the HTML extractor |
| `embeddings=synthetic` | Clustering and retrieval **plumbing**. Not quality |
| `embeddings=voyage` | The only configuration whose retrieval numbers mean anything |

Use `seed` while iterating on algorithms. Use `seed-real` before trusting any
retrieval or clustering quality number — that is the whole reason the LLM
backend exists.

Both are cached: content by `(topic, subtopic, include_head_term, page_type,
title)` and embeddings by `(model, dimension, text)`. Regenerating with the same
seed costs nothing. Embedding cache keys include the model, because keying on
text alone would silently serve vectors from a different space after a model
change.

## Ground truth survives both backends

An LLM told "never write this phrase" sometimes writes it anyway, and an LLM
told to embed the anchor `"click here"` will helpfully improve it — either of
which silently destroys a planted fixture.

So every generation is verified against its constraints and regenerated on
failure, up to three attempts with a sharpened prompt. Pages that still fail
fall back to template content and are tagged `CONTENT_FALLBACK` in `_planted`,
so a failure is visible rather than silent.

```bash
make verify        # template content
make verify-real   # LLM content
```

```
PASS  head-term constraint     0 violations
PASS  links embedded exactly   missing=0 anchor_rewritten=0
PASS  heading hierarchy        no_h1=0 h5_h6=0
PASS  body_text stripped       tag_leak=0
PASS  utility pages orphaned   14/14
PASS  bridge gaps separated    gaps=[2, 0] min_non_gap=50
PASS  body links only          {'body'}
PASS  no-head-term rate        19.3%
PASS  no content fallbacks     0 pages fell back to template
```

## Package layout

```
scripts/
├── generate_corpus.py     CLI, wires the stages together
├── verify_corpus.py       ground-truth assertions, no DB needed
└── corpus/
    ├── taxonomy.py        topics, planted structure, CTR curve — the spec
    ├── structure.py       pages, body links, GSC metrics (deterministic)
    ├── content.py         template + LLM backends, constraint verification
    ├── embeddings.py      synthetic + Voyage backends, token batching
    └── writers.py         Neo4j and Mongo
```

`taxonomy.py` is the specification. Change a topic, a spread or a bridge gap
there and the assertion suite changes with it.

## Body links only

The crawler extracts links from body content. Nav, header, footer and sidebar
links are never captured, so they never enter Neo4j and no algorithm sees them.
That matches how SEOs work — template links are audited once, per template, not
per page.

The corpus reflects this: 14 nav/footer utility pages exist and are reachable on
a real site, but nothing in body copy references them, so they appear as orphans.

Measured justification, before the machinery was removed: with template links
included at a realistic 77% ratio, 10/10 of the top PageRank and betweenness
pages were `/terms`, `/contact`, `/sitemap`. Clustering was unaffected — Leiden
recovered the planted topics either way, ARI delta −0.007. So the extraction rule
matters for authority and centrality, not for clustering.

## Running the comparison

```bash
make eval             # Leiden vs HDBSCAN against ground truth
make eval-umap        # HDBSCAN after UMAP to 10d
make eval-stability   # label reproducibility across 4 runs
```

Reports ARI and NMI against both the topic and subtopic partitions, noise
precision and recall, and per-topic purity — that last one is where density
sensitivity shows up, since a global resolution parameter cannot serve a
cluster at spread 0.18 and one at 0.55 equally.

Then two things only HDBSCAN enables:

**Hub bridges.** Ranks cluster pairs by `0.4 × centroid similarity + 0.6 ×
query Jaccard − link density`. The two planted gaps should come out top. This
is the hub-to-hub structural recommendation — a handful of high-leverage links
rather than thousands of page-level ones.

**Pillar check.** Compares the declared pillar against the page nearest the
cluster centroid. The two planted mismatches should be caught.

### What would settle the argument

```
HDBSCAN wins if   noise_recall is high with high precision,
                  purity holds across both tight and diffuse topics,
                  ari_subtopic is meaningfully above Leiden's,
                  and both bridge gaps rank top by bridge_gap

Leiden wins if    ari_topic is comparable and label stability is better,
                  since Leiden with a fixed seed is reproducible and
                  HDBSCAN's labels churn between runs
```

Stability is the one to watch. Cluster ids feed the model as a categorical
feature — if `make eval-stability` shows pairwise ARI below ~0.8, HDBSCAN's
labels are too volatile to use directly and you would need centroid matching
between runs to keep hub ids stable.

## Leiden vs Louvain

```bash
make bench
```

Projects the graph, runs both in `stats` mode, then writes Leiden and shows
whether the recovered communities line up with the planted topics. Community
Edition caps GDS at 4 cores, so `concurrency: 4` is set explicitly.

## Projection hygiene

Community allows 3 in-memory projections per instance. A run that dies without
dropping its projection leaks it, and the *next* run fails on a limit error
that points nowhere near the real cause.

```bash
make projections        # what's held
make drop-projections   # clear leaks
```

Worth building the context-manager pattern into your Prefect tasks from the
first PR rather than retrofitting it.

## Scaling up

```bash
make seed-big   # 5000 pages
```

Use this when you're working on chunked LambdaMART inference or checking that
betweenness sampling holds up. At 500 pages everything is fast enough that
performance problems stay invisible.

## Notes

**Mongo runs as a single-node replica set** so transactions and change streams
work. Connection strings need `?directConnection=true` or the driver tries to
discover other members and hangs.

**Neo4j 5.26** — `vector.similarity.cosine()` needs 5.18+, and the audit query
uses it. `make verify` checks.

**Memory**: Neo4j is capped at 4 GB, Mongo at 2 GB. Give Docker Desktop 8 GB.

## When you outgrow this

The generator replaces the crawler and GSC pull, which is the right trade while
you're building the intelligence layer. But synthetic body text has regular
structure that real pages don't, so extraction hit rates measured here will be
optimistic. Treat the first real corpus as the moment those numbers become
meaningful.
