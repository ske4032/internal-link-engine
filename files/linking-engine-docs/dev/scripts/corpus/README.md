# scripts/ — corpus generation and evaluation

Everything here exists to answer one question before real clients are involved:
**does the intelligence core behave correctly?** The synthetic corpus plants a
known right answer; the scripts generate it, verify it survived generation, and
score clustering against it.

```
scripts/
├── generate_corpus.py      CLI — build the corpus, write to Neo4j + Mongo
├── verify_corpus.py        CLI — assert planted ground truth survived (no DB)
├── eval_clustering.py      CLI — Leiden vs HDBSCAN against ground truth (reads DB)
└── corpus/
    ├── taxonomy.py         the specification: topics, planted structure, CTR curve
    ├── structure.py        pages, body links, GSC queries, ground-truth report
    ├── content.py          page HTML: template or LLM backend, with verification
    ├── embeddings.py       page vectors: synthetic or Voyage backend
    └── writers.py          persistence to Neo4j and MongoDB
```

---

## Quick start

```bash
make up                  # neo4j + mongo
make seed                # generate_corpus.py, template + synthetic, instant
make verify              # verify_corpus.py, expect 9/9
make eval                # eval_clustering.py, ARI / noise / purity
```

---

## How the modules fit together

```
taxonomy.py ──┬──► structure.py ──► content.py ──► embeddings.py ──► writers.py
              │         │               │                │
              │    pages, links,    page.html,       page.embedding,
              │    queries, truth   page.body_text   query["embedding"]
              │
              └──► read by every other module as the single source of truth

generate_corpus.py   runs the whole chain and writes the result
verify_corpus.py     runs structure + content in memory, asserts, writes nothing
eval_clustering.py   reads what generate_corpus.py wrote, scores clustering
```

Stages are strictly ordered. `structure.py` decides **what** each page is about
and whether it may contain its own head term. `content.py` only has to honour
that. `embeddings.py` only has to vectorise what `content.py` produced. No stage
reaches back and changes a decision made upstream.

---

## Two independent backends

Content and embeddings are chosen separately:

| `--content` | `--embeddings` | Use for | Cost |
|---|---|---|---|
| `template` | `synthetic` | Plumbing, algorithm iteration, CI | Free, instant |
| `llm` | `synthetic` | Testing HTML extraction on real markup | Gemini only, ~10 min |
| `llm` | `voyage` | Any retrieval or clustering **quality** number | Gemini + Voyage, ~15 min |
| `template` | `voyage` | Rarely useful — real vectors over unrealistic prose | Voyage only |

Only `llm` + `voyage` produces numbers worth trusting for quality. Everything else
proves the wiring works.

---

## `corpus/taxonomy.py`

**The specification.** Every planted structure the assertion suite checks against
is declared here. Change a topic, spread or bridge gap and the expected results
change with it — treat edits here as changing the test, not tuning a parameter.

| Constant | What it controls |
|---|---|
| `DIM = 2048` | Embedding dimension. Must match the Neo4j vector index and voyage-4-large's `output_dimension` |
| `TOPICS` | 5 topics × 12 subtopics. Each has a `head` term, a `spread` (cluster density), subtopic terms and verbs |
| `BRIDGE_GAPS` | Topic pairs that share search demand but almost never link. Hub-to-hub scoring must rank these top |
| `WELL_CONNECTED` | A densely linked pair, for contrast |
| `PILLAR_MISMATCH` | Topics whose declared pillar is deliberately displaced from its cluster centre |
| `NO_HEAD_TERM_RATE = 0.22` | Share of pages whose body must *not* contain their own head term. Drives the extraction ladder past rung 1 |
| `P_CROSS_*` | Cross-topic link probabilities — baseline 0.28, bridge gaps 0.01, well-connected 0.55 |
| `GENERIC_ANCHORS` | "click here", "read more"… from WCAG 2.4.4 / 2.4.9 link-text guidance |
| `NAV_TARGETS`, `FOOTER_TARGETS` | Utility pages that exist but receive no body links — orphans in the graph, by design |
| `NOISE_SUBJECTS` | Off-topic pages that *do* receive body links. HDBSCAN should label them −1 |
| `CTR_CURVE`, `ctr_at()` | CTR by position. Table for 1–10, power-law decay beyond |
| `PAGE_TYPE_*` | Lognormal impression means and exponential position scales per page type |

**Why `spread` matters.** It ranges 0.18 (pressbrake, tight) to 0.55 (maintenance,
diffuse). Leiden's resolution is global and cannot serve both; HDBSCAN adapts per
cluster. That range is what makes the comparison meaningful.

---

## `corpus/structure.py`

**Deterministic site structure.** No LLM, no network. Same seed, identical corpus
every time — which is what makes the assertions meaningful.

### Dataclasses

**`Page`** — one URL. Key fields beyond the obvious:

| Field | Purpose |
|---|---|
| `include_head_term` | Whether the body may contain the page's own head term. Set here, honoured by `content.py` |
| `html` / `body_text` | Empty here. Filled by `content.py` |
| `embedding` | Empty here. Filled by `embeddings.py` |
| `planted` | Ground-truth tags: `DECLARED_PILLAR`, `PILLAR_MISMATCH`, `NO_HEAD_TERM`, `TRUE_NOISE`, `NAV_TARGET`, `FOOTER_TARGET`, `ORPHAN_NEW_STRATEGIC`, `CANNIBALISATION`, `CONTENT_FALLBACK:*` |
| `content_key` | Cache key for generated content. Excludes URL, so two pages with identical spec share an entry |

**`Link`** — one body link. `link_position` is always `"body"`: the crawler
extracts body links only, so nav and footer links never exist in the graph
(ADR-004). `planted` tags: `GENERIC_ANCHOR`, `OFF_TOPIC`, `NOFOLLOW`, `HEALTHY`,
`NOISE_INBOUND`, `OVER_OPTIMISED`.

### Functions

**`build_pages(n_pages, rng)`** → `(pages, by_topic)`
Creates, in order: one pillar per topic, spokes across subtopics, noise pages,
nav and footer utility pages, three orphan NEW pages with priority-5 keywords,
and two cannibalisation pages.

**`build_links(pages, by_topic, rng)`** → `(links, counts)`
Dense intra-topic linking, controlled cross-topic linking, then planted fixtures:
generic anchors (~18%), off-topic links (~8%), nofollow (~4%), links into noise
pages, and twelve identical exact-match anchors onto one over-optimised target.

> **Anchor leakage guard.** An anchor renders into the *source* page's body. If
> the source is planted `NO_HEAD_TERM`, an anchor containing that page's head term
> would silently break the fixture — so it rerolls into the NATURAL bucket. The
> over-optimised block applies the same guard. Both were real bugs, caught by
> `verify_corpus.py`.

**`build_queries(pages, by_topic, rng)`** → `list[dict]`
GSC-shaped rows. Lognormal impressions, exponential position, CTR from
`ctr_at()` with noise — the heavy tail matters because `opportunity_value =
impressions × (CTR@1 − CTR@current)` is dominated by it. Orphan NEW pages get no
queries. Bridge-gap topic pairs receive *shared* query terms, so demand overlaps
while links don't.

**`ground_truth(pages, links, queries, counts)`** → `dict`
The planted counts and `expected_verdicts` (REANCHOR, REMOVE, FIX, ADD_LINK). This
is what `generate_corpus.py --report` prints.

---

## `corpus/content.py`

**Page HTML.** Two backends behind one entry point.

| Function | Purpose |
|---|---|
| `generate_all(pages, links_by_source, backend, seed, …)` | Entry point. Fills `page.html`, `page.body_text`, `page.content_hash` in place. Returns stats |
| `template_html(page, links, rng)` | Offline backend. Real `<h1>`–`<h3>` and `<a>` tags, structurally regular prose |
| `build_prompt(page, links)` | LLM prompt, including the head-term constraint and strict anchor rules |
| `generate_llm(page, links, client, semaphore, …)` | One page via Gemini. Cached, verified, retried |
| `verify(page, links, html)` | Returns a list of violations — empty means usable |
| `strip_html(html)` | HTML → text for embedding. Removes script/style, tags, collapses whitespace |

### Why verification exists

An LLM told "never write this phrase" will sometimes write it. An LLM told to
embed the anchor `"click here"` will helpfully improve it. Either silently
destroys a planted fixture. `verify()` checks:

- head term present if and only if `include_head_term`
- every link's `href` present, with the anchor text **exactly** as given
- an `<h1>` exists, no `<h5>` / `<h6>`
- at least 300 words for topical pages

Failures retry up to three times, with the violations appended to the prompt and
temperature lowered. Pages that still fail fall back to template content and are
tagged `CONTENT_FALLBACK:<first problem>` — visible, never silent.

### Cache

`.cache/content/<key>.html`, keyed by `content_key` plus the page's anchor list.
Regenerating with the same seed costs nothing. Delete the directory to force
regeneration, or pass `--no-cache`.

### Configuration

`MODEL_ID = "gemini-2.5-flash"`. Needs `GEMINI_API_KEY` and the `llm` extra
(`google-genai`).

---

## `corpus/embeddings.py`

**Page and query vectors.** Two backends.

### Synthetic

| Function | Purpose |
|---|---|
| `build_space(seed)` | Per-topic centroid, subtopic centres, and 8 anisotropic axes per topic |
| `embed_synthetic(rng, space, topic, subtopic, eccentric)` | One page vector. `eccentric` displaces planted pillar-mismatch pages along the highest-variance axis, scaled by spread |
| `embed_query_synthetic(…)` | Query vectors — tighter than pages, since short text drifts less |
| `apply_synthetic(pages, seed)` | Fills every `page.embedding` |

Anisotropy is deliberate. Real embeddings do not form spherical clusters, and a
spherical synthetic space would flatter density-based clustering unfairly.

### Voyage

| Function | Purpose |
|---|---|
| `embed_voyage(texts, model, dim, input_type, use_cache)` | Batched, cached, L2-normalised |
| `_token_batches(client, texts)` | Batches by **token count**, never list length |
| `_cache_key(model, dim, text)` | Includes model and dimension |
| `apply_voyage(pages, queries, dim)` | Embeds page bodies and unique query strings |

**Limits** — voyage-4-large caps at **120K tokens per request**, not 1M (that is
the lite models). Batching uses 110K for headroom and at most 1,000 items.

**`input_type="document"` everywhere.** Page bodies and query strings alike are
corpus text being matched — there is no query side in this pipeline.

**Cache key includes the model.** Keying on text alone would silently serve
vectors from a different embedding space after a model change.

Needs `VOYAGE_API_KEY` and the `voyage` extra (`voyageai`, `tokenizers`).

---

## `corpus/writers.py`

**Persistence.** Both writers **wipe the `demo` tenant first** — they are for
corpus loading, never for production data.

**`write_neo4j(uri, user, pwd, pages, links)`**
- `MATCH (n) DETACH DELETE n` — clears the entire database
- `(:Page)` with `content_embedding`, lifecycle, crawl depth, content hash
- `(:Keyword)` — one per topic head term
- `-[:TARGETS_KEYWORD]->` — `CLIENT_STRATEGIC`, priority 5 for NEW pages, else 4
- `-[:LINKS_TO]->` — anchor, type, position, follow, target status

`html`, `body_text` and `planted` are deliberately *not* written to Neo4j — the
graph holds structure and vectors only.

**`write_mongo(uri, pages, queries)`**

| Collection | Contents |
|---|---|
| `pages` | html, bodyText, plus ground truth as `_topic`, `_subtopic`, `_includeHeadTerm`, `_planted` |
| `gsc_queries` | One row per query-page pair, with embedding |
| `gsc_metrics` | Per-URL rollup: impressions, clicks, average position, query count |
| `strategic_keywords` | Head term per topical page |
| `ctr_curves` | The synthetic curve, tagged `source: SYNTHETIC` |

The underscore-prefixed fields are ground truth. **Nothing in the pipeline may
read them** — only the evaluation scripts. A pipeline stage reading `_topic` would
be grading itself against the answer key.

---

## `generate_corpus.py`

**CLI.** Runs the whole chain and writes the result.

```bash
uv run python scripts/generate_corpus.py [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--pages` | 600 | Target size. Actual is ~614 after fixed pages are added |
| `--seed` | 42 | Same seed, identical corpus |
| `--content` | `template` | `template` or `llm` |
| `--embeddings` | `synthetic` | `synthetic` or `voyage` |
| `--concurrency` | 5 | Parallel Gemini requests |
| `--no-cache` | off | Ignore cached content and embeddings |
| `--report` | off | Print ground truth and exit — touches nothing |
| `--neo4j-uri` / `-user` / `-password` | local | Connection |
| `--mongo-uri` | local, `directConnection=true` | Needed for a single-node replica set |

Order: `build` → content → embeddings → Neo4j → Mongo → print ground truth plus
generation stats. Any `CONTENT_FALLBACK` pages are reported with a warning.

---

## `verify_corpus.py`

**Ground-truth assertions, no database.** Builds the corpus in memory, generates
content, runs nine checks. Exits 1 on any failure, so it works in CI.

```bash
uv run python scripts/verify_corpus.py --pages 600
uv run python scripts/verify_corpus.py --content llm      # verify real prose
```

| Check | Passes when |
|---|---|
| head-term constraint | No page contradicts its `include_head_term` flag |
| links embedded exactly | Every href present, every anchor unaltered |
| heading hierarchy | `<h1>` present, no `<h5>`/`<h6>` |
| body_text stripped | No tag characters leaked into embedded text |
| utility pages orphaned | All 14 nav/footer pages have zero inbound body links |
| bridge gaps separated | Both planted gaps have fewer cross-links than any other pair |
| body links only | Every link has `link_position == "body"` |
| no-head-term rate | Between 15% and 30% |
| no content fallbacks | No page fell back from LLM to template |

Run it after **any** generator change. It takes seconds with the template backend.

---

## `eval_clustering.py`

**Leiden vs HDBSCAN, scored against ground truth.** Read-only — reads what
`generate_corpus.py` wrote, writes nothing, can run repeatedly.

```bash
uv run python scripts/eval_clustering.py
uv run python scripts/eval_clustering.py --umap                 # HDBSCAN after UMAP to 10d
uv run python scripts/eval_clustering.py --min-cluster-size 8
uv run python scripts/eval_clustering.py --stability 4          # label churn across runs
```

| Function | Output |
|---|---|
| `load(…)` | URLs, embeddings, edges, ground truth, queries |
| `run_leiden(…)` | Community per page |
| `run_hdbscan(emb, use_umap, min_cluster_size, min_samples)` | Label per page, −1 for noise |
| `score(…)` | ARI and NMI vs `_topic`, ARI vs `_subtopic`, noise precision and recall, per-topic purity |
| `hub_bridges(…)` | Cluster pairs ranked by `0.4·centroid sim + 0.6·query Jaccard − link density` |
| `pillar_check(…)` | Declared pillar vs the page nearest each cluster centroid |

### What to look for

```
ARI(leiden, hdbscan)     0.3–0.7   complementary — keep both
                         > 0.85    redundant
noise_recall             < 0.5     HDBSCAN's main advantage failed
bridge gaps              both planted gaps should rank top 3
pillar check             both PILLAR_MISMATCH topics should disagree
stability (pairwise ARI) < 0.8     hubId needs centroid matching before use as a feature
```

On a real crawl there is no `_topic` — use the LLM eval labels instead, stored in
their own `eval_labels` collection, and report kappa next to every ARI.

---

## Known issue — `eval_clustering.py` still uses Neo4j GDS

`run_leiden()` calls `graphdatascience` — a GDS projection, `gds.leiden.stream`,
and a silent fallback to Louvain if Leiden is unavailable. That predates ADR-002,
which moved all graph algorithms to igraph and `leidenalg`.

Consequences:
- The evaluation scores **GDS Leiden**, while the pipeline will run **leidenalg**.
  Results may not transfer.
- `graphdatascience` is not in `dev/pyproject.toml`, so a fresh environment fails
  on import.
- The Louvain fallback can silently swap algorithms — the result says which it
  used, but it's easy to miss.

Fix: load edges once, build an `igraph.Graph`, call
`la.find_partition(g, la.RBConfigurationVertexPartition, seed=seed)`. Not yet
applied.

---

## Caches

```
.cache/content/      generated HTML, per content key + anchor list
.cache/embeddings/   .npy vectors, per (model, dimension, text)
```

Both safe to delete. Both make regeneration with the same seed free.

## Environment

```
GEMINI_API_KEY      content=llm
VOYAGE_API_KEY      embeddings=voyage
```

```bash
uv sync                          # core
uv sync --extra llm              # + google-genai
uv sync --extra voyage           # + voyageai, tokenizers
uv sync --extra umap             # + umap-learn, only if HDBSCAN underperforms at 2048d
```

## Adding a new planted structure

1. Declare it in `taxonomy.py`
2. Generate it in `structure.py`, tagging pages or links in `planted`
3. If it constrains content, enforce it in `content.verify()`
4. Add it to `ground_truth()` so `--report` shows the expected count
5. Add a check in `verify_corpus.run_checks()`
6. Run `make verify` and confirm the new check actually fails when you break it
