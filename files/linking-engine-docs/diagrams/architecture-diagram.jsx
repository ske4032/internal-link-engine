import { useState } from "react";

const FONT = "'IBM Plex Mono', 'Fira Code', 'Courier New', monospace";

const C = {
  bg:         "#1a2035",
  surface:    "#222a42",
  border:     "#38486a",
  borderHi:   "#4a6090",

  inputBg:    "#1a3658",
  inputBdr:   "#4090cc",
  inputText:  "#90ccff",

  storeBg:    "#143a20",
  storeBdr:   "#30a050",
  storeText:  "#70ee9a",

  algoBg:     "#2a1848",
  algoBdr:    "#8040c0",
  algoText:   "#cc90ff",

  cacheBg:    "#382408",
  cacheBdr:   "#b07820",
  cacheText:  "#ffcc55",

  infraBg:    "#162040",
  infraBdr:   "#3a60b0",
  infraText:  "#80aaee",

  outputBg:   "#381414",
  outputBdr:  "#c03030",
  outputText: "#ff8080",

  feedBg:     "#0e2828",
  feedBdr:    "#208080",
  feedText:   "#50e0e0",

  decisionBg:  "#2a1e08",
  decisionBdr: "#a06a10",
  decisionText:"#ffcc50",

  arrow:      "#405878",
  arrowHot:   "#60a8e0",
  text:       "#eef4ff",
  textMuted:  "#90a8c8",
  textDim:    "#506080",
  white:      "#f4f8ff",
};

// Decision comments — keyed by component id
const DECISIONS = {
  crawler: {
    title: "Crawler over scraper",
    body: "Full crawler respects robots.txt, handles JS-rendered pages, follows redirects correctly, and captures HTTP status codes. A simple scraper misses canonical tags and noindex directives — critical for SEO accuracy.",
  },
  gsc: {
    title: "GSC API over GA4",
    body: "GSC provides query-level impression and position data that GA4 does not expose. Impressions at position 4–20 are the primary signal for opportunity identification. GA4 would miss this entirely.",
  },
  kafka: {
    title: "Kafka over direct writes",
    body: "Decouples ingestion from processing. Embedding generation and graph updates lag behind crawl without blocking it. Enables replay if the embedding provider is unreachable. The pipeline is event-driven, not scheduled.",
  },
  neo4j: {
    title: "Neo4j Community (one instance per tenant)",
    body: "Enterprise Edition required for multi-database support costs $10k–$50k+/year. Community Edition with one instance per tenant via Helm achieves clean isolation at zero licence cost. As of v5.5 graph algorithms no longer run in GDS at all, so its algorithm tier is moot here — Neo4j is edges and vectors only.",
  },
  mongodb: {
    title: "MongoDB over PostgreSQL for content",
    body: "Page content, GSC rows, and recommendation payloads are document-shaped with variable schema. MongoDB's flexible document model handles content evolution without migrations. This is a standalone project — no existing PostgreSQL to reuse.",
  },
  valkey: {
    title: "Valkey over Redis",
    body: "Valkey is the open-source Redis fork (BSD licence) after Redis changed to SSPL in 2024. Fully API-compatible — zero code changes. Avoids licence risk for a commercial SaaS product.",
  },
  minio: {
    title: "MinIO over S3",
    body: "Self-hosted object store on the same server eliminates egress costs for model artefact reads. GNN checkpoints are large (hundreds of MB) and loaded on every pipeline run. S3 egress at that volume adds up quickly.",
  },
  graphsage: {
    title: "GraphSAGE over GCN",
    body: "GCN is transductive — it cannot generalise to new nodes not seen during training. New pages are added daily. GraphSAGE is inductive: it learns an aggregation function that works on any node, including pages added after training.",
  },
  lambdamart: {
    title: "LambdaMART over neural regression",
    body: "Traffic delta is not attributable to a single link — too confounded. LambdaMART ranks opportunities using real team acceptance feedback as labels. Optimises NDCG@10 directly rather than a proxy regression target.",
  },
  attention: {
    title: "Cross-attention over cosine similarity",
    body: "Cosine similarity compares pages in isolation. Cross-attention models how the source page's neighbourhood context interacts with the target's. A high-PageRank source next to a content hub is a different signal than a high-PageRank source in an isolated cluster.",
  },
  embed: {
    title: "voyage-4-large API over local Qwen3-Embedding-0.6B",
    body: "A local bake-off between voyage-4-nano and Qwen3-Embedding-0.6B on real client content showed no meaningful quality difference — content_embedding is one input among ~270 GNN features, and both the GNN and LambdaMART are trained on acceptance feedback. With quality neutral, the decision moves to operations: the API removes the TEI service, frees 2-3GB RAM, eliminates the CPU embedding bottleneck, and is free within beta client count (200M tokens). voyage-4-nano (Apache 2.0, local, CPU) is retained as the exit path — all Voyage 4 models share an embedding space, so switching needs no re-index.",
  },
  auditscore: {
    title: "Existing links as scored objects",
    body: "Through v3 the engine could only say 'add a link'. Existing LINKS_TO edges were structural input to PageRank and GraphSAGE, plus an exclusion filter in Step 4 — they were never evaluated. That is a gap against how the job actually works: on a mature site a pillar with forty inbound links all anchored 'read more' is a bigger and cheaper win than a forty-first link. Stage 1 scores every edge on anchor quality, keyword alignment, context relevance, equity efficiency, and technical health. It needs no trained model — a Cypher pass plus embedding comparisons over edges already in the graph.",
  },
  auditout: {
    title: "Five action types, not one",
    body: "ADD_LINK alone could not express audit findings. REANCHOR (anchor generic or misaligned), REMOVE (no topical justification, dilutes equity), FIX (broken, redirected, nofollowed). REANCHOR is likely the highest-volume recommendation on any established site and shares the anchor resolution machinery with ADD_LINK — one scoring function, two applications. REPOSITION was considered but dropped: the crawler extracts body links only, so nav and footer links are never captured, and there is nothing to move a link out of. Open question for client one: what share of a typical link graph is genuinely fixable. At 5% this is a useful audit feature; at 30%+ it is the main product.",
  },
  anchordist: {
    title: "Extraction before generation",
    body: "Anchor text is a ranking signal, not a copywriting problem. The keyword is the specification and the operation is extraction — find where the topic is already mentioned in the source copy and link that phrase. That is what an SEO does by hand; you don't rewrite the client's content. Ladder: exact keyword, then close variant (stem, plural, modifier), then semantically related phrase. All three are string and embedding matching — deterministic, cheap, no hallucination surface. In v5 there is no fourth generative step; failure routes to CONTENT_GAP instead. The unverified number that matters is the extraction hit rate: the fraction of pairs where the target's keyword appears in the source body.",
  },
  gds: {
    title: "Leiden on the keyword graph, not just the link graph",
    body: "Clustering pages by the links they already have is circular — a new site or a new section has no links, so no clusters, so no recommendations. Keyword relationships exist before any links do. A second Leiden pass over the bipartite TARGETS_KEYWORD projection produces keywordCommunityId, from which pillar and spoke roles fall out: pillar is the head term by aggregate volume, spokes are long-tail variants. Where the two clusterings disagree is the signal — pages in the same keyword cluster but different link clusters are exactly the missing links Stage 2 is looking for. Leiden replaced Louvain in v5.5: Louvain can produce internally disconnected communities, incoherent for a topic cluster. Leiden guarantees connectivity and is usually faster. Both run via leidenalg outside Neo4j entirely, not through GDS.",
  },
  ollama: {
    title: "CONTENT_GAP — the LLM's only remaining role",
    body: "Through v4 the LLM was a fallback anchor generator: when extraction found no usable phrase, it invented one to insert into the copy. That is a dishonest fix — it bolts a link onto content that was never about the topic. In v5 the LLM returns a verdict instead: NO_TOPICAL_MENTION (the source doesn't cover the topic, new content needed) or AWKWARD_PHRASING (topic present, no clean phrase, light rewrite needed). This routes to a writer queue, not a link-implementation queue. A priority gate is mandatory: semantic similarity put every candidate pair in the set, so the topic is always related — without gating on strategic keyword priority or opportunity value, every failed extraction becomes a content request and floods the queue. Volume is low enough that local Qwen3.5-4B suffices; the run-type routing and Batch API path from v4 are gone.",
  },
  auditflags: {
    title: "Pass B — the two cosine dimensions, and why they must be cosine",
    body: "contextRelevance and anchorTargetFit both compare a short string to a long page: a ~20-word sentence or a 2-4 word anchor against a page of ~2,000 tokens. Jaccard is a ratio, so a perfect match still maxes out near 3/2000 — every score collapses into noise near zero. Cosine treats both sides as directions in semantic space regardless of length, which is why it is correct here and wrong for anchor diversity (see 7d), where both sides are short and the question is literal word overlap, not meaning. anchorTargetFit is new in v5.5: keywordAlignment can be fooled by an anchor that lexically matches the target's keyword while the target page is actually about something else. Pass B needs embeddings, so it runs after Step 1 finishes rather than during it.",
  },
  hdbscan: {
    title: "A third clustering, answering a third question",
    body: "linkCommunityId asks what is connected. keywordCommunityId asks what we say a page is about. hubId asks what a page is actually about, clustered on content_embedding directly. HDBSCAN's unique output is the -1 noise label — a page belonging to no coherent topic — which neither Leiden pass can express, because Leiden must place every node somewhere. It also adapts to per-cluster density where Leiden's resolution parameter is global, and its condensed tree gives a real hierarchy rather than Leiden's approximate collapse levels. Two new capabilities depend on it: hub-to-hub bridge scoring (rank cluster pairs by semantic similarity plus query overlap minus link density, surfacing topics that should be cross-linked but aren't) and a pillar check (is the declared pillar page actually the one nearest its cluster's centroid?). Whether all three clusterings earn their place is a measured decision, not an assumed one — run before building discovery: if Leiden and HDBSCAN agree above ~0.85 ARI on the synthetic corpus, one is a duplicate feature.",
  },
  sentexkw: {
    title: "GSC keyword by opportunity value, not position band",
    body: "v4 selected the top GSC query where the target ranked position 5-20. That was the same defect as the removed Step 4 filter, relocated into keyword selection — a page at position 3 on a high-value commercial term still needs internal links to reach position 1, and the band excluded it. v5 ranks queries by estimated click upside: impressions x (CTR at position 1 minus CTR at current position). This favours high-demand terms wherever they rank and naturally deprioritises positions 1-2, where upside is near zero, without a hard cutoff. The CTR-by-position curve must be derived per tenant from that site's own GSC data — curves differ substantially by vertical and SERP feature mix.",
  },
  anchorscore: {
    title: "Type distribution is a preference, not a constraint",
    body: "The collection of anchors pointing at a page signals what that page is about. Too many exact-match anchors reads as manipulation; too few and there is no keyword signal at all. The 15/20/50/15 profile drives which type to prefer when the target's existing distribution is skewed. But ordering matters and was ambiguous through v4: extraction runs first, and the distribution only chooses between candidates that already exist in the copy. If the sole available phrase is a partial match, it is used even when the profile wants exact — you don't skip a good link because a ratio would prefer a different type. Naturalness was also dropped from scoring in v5: it existed to catch awkward LLM phrasing, and an extracted phrase already reads naturally in its own sentence.",
  },
  reranker: {
    title: "No reranking stage",
    body: "Step 5 cross-attention is already a cross-encoder, and one fine-tuned monthly on acceptance feedback. A general query-document reranker has never seen an SEO linking decision. The shape is also wrong — this pipeline has no queries, only page-to-page and sentence-to-page comparison. And rerank-3 billing at top-50 with full bodies is ~690M tokens per client per run. Revisit only if NDCG@10 plateaus and error analysis shows topical-relevance misses.",
  },
  hnsw: {
    title: "No GSC filter — eligibility is not priority",
    body: "v2 required targets to have impressions > 500 AND position 4-20. Two defects: avg_position is a mean over a heavily skewed query distribution, and the filter structurally excluded new pages, striking-distance pages, orphans, strategic pages, and pillar pages — making the standard hub-and-spoke topic cluster pattern unrepresentable. Eligibility now reduces to hard constraints; everything the filter decided is a LambdaMART feature. Output concentration is handled by saturation features and a serving-time diversity cap.",
  },
  spring: {
    title: "Prefect over Spring Batch, Argo Workflows, ZenML",
    body: "The stack moved to Python end to end, so Spring Batch is gone. Prefect gives the same properties — step-level retry, skip policies, persistent run history — in-process with typed Python objects passed between tasks, no serialisation boundary. Argo Workflows was considered since ArgoCD is already used for deployment, but its unit is a container, the wrong granularity for passing a 2048d array or a 6153d pair vector between steps. ZenML overlaps MLflow, and roughly 70 percent of this pipeline is not ML at all — wrapping crawl and cache invalidation in ZenML step abstractions buys lineage nobody asked for.",
  },
  k3s: {
    title: "K3s over full Kubernetes",
    body: "Full K8s requires 3+ control plane nodes and significant RAM overhead. K3s runs a complete Kubernetes API on a single server node with 512MB RAM overhead. Sufficient for this workload and aligns with existing Chirp infrastructure experience.",
  },
};

const LAYERS = [
  {
    id: "input",
    label: "INPUT LAYER",
    tag: "EXTERNAL",
    color: C.inputText,
    bg: C.inputBg,
    bdr: C.inputBdr,
    nodes: [
      { id: "crawler",  label: "Crawler",        sub: "Screaming Frog / Custom\nrespects robots.txt · JS render",   hasDecision: true },
      { id: "gsc",      label: "GSC API",         sub: "16 months · impressions\nposition · CTR · queries",          hasDecision: true },
      { id: "content",  label: "Page Content",    sub: "title · h1 · meta\nbody text · crawl metadata",             hasDecision: false },
      { id: "manual",   label: "Manual Feed",     sub: "CSV upload\nAPI endpoint",                                   hasDecision: false },
    ],
  },
  {
    id: "ingest",
    label: "INGESTION & EVENT BUS",
    tag: "INFRASTRUCTURE",
    color: C.infraText,
    bg: C.infraBg,
    bdr: C.infraBdr,
    nodes: [
      { id: "spring",  label: "Prefect",          sub: "Ingestion Service — Python\ndelta crawl · GSC pull · dedup\ncanonicalise · noindex filter\nstep retry + skip policies",     hasDecision: true, wide: true },
      { id: "kafka",   label: "Kafka",             sub: "Event Bus\nembedding jobs\nre-score events\nasync decoupling",  hasDecision: true },
    ],
  },
  {
    id: "storage",
    label: "STORAGE LAYER",
    tag: "PERSISTENCE",
    color: C.storeText,
    bg: C.storeBg,
    bdr: C.storeBdr,
    nodes: [
      { id: "neo4j",   label: "Neo4j 5.x",        sub: "Community — edges + vectors ONLY\nLINKS_TO (scored) · SUGGESTED_ACTION\nTARGETS_KEYWORD · content_embedding 2048d\ngnn_embedding 2048d ←HNSW both\nalgorithms computed OUTSIDE — see igraph", hasDecision: true },
      { id: "mongodb", label: "MongoDB",           sub: "Standalone — not shared\npages · gsc_metrics\ngsc_queries\nrecommendations\nanchor_feedback",                         hasDecision: true },
      { id: "valkey",  label: "Valkey",            sub: "Redis-fork BSD licence\nrecs cache · embed cache\nTTL 24h",                                                          hasDecision: true },
      { id: "minio",   label: "MLflow + MinIO",    sub: "Tracking + registry\npromotion gate = stage transition\nMinIO as artefact backend\n(already on your server)",         hasDecision: true },
    ],
  },
  {
    id: "audit",
    label: "STAGE 1 — LINK AUDIT  ·  NO MODEL REQUIRED",
    tag: "AUDIT",
    color: C.algoText,
    bg: C.algoBg,
    bdr: C.algoBdr,
    nodes: [
      { id: "auditscore", label: "Pass A — no embeddings",  sub: "anchor quality · kw alignment\nequity efficiency · technical health\nruns DURING Step ① embedding",           hasDecision: true },
      { id: "auditflags", label: "Pass B — needs embeddings", sub: "contextRelevance · anchorTargetFit\nboth cosine — length asymmetry\nGENERIC · MISALIGNED · WASTED_EQUITY", hasDecision: true },
      { id: "auditout",   label: "Audit Verdicts",         sub: "REANCHOR · REMOVE\nFIX · or no action",                                                        hasDecision: true },
    ],
  },
  {
    id: "pipeline",
    label: "STAGE 2 — DISCOVERY  ·  EVENT-DRIVEN",
    tag: "ALGORITHMS",
    color: C.algoText,
    bg: C.algoBg,
    bdr: C.algoBdr,
    nodes: [
      { id: "embed",     label: "① Embedding Generation",  sub: "voyage-4-large (Voyage API)\n2048d — chunking deferred\n120K tokens/req · input_type=document\nfallback: voyage-4-nano local", hasDecision: true },
      { id: "gds",       label: "② Graph Analytics",       sub: "igraph + leidenalg — NOT GDS\nPageRank · EXACT betweenness\nLeiden ×2: link + keyword graph\n~2-4 min, runs parallel with ①", hasDecision: true },
      { id: "hdbscan",   label: "②b Content Clustering",   sub: "HDBSCAN over content_embedding\n→ hubId, -1 = noise label\nhub-to-hub bridges · pillar check\nconditional on eval vs Leiden", hasDecision: true },
      { id: "graphsage", label: "③ GNN Encoding",          sub: "GraphSAGE — inductive\n3-layer MEAN aggregation\nh_v=σ(W·MEAN(N(v)∪v))\n→ gnn_embedding 2048d · BARRIER",     hasDecision: true },
      { id: "hnsw",      label: "④ Candidate Retrieval",   sub: "hard constraints only\nHNSW top 50 + hubId + both\ncommunityIds as signals, not gates\nJaccard: comparable set sizes only", hasDecision: true },
      { id: "attention", label: "⑤ Cross-Attention",       sub: "Multi-Head Attention\nQKᵀ/√2048 → softmax\ninteraction vector 2048d\npair_feature_vector ~6153d",                hasDecision: true },
      { id: "lambdamart",label: "⑥ LambdaMART Rank",       sub: "LightGBM — NDCG@10\nchunked inference ~50k/batch\n6a: proxy labels · 6b: real labels\nopportunity_score 0–100", hasDecision: true },
    ],
  },
  {
    id: "anchor",
    label: "ANCHOR RESOLUTION — DETERMINISTIC",
    tag: "NLP",
    color: C.feedText,
    bg: C.feedBg,
    bdr: C.feedBdr,
    nodes: [
      { id: "sentexkw", label: "⑦a Keyword Resolution",    sub: "strategic primary →\nGSC by opportunity value\n→ title/h1 fallback",           hasDecision: true },
      { id: "anchordist",label: "⑦b Extraction Ladder",    sub: "exact → variant → 2.5 jaccard\n→ semantic cosine · no LLM\nrung 2.5 diverts traffic from 3", hasDecision: true },
      { id: "ollama",   label: "⑦c CONTENT_GAP",           sub: "no phrase exists + gated\nverdict, not an anchor\nQwen3.5-4B local",              hasDecision: true },
      { id: "anchorscore",label: "⑦d Score + Distribute",  sub: "keyword: .7 jaccard + .3 cosine\ndiversity: JACCARD not cosine\ntype mix = preference, not a rule", hasDecision: true },
    ],
  },
  {
    id: "output",
    label: "OUTPUT & SERVING LAYER",
    tag: "OUTPUT",
    color: C.outputText,
    bg: C.outputBg,
    bdr: C.outputBdr,
    nodes: [
      { id: "materialise", label: "⑧ Materialise",         sub: "Neo4j SUGGESTED_ACTION edge\n6 action types · MongoDB\nValkey invalidate\nExpire > 30 days",                  hasDecision: false },
      { id: "api",         label: "REST API",              sub: "FastAPI · Traefik\nJWT auth · Pydantic contracts\nValkey cache hit\n/v1/recommendations",                     hasDecision: false },
      { id: "dashboard",   label: "Dashboard",             sub: "ranked opportunities\nplacement sentences\nanchor candidates\nassumptions + signals",                         hasDecision: false },
    ],
  },
  {
    id: "feedback",
    label: "FEEDBACK & TRAINING LOOP  —  MONTHLY",
    tag: "LEARNING",
    color: C.cacheText,
    bg: C.cacheBg,
    bdr: C.cacheBdr,
    nodes: [
      { id: "fbcollect",  label: "Feedback Collection",    sub: "accept · dismiss · modify\nanchor_used captured\nMongoDB anchor_feedback",                                    hasDecision: false },
      { id: "gnn_train",  label: "GNN Fine-tune",          sub: "Contrastive loss\naccepted pairs → closer\ndismissed pairs → apart",                                         hasDecision: false },
      { id: "lm_retrain", label: "LambdaMART Retrain",     sub: "real acceptance labels\nMLflow registry stage transition\npromotion = beats production NDCG@10",           hasDecision: false },
      { id: "anchor_update",label: "Anchor Weight Update", sub: "acceptance by type + strategy\nupdate scorer weights\nupdate prompt templates",                                      hasDecision: false },
    ],
  },
  {
    id: "infra",
    label: "INFRASTRUCTURE",
    tag: "DEPLOYMENT",
    color: C.infraText,
    bg: C.infraBg,
    bdr: C.infraBdr,
    nodes: [
      { id: "k3s",       label: "K3s + ArgoCD",            sub: "GitOps · standalone server\nHelmfile per tenant\none Neo4j pod per client",                                  hasDecision: true },
      { id: "monitor",   label: "Prometheus + Grafana",    sub: "pipeline health\nNDCG@10 tracking\nAPI latency · error rates",                                               hasDecision: false },
      { id: "uptime",    label: "Uptime Kuma",             sub: "service availability\nexternal monitoring",                                                                  hasDecision: false },
    ],
  },
];

export default function ArchDiagram() {
  const [activeDecision, setActiveDecision] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  const activeData = activeDecision ? DECISIONS[activeDecision] : null;

  return (
    <div style={{
      minHeight: "100vh",
      background: C.bg,
      fontFamily: FONT,
      color: C.text,
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 28px",
        borderBottom: `1px solid ${C.border}`,
        background: C.surface,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8,
            background: `linear-gradient(135deg, ${C.inputBdr}, ${C.algoBdr})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, boxShadow: `0 0 16px ${C.algoBdr}44`,
          }}>🔗</div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.08em", color: C.white }}>
              SEO INTERNAL LINKING INTELLIGENCE ENGINE
            </div>
            <div style={{ fontSize: 9, color: C.textMuted, letterSpacing: "0.15em", marginTop: 2 }}>
              SYSTEM ARCHITECTURE — PHASE 4 PRODUCTION · STANDALONE SERVER · NEO4J COMMUNITY + MONGODB
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <div style={{
            padding: "4px 10px", borderRadius: 4, fontSize: 9,
            background: `${C.decisionBg}`, border: `1px solid ${C.decisionBdr}`,
            color: C.decisionText, letterSpacing: "0.1em",
          }}>
            💬 CLICK HIGHLIGHTED NODES FOR DECISIONS
          </div>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1 }}>
        {/* Main diagram */}
        <div style={{
          flex: 1, padding: "24px 28px", overflowY: "auto",
          display: "flex", flexDirection: "column", gap: 0,
        }}>
          {LAYERS.map((layer, li) => (
            <div key={layer.id}>
              {/* Layer */}
              <div style={{
                border: `1px solid ${layer.bdr}44`,
                borderRadius: 10,
                background: layer.bg,
                padding: "16px 18px 18px",
                position: "relative",
                marginBottom: 0,
              }}>
                {/* Layer header */}
                <div style={{
                  display: "flex", alignItems: "center", gap: 10, marginBottom: 14,
                }}>
                  <div style={{
                    padding: "2px 8px", borderRadius: 3,
                    background: `${layer.bdr}22`, border: `1px solid ${layer.bdr}55`,
                    fontSize: 8, color: layer.color, letterSpacing: "0.15em",
                  }}>{layer.tag}</div>
                  <div style={{
                    fontSize: 11, fontWeight: 700, color: layer.color,
                    letterSpacing: "0.1em",
                  }}>{layer.label}</div>
                </div>

                {/* Nodes */}
                <div style={{
                  display: "flex", gap: 12, flexWrap: "wrap",
                }}>
                  {layer.nodes.map(node => {
                    const hasD = node.hasDecision && DECISIONS[node.id];
                    const isHov = hoveredNode === node.id;
                    return (
                      <div
                        key={node.id}
                        style={{
                          flex: node.wide ? "2 1 300px" : "1 1 180px",
                          minWidth: node.wide ? 300 : 180,
                          maxWidth: node.wide ? 420 : 280,
                          border: `1px solid ${hasD ? layer.bdr : layer.bdr + "55"}`,
                          borderRadius: 7,
                          background: isHov ? `${layer.bdr}18` : `${C.bg}cc`,
                          padding: "10px 12px",
                          cursor: hasD ? "pointer" : "default",
                          position: "relative",
                          transition: "all 0.15s ease",
                          boxShadow: hasD && isHov ? `0 0 12px ${layer.bdr}44` : "none",
                        }}
                        onMouseEnter={() => setHoveredNode(node.id)}
                        onMouseLeave={() => setHoveredNode(null)}
                        onClick={() => hasD && setActiveDecision(
                          activeDecision === node.id ? null : node.id
                        )}
                      >
                        {/* Decision badge */}
                        {hasD && (
                          <div style={{
                            position: "absolute", top: 6, right: 8,
                            width: 16, height: 16, borderRadius: "50%",
                            background: activeDecision === node.id
                              ? C.decisionText
                              : `${C.decisionBdr}88`,
                            border: `1px solid ${C.decisionBdr}`,
                            display: "flex", alignItems: "center", justifyContent: "center",
                            fontSize: 9, color: activeDecision === node.id ? C.bg : C.decisionText,
                            fontWeight: 700,
                          }}>?</div>
                        )}

                        <div style={{
                          fontSize: 10, fontWeight: 700, color: layer.color,
                          letterSpacing: "0.04em", marginBottom: 6,
                          paddingRight: hasD ? 20 : 0,
                        }}>{node.label}</div>

                        <div style={{ fontSize: 8, color: C.textMuted, lineHeight: 1.7 }}>
                          {node.sub.split("\n").map((line, i) => (
                            <div key={i} style={{
                              display: "flex", alignItems: "flex-start", gap: 4,
                            }}>
                              <span style={{ color: C.textDim, marginTop: 1 }}>·</span>
                              <span>{line}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Arrow between layers */}
              {li < LAYERS.length - 1 && (
                <div style={{
                  display: "flex", flexDirection: "column",
                  alignItems: "center", padding: "6px 0",
                }}>
                  <div style={{
                    width: 1, height: 16,
                    background: `linear-gradient(to bottom, ${C.borderHi}, ${C.borderHi}44)`,
                  }} />
                  <div style={{
                    width: 0, height: 0,
                    borderLeft: "5px solid transparent",
                    borderRight: "5px solid transparent",
                    borderTop: `6px solid ${C.borderHi}`,
                  }} />
                </div>
              )}
            </div>
          ))}

          {/* Legend */}
          <div style={{
            marginTop: 28, padding: "14px 18px",
            border: `1px solid ${C.border}`,
            borderRadius: 8, background: C.surface,
          }}>
            <div style={{
              fontSize: 8, color: C.textMuted, letterSpacing: "0.15em", marginBottom: 10,
            }}>LEGEND</div>
            <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
              {[
                { label: "Input / External",    color: C.inputText,  bdr: C.inputBdr },
                { label: "Infrastructure",      color: C.infraText,  bdr: C.infraBdr },
                { label: "Storage",             color: C.storeText,  bdr: C.storeBdr },
                { label: "Algorithm / ML",      color: C.algoText,   bdr: C.algoBdr },
                { label: "NLP / Anchor",        color: C.feedText,   bdr: C.feedBdr },
                { label: "Output / Serving",    color: C.outputText, bdr: C.outputBdr },
                { label: "Feedback / Training", color: C.cacheText,  bdr: C.cacheBdr },
              ].map(l => (
                <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{
                    width: 10, height: 10, borderRadius: 2,
                    background: `${l.bdr}44`, border: `1px solid ${l.bdr}`,
                  }} />
                  <span style={{ fontSize: 8, color: l.color }}>{l.label}</span>
                </div>
              ))}
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{
                  width: 16, height: 16, borderRadius: "50%",
                  background: `${C.decisionBdr}88`,
                  border: `1px solid ${C.decisionBdr}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 9, color: C.decisionText, fontWeight: 700,
                }}>?</div>
                <span style={{ fontSize: 8, color: C.decisionText }}>Key decision — click to expand</span>
              </div>
            </div>
          </div>
        </div>

        {/* Decision panel */}
        <div style={{
          width: activeData ? 300 : 0,
          minWidth: activeData ? 300 : 0,
          borderLeft: activeData ? `1px solid ${C.border}` : "none",
          background: C.surface,
          transition: "all 0.2s ease",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}>
          {activeData && (
            <div style={{ padding: 20, overflowY: "auto" }}>
              <div style={{
                fontSize: 8, color: C.textMuted, letterSpacing: "0.15em", marginBottom: 10,
              }}>ARCHITECTURE DECISION</div>

              <div style={{
                padding: "10px 12px", borderRadius: 6,
                background: C.decisionBg,
                border: `1px solid ${C.decisionBdr}`,
                marginBottom: 16,
              }}>
                <div style={{
                  fontSize: 10, fontWeight: 700, color: C.decisionText,
                  letterSpacing: "0.04em", marginBottom: 8,
                }}>💬 {activeData.title}</div>
                <div style={{
                  fontSize: 9, color: C.textMuted, lineHeight: 1.8,
                }}>{activeData.body}</div>
              </div>

              {/* All decisions list */}
              <div style={{
                fontSize: 8, color: C.textMuted, letterSpacing: "0.12em", marginBottom: 10,
              }}>ALL DECISIONS ({Object.keys(DECISIONS).length})</div>

              {Object.entries(DECISIONS).map(([id, d]) => (
                <div
                  key={id}
                  style={{
                    padding: "7px 10px", borderRadius: 5, marginBottom: 4,
                    border: `1px solid ${activeDecision === id ? C.decisionBdr : C.border}`,
                    background: activeDecision === id ? C.decisionBg : "transparent",
                    cursor: "pointer",
                    transition: "all 0.1s",
                  }}
                  onClick={() => setActiveDecision(activeDecision === id ? null : id)}
                >
                  <div style={{
                    fontSize: 8, fontWeight: 700,
                    color: activeDecision === id ? C.decisionText : C.textMuted,
                    letterSpacing: "0.03em",
                  }}>{d.title}</div>
                </div>
              ))}

              <button
                onClick={() => setActiveDecision(null)}
                style={{
                  marginTop: 12, width: "100%", padding: "7px",
                  background: "none", border: `1px solid ${C.border}`,
                  borderRadius: 4, color: C.textMuted,
                  cursor: "pointer", fontSize: 8, letterSpacing: "0.1em",
                  fontFamily: FONT,
                }}
              >CLOSE PANEL</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
