import { useState, useRef, useEffect, useCallback } from "react";

const COLORS = {
  bg: "#1e2640",
  surface: "#28304e",
  border: "#404d70",
  input: "#1a3a5c",
  inputBorder: "#4a9eda",
  algo: "#2a1a3a",
  algoBorder: "#9a4acd",
  store: "#1a3a2a",
  storeBorder: "#4acd7a",
  output: "#3a1a1a",
  outputBorder: "#cd4a4a",
  cache: "#3a2a1a",
  cacheBorder: "#cd9a4a",
  infra: "#1a2a3a",
  infraBorder: "#4a7acd",
  text: "#f2f6ff",
  textMuted: "#bcc8e0",
  textDim: "#7888aa",
  accent: "#60b8f8",
  green: "#60dd8a",
  purple: "#b878f0",
  orange: "#e0aa50",
  red: "#e06060",
  arrow: "#607090",
  arrowHot: "#60b8f8",
};

const NODES = [
  // ── INPUTS ──────────────────────────────────────────
  { id: "crawl",    x: 60,   y: 80,   w: 160, h: 70,  type: "input",   icon: "🕷️", label: "Crawler",        sub: "pages · links · anchors" },
  { id: "gsc",      x: 260,  y: 80,   w: 160, h: 70,  type: "input",   icon: "📊", label: "GSC API",         sub: "impressions · CTR · queries" },
  { id: "content",  x: 460,  y: 80,   w: 160, h: 70,  type: "input",   icon: "📄", label: "Page Content",    sub: "title · h1 · meta · body" },
  { id: "feedback", x: 660,  y: 80,   w: 160, h: 70,  type: "input",   icon: "👤", label: "SEO Feedback",    sub: "accept · dismiss · modify" },

  // ── INGESTION ────────────────────────────────────────
  { id: "ingest",   x: 160,  y: 220,  w: 360, h: 70,  type: "infra",   icon: "⚙️", label: "Prefect — Ingestion Service", sub: "delta crawl · GSC pull · canonicalise · dedup · noindex filter" },
  { id: "kafka",    x: 560,  y: 220,  w: 160, h: 70,  type: "infra",   icon: "⚡", label: "Kafka",           sub: "embedding jobs · re-score events" },

  // ── STORAGE ──────────────────────────────────────────
  { id: "neo4j",    x: 60,   y: 370,  w: 240, h: 100, type: "store",   icon: "🕸️", label: "Neo4j 5.x",       sub: "edges + vectors ONLY — no algos\nLINKS_TO (scored) · SUGGESTED_ACTION\ncontent_embedding [2048d] HNSW\ngnn_embedding [2048d] HNSW\nid mapping rebuilt fresh per run" },
  { id: "mongo",    x: 330,  y: 370,  w: 220, h: 100, type: "store",   icon: "🍃", label: "MongoDB",         sub: "pages · gsc_metrics\ngsc_queries · recommendations\nanchor_feedback" },
  { id: "valkey",   x: 580,  y: 370,  w: 130, h: 100, type: "cache",   icon: "⚡", label: "Valkey",          sub: "recs cache\nembed cache\nTTL: 24h" },
  { id: "minio",    x: 730,  y: 370,  w: 130, h: 100, type: "cache",   icon: "🗄️", label: "MLflow", sub: "tracking + registry\npromotion = stage\ntransition, gated" },

  // ── PIPELINE ─────────────────────────────────────────
  { id: "s1",       x: 60,   y: 550,  w: 180, h: 80,  type: "algo",    icon: "①", label: "Embedding Generation", sub: "voyage-4-large API → 2048d\n120K tok/req · chunking deferred" },
  { id: "s2",       x: 270,  y: 550,  w: 170, h: 80,  type: "algo",    icon: "②", label: "Graph Analytics",  sub: "igraph — NOT GDS\nPageRank · exact betweenness\nLeiden ×2 · ~2-4 min, ∥ Step ①" },
  { id: "s2b",      x: 460,  y: 550,  w: 170, h: 80,  type: "algo",    icon: "②b", label: "Content Clustering", sub: "HDBSCAN → hubId\n-1 = noise · hub bridges" },
  { id: "s3",       x: 650,  y: 550,  w: 170, h: 80,  type: "algo",    icon: "③", label: "GNN Encoding",     sub: "GraphSAGE — BARRIER\n→ gnn_embedding 2048d" },
  { id: "s4",       x: 60,   y: 700,  w: 180, h: 80,  type: "algo",    icon: "④", label: "Candidate Pairs",  sub: "hard constraints only\nHNSW top50 + hubId signal" },
  { id: "s5",       x: 270,  y: 700,  w: 180, h: 80,  type: "algo",    icon: "⑤", label: "Cross-Attention",  sub: "Multi-Head Attention\npair_vector ~6153d @2048d" },
  { id: "s6",       x: 480,  y: 700,  w: 180, h: 80,  type: "algo",    icon: "⑥", label: "LambdaMART Rank",  sub: "LightGBM · chunked ~50k\n6a proxy · 6b real labels" },

  // ── ANCHOR PIPELINE ──────────────────────────────────
  { id: "s7a",      x: 60,   y: 855,  w: 150, h: 75,  type: "algo",    icon: "⑦a", label: "Keyword Resolution", sub: "strategic → GSC by\nopportunity value" },
  { id: "s7b",      x: 230,  y: 855,  w: 150, h: 75,  type: "algo",    icon: "⑦b", label: "Extraction Ladder", sub: "exact→variant→2.5 jaccard\n→ semantic cosine · no LLM" },
  { id: "s7c",      x: 400,  y: 855,  w: 150, h: 75,  type: "algo",    icon: "⑦c", label: "CONTENT_GAP",      sub: "gated verdict\nrouted to writer" },
  { id: "s7d",      x: 570,  y: 855,  w: 150, h: 75,  type: "algo",    icon: "⑦d", label: "Anchor Scoring",   sub: "keyword: .7 jaccard+.3 cos\ndiversity: JACCARD not cosine" },

  // ── MATERIALISE ──────────────────────────────────────
  { id: "s8",       x: 270,  y: 1005, w: 320, h: 70,  type: "algo",    icon: "⑧", label: "Materialise Recommendations", sub: "Neo4j SUGGESTED_ACTION (5 types) · MongoDB · Valkey invalidate · Expire >30d" },

  // ── OUTPUT ───────────────────────────────────────────
  { id: "api",      x: 60,   y: 1150, w: 200, h: 80,  type: "output",  icon: "🚀", label: "REST API",         sub: "FastAPI · Traefik\nJWT · Pydantic · Valkey cache" },
  { id: "output",   x: 290,  y: 1150, w: 220, h: 80,  type: "output",  icon: "📋", label: "Recommendation",   sub: "score 0–100 · tier\nplacements · anchors\nassumptions · signals" },
  { id: "training", x: 540,  y: 1150, w: 200, h: 80,  type: "infra",   icon: "🔄", label: "Monthly Retraining", sub: "GNN contrastive fine-tune\nLambdaMART · NDCG@10 gate\nMLflow stage transition" },

  // ── INFRA ────────────────────────────────────────────
  { id: "k3s",      x: 60,   y: 1310, w: 200, h: 65,  type: "infra",   icon: "☸️", label: "K3s + ArgoCD",     sub: "GitOps · standalone server" },
  { id: "monitor",  x: 290,  y: 1310, w: 200, h: 65,  type: "infra",   icon: "📡", label: "Prometheus + Grafana", sub: "pipeline health · NDCG@10" },
  { id: "uptime",   x: 520,  y: 1310, w: 200, h: 65,  type: "infra",   icon: "💚", label: "Uptime Kuma",      sub: "service availability" },
];

const EDGES = [
  // inputs → ingest
  { from: "crawl",    to: "ingest",   label: "" },
  { from: "gsc",      to: "ingest",   label: "" },
  { from: "content",  to: "ingest",   label: "" },
  { from: "ingest",   to: "kafka",    label: "changed pages" },
  // ingest → storage
  { from: "ingest",   to: "neo4j",    label: "Page + LINKS_TO" },
  { from: "ingest",   to: "mongo",    label: "content + GSC" },
  // kafka → s1
  { from: "kafka",    to: "s1",       label: "embed queue" },
  // storage → pipeline
  { from: "neo4j",    to: "s1",       label: "" },
  { from: "neo4j",    to: "s2",       label: "" },
  { from: "mongo",    to: "s3",       label: "GSC features" },
  { from: "mongo",    to: "s4",       label: "GSC filter" },
  { from: "mongo",    to: "s6",       label: "labels" },
  // pipeline steps
  { from: "s1",       to: "s3",       label: "content_emb" },
  { from: "s1",       to: "s2b",      label: "content_emb" },
  { from: "s2b",      to: "s4",       label: "hubId signal" },
  { from: "s2",       to: "s3",       label: "PR·comm·btwn" },
  { from: "s3",       to: "s4",       label: "gnn_emb" },
  { from: "s3",       to: "s5",       label: "gnn_emb" },
  { from: "s4",       to: "s5",       label: "candidate pairs" },
  { from: "s5",       to: "s6",       label: "pair_vec ~6153d" },
  { from: "feedback", to: "s6",       label: "labels" },
  // s6 → anchor pipeline
  { from: "s6",       to: "s7a",      label: "score>45" },
  { from: "s6",       to: "s8",       label: "all scores" },
  // anchor pipeline
  { from: "mongo",    to: "s7a",      label: "body text" },
  { from: "neo4j",    to: "s7a",      label: "gnn_emb" },
  { from: "neo4j",    to: "s7b",      label: "LINKS_TO anchors" },
  { from: "s7a",      to: "s7b",      label: "resolved keyword" },
  { from: "s7b",      to: "s7d",      label: "extracted phrase" },
  { from: "s7b",      to: "s7c",      label: "no phrase found" },
  { from: "s7d",      to: "s8",       label: "top anchors" },
  // materialise → output
  { from: "s8",       to: "neo4j",    label: "SUGGESTED_ACTION" },
  { from: "s8",       to: "mongo",    label: "rec payload" },
  { from: "s8",       to: "valkey",   label: "invalidate" },
  // output
  { from: "mongo",    to: "api",      label: "" },
  { from: "valkey",   to: "api",      label: "cache hit" },
  { from: "api",      to: "output",   label: "" },
  // feedback loop
  { from: "output",   to: "feedback", label: "team review", dashed: true },
  { from: "feedback", to: "training", label: "accept/dismiss", dashed: true },
  { from: "training", to: "minio",    label: "new models", dashed: true },
  { from: "minio",    to: "s3",       label: "load model", dashed: true },
  { from: "minio",    to: "s6",       label: "load model", dashed: true },
];

const TYPE_STYLE = {
  input:  { bg: "#1e4870", border: "#60b8f8", tag: "INPUT",     tagBg: "#60b8f833", tagColor: "#90d0ff" },
  algo:   { bg: "#3a2060", border: "#b070ee", tag: "ALGORITHM", tagBg: "#b070ee33", tagColor: "#d090ff" },
  store:  { bg: "#1a4a30", border: "#50d070", tag: "STORE",     tagBg: "#50d07033", tagColor: "#80f0a0" },
  cache:  { bg: "#4a3010", border: "#e0a040", tag: "CACHE",     tagBg: "#e0a04033", tagColor: "#ffcc60" },
  output: { bg: "#4a1a1a", border: "#e06060", tag: "OUTPUT",    tagBg: "#e0606033", tagColor: "#ff9090" },
  infra:  { bg: "#1a3060", border: "#5090e0", tag: "INFRA",     tagBg: "#5090e033", tagColor: "#80b8ff" },
};

const LEGEND = [
  { type: "input",  label: "Input / External Source" },
  { type: "algo",   label: "Algorithm / Processing" },
  { type: "store",  label: "Persistent Storage" },
  { type: "cache",  label: "Cache / Object Store" },
  { type: "output", label: "Output / API" },
  { type: "infra",  label: "Infrastructure" },
];

const CANVAS_W = 900;
const CANVAS_H = 1450;

function getCenter(node) {
  return { x: node.x + node.w / 2, y: node.y + node.h / 2 };
}

function getEdgePoints(from, to) {
  const fc = getCenter(from);
  const tc = getCenter(to);
  const dx = tc.x - fc.x;
  const dy = tc.y - fc.y;
  // Pick best exit/entry side
  let sx, sy, ex, ey;
  if (Math.abs(dy) > Math.abs(dx)) {
    // vertical dominant
    sx = fc.x; sy = dy > 0 ? from.y + from.h : from.y;
    ex = tc.x; ey = dy > 0 ? to.y : to.y + to.h;
  } else {
    // horizontal dominant
    sx = dx > 0 ? from.x + from.w : from.x; sy = fc.y;
    ex = dx > 0 ? to.x : to.x + to.w;       ey = tc.y;
  }
  return { sx, sy, ex, ey };
}

export default function SystemDesign() {
  const [scale, setScale] = useState(0.72);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState(null);
  const [selected, setSelected] = useState(null);
  const svgRef = useRef(null);

  const onMouseDown = useCallback((e) => {
    if (e.target === svgRef.current || e.target.classList.contains("canvas-bg")) {
      setDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
      setPanStart({ ...pan });
    }
  }, [pan]);

  const onMouseMove = useCallback((e) => {
    if (!dragging) return;
    setPan({
      x: panStart.x + (e.clientX - dragStart.x),
      y: panStart.y + (e.clientY - dragStart.y),
    });
  }, [dragging, dragStart, panStart]);

  const onMouseUp = useCallback(() => setDragging(false), []);

  const onWheel = useCallback((e) => {
    e.preventDefault();
    setScale(s => Math.min(2, Math.max(0.3, s - e.deltaY * 0.001)));
  }, []);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const nodeById = Object.fromEntries(NODES.map(n => [n.id, n]));
  const selectedNode = selected ? nodeById[selected] : null;

  return (
    <div style={{
      width: "100vw", height: "100vh", background: COLORS.bg,
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "12px 20px", borderBottom: `1px solid ${COLORS.border}`,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: COLORS.surface, flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg, #4a9eda, #9a4acd)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>🔗</div>
          <div>
            <div style={{ color: COLORS.text, fontSize: 14, fontWeight: 700, letterSpacing: "0.05em" }}>
              SEO INTERNAL LINKING ENGINE
            </div>
            <div style={{ color: COLORS.textMuted, fontSize: 10, letterSpacing: "0.1em" }}>
              SYSTEM DESIGN — PHASE 4 PRODUCTION
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {LEGEND.map(l => (
            <div key={l.type} style={{
              display: "flex", alignItems: "center", gap: 5,
              padding: "3px 8px", borderRadius: 4,
              border: `1px solid ${TYPE_STYLE[l.type].border}22`,
              background: TYPE_STYLE[l.type].bg,
            }}>
              <div style={{
                width: 8, height: 8, borderRadius: 2,
                background: TYPE_STYLE[l.type].border,
              }} />
              <span style={{ color: TYPE_STYLE[l.type].tagColor, fontSize: 9, letterSpacing: "0.08em" }}>
                {l.label}
              </span>
            </div>
          ))}
          <div style={{
            display: "flex", gap: 4, marginLeft: 8,
            padding: "3px 8px", borderRadius: 4, border: `1px solid ${COLORS.border}`,
          }}>
            <button onClick={() => setScale(s => Math.min(2, s + 0.1))} style={{
              background: "none", border: "none", color: COLORS.textMuted,
              cursor: "pointer", fontSize: 14, padding: "0 3px",
            }}>+</button>
            <span style={{ color: COLORS.textDim, fontSize: 10, alignSelf: "center" }}>
              {Math.round(scale * 100)}%
            </span>
            <button onClick={() => setScale(s => Math.max(0.3, s - 0.1))} style={{
              background: "none", border: "none", color: COLORS.textMuted,
              cursor: "pointer", fontSize: 14, padding: "0 3px",
            }}>−</button>
            <button onClick={() => { setScale(0.72); setPan({ x: 0, y: 0 }); }} style={{
              background: "none", border: "none", color: COLORS.accent,
              cursor: "pointer", fontSize: 9, padding: "0 3px", letterSpacing: "0.05em",
            }}>FIT</button>
          </div>
        </div>
      </div>

      {/* Canvas + detail panel */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* SVG canvas */}
        <svg
          ref={svgRef}
          style={{
            flex: 1, background: COLORS.bg,
            cursor: dragging ? "grabbing" : "grab",
          }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        >
          <defs>
            <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill={COLORS.arrow} />
            </marker>
            <marker id="arrow-hot" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill={COLORS.accent} />
            </marker>
            <marker id="arrow-dash" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
              <path d="M0,0 L0,6 L8,3 z" fill={COLORS.orange} />
            </marker>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            {/* Grid pattern */}
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#2a3560" strokeWidth="0.5" opacity="0.8" />
            </pattern>
          </defs>

          <g transform={`translate(${pan.x},${pan.y}) scale(${scale})`}>
            {/* Grid */}
            <rect
              className="canvas-bg"
              x={-2000} y={-2000} width={6000} height={6000}
              fill="url(#grid)"
            />

            {/* Section labels */}
            {[
              { x: 60, y: 50, label: "INPUT SOURCES", color: "#90d0ff" },
              { x: 60, y: 195, label: "INGESTION & EVENT BUS", color: "#80b8ff" },
              { x: 60, y: 345, label: "STORAGE LAYER", color: "#80f0a0" },
              { x: 60, y: 525, label: "STAGE 2 — DISCOVERY · EVENT-DRIVEN", color: "#d090ff" },
              { x: 60, y: 830, label: "ANCHOR OPTIMISATION SUB-PIPELINE", color: "#d090ff" },
              { x: 60, y: 980, label: "MATERIALISATION", color: "#ffcc60" },
              { x: 60, y: 1125, label: "SERVING & TRAINING", color: "#ff9090" },
              { x: 60, y: 1285, label: "INFRASTRUCTURE", color: "#80b8ff" },
            ].map((s, i) => (
              <text key={i} x={s.x} y={s.y}
                fill={s.color} fontSize="9" letterSpacing="0.15em"
                fontFamily="'JetBrains Mono', monospace" opacity="0.95"
              >{s.label}</text>
            ))}

            {/* Section dividers */}
            {[170, 310, 460, 690, 845, 990, 1140, 1300].map((y, i) => (
              <line key={i} x1={50} y1={y} x2={CANVAS_W - 30} y2={y}
                stroke={COLORS.border} strokeWidth="1" strokeDasharray="4,8" opacity="0.7" />
            ))}

            {/* Edges */}
            {EDGES.map((edge, i) => {
              const fn = nodeById[edge.from];
              const tn = nodeById[edge.to];
              if (!fn || !tn) return null;
              const { sx, sy, ex, ey } = getEdgePoints(fn, tn);
              const isHot = hovered === edge.from || hovered === edge.to ||
                            selected === edge.from || selected === edge.to;
              const midX = (sx + ex) / 2;
              const midY = (sy + ey) / 2;
              const color = edge.dashed ? COLORS.orange : (isHot ? COLORS.accent : COLORS.arrow);
              const markerId = edge.dashed ? "arrow-dash" : (isHot ? "arrow-hot" : "arrow");
              return (
                <g key={i}>
                  <path
                    d={`M${sx},${sy} C${sx},${midY} ${ex},${midY} ${ex},${ey}`}
                    fill="none"
                    stroke={color}
                    strokeWidth={isHot ? 1.5 : 1}
                    strokeDasharray={edge.dashed ? "5,4" : "none"}
                    markerEnd={`url(#${markerId})`}
                    opacity={isHot ? 0.9 : 0.45}
                  />
                  {edge.label && (
                    <text x={midX} y={midY - 4} textAnchor="middle"
                      fontSize="7" fill={color} opacity={isHot ? 1 : 0.55}
                      fontFamily="'JetBrains Mono', monospace"
                    >{edge.label}</text>
                  )}
                </g>
              );
            })}

            {/* Nodes */}
            {NODES.map(node => {
              const style = TYPE_STYLE[node.type];
              const isHov = hovered === node.id;
              const isSel = selected === node.id;
              const lines = node.sub.split("\n");
              return (
                <g
                  key={node.id}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHovered(node.id)}
                  onMouseLeave={() => setHovered(null)}
                  onClick={() => setSelected(s => s === node.id ? null : node.id)}
                >
                  {/* Glow for selected */}
                  {isSel && (
                    <rect
                      x={node.x - 3} y={node.y - 3}
                      width={node.w + 6} height={node.h + 6}
                      rx="10" fill="none"
                      stroke={style.border}
                      strokeWidth="2"
                      opacity="0.5"
                      filter="url(#glow)"
                    />
                  )}
                  {/* Main box */}
                  <rect
                    x={node.x} y={node.y}
                    width={node.w} height={node.h}
                    rx="8"
                    fill={style.bg}
                    stroke={isSel ? style.border : (isHov ? style.border : style.border + "88")}
                    strokeWidth={isSel ? 2 : (isHov ? 1.5 : 1)}
                  />
                  {/* Top tag strip */}
                  <rect
                    x={node.x} y={node.y}
                    width={node.w} height={16}
                    rx="8" fill={style.tagBg}
                    style={{ borderBottomLeftRadius: 0, borderBottomRightRadius: 0 }}
                  />
                  <rect x={node.x} y={node.y + 8} width={node.w} height={8} fill={style.tagBg} />
                  {/* Tag text */}
                  <text
                    x={node.x + node.w / 2} y={node.y + 11}
                    textAnchor="middle" fontSize="7"
                    fill={style.tagColor} letterSpacing="0.12em"
                    fontFamily="'JetBrains Mono', monospace"
                  >{style.tag}</text>

                  {/* Icon + label */}
                  <text
                    x={node.x + 10} y={node.y + 32}
                    fontSize="13" fill={COLORS.text}
                  >{node.icon}</text>
                  <text
                    x={node.x + 28} y={node.y + 32}
                    fontSize="10" fill={COLORS.text}
                    fontWeight="700" letterSpacing="0.02em"
                    fontFamily="'JetBrains Mono', monospace"
                  >{node.label}</text>

                  {/* Sub lines */}
                  {lines.map((line, li) => (
                    <text
                      key={li}
                      x={node.x + 10} y={node.y + 46 + li * 11}
                      fontSize="7.5" fill={COLORS.textMuted}
                      fontFamily="'JetBrains Mono', monospace"
                    >{line}</text>
                  ))}
                </g>
              );
            })}
          </g>
        </svg>

        {/* Detail panel */}
        {selectedNode && (
          <div style={{
            width: 260, borderLeft: `1px solid ${COLORS.border}`,
            background: COLORS.surface, padding: 16,
            overflowY: "auto", flexShrink: 0,
          }}>
            <div style={{ marginBottom: 12 }}>
              <div style={{
                display: "inline-block",
                padding: "2px 8px", borderRadius: 4, marginBottom: 8,
                background: TYPE_STYLE[selectedNode.type].tagBg,
                border: `1px solid ${TYPE_STYLE[selectedNode.type].border}44`,
              }}>
                <span style={{
                  color: TYPE_STYLE[selectedNode.type].tagColor,
                  fontSize: 9, letterSpacing: "0.12em",
                }}>{TYPE_STYLE[selectedNode.type].tag}</span>
              </div>
              <div style={{ color: COLORS.text, fontSize: 13, fontWeight: 700, marginBottom: 4 }}>
                {selectedNode.icon} {selectedNode.label}
              </div>
              <div style={{ color: COLORS.textMuted, fontSize: 10, lineHeight: 1.6 }}>
                {selectedNode.sub.split("\n").map((l, i) => (
                  <div key={i} style={{ padding: "1px 0" }}>· {l}</div>
                ))}
              </div>
            </div>

            {/* Connected edges */}
            <div style={{ borderTop: `1px solid ${COLORS.border}`, paddingTop: 12 }}>
              <div style={{ color: COLORS.textDim, fontSize: 9, letterSpacing: "0.1em", marginBottom: 8 }}>
                CONNECTIONS
              </div>
              {EDGES.filter(e => e.from === selectedNode.id || e.to === selectedNode.id).map((e, i) => {
                const isFrom = e.from === selectedNode.id;
                const otherId = isFrom ? e.to : e.from;
                const other = nodeById[otherId];
                if (!other) return null;
                return (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 6,
                    padding: "4px 6px", borderRadius: 4, marginBottom: 3,
                    background: COLORS.bg, cursor: "pointer",
                    border: `1px solid ${COLORS.border}`,
                  }} onClick={() => setSelected(otherId)}>
                    <span style={{
                      color: isFrom ? COLORS.accent : COLORS.green,
                      fontSize: 10,
                    }}>{isFrom ? "→" : "←"}</span>
                    <span style={{ fontSize: 9, color: COLORS.text }}>{other.label}</span>
                    {e.label && (
                      <span style={{
                        fontSize: 8, color: COLORS.textDim,
                        marginLeft: "auto", fontStyle: "italic",
                      }}>{e.label}</span>
                    )}
                  </div>
                );
              })}
            </div>

            <button
              onClick={() => setSelected(null)}
              style={{
                marginTop: 12, width: "100%", padding: "6px",
                background: "none", border: `1px solid ${COLORS.border}`,
                borderRadius: 4, color: COLORS.textMuted,
                cursor: "pointer", fontSize: 10, letterSpacing: "0.08em",
              }}
            >DESELECT</button>
          </div>
        )}
      </div>

      {/* Footer hint */}
      <div style={{
        padding: "6px 20px", borderTop: `1px solid ${COLORS.border}`,
        background: COLORS.surface, flexShrink: 0,
        display: "flex", gap: 20,
      }}>
        {[
          ["DRAG", "pan canvas"],
          ["SCROLL", "zoom in/out"],
          ["CLICK NODE", "inspect connections"],
          ["DASHED LINES", "feedback / training loop"],
        ].map(([k, v]) => (
          <span key={k} style={{ fontSize: 9, color: COLORS.textDim, letterSpacing: "0.05em" }}>
            <span style={{ color: COLORS.accent }}>{k}</span> {v}
          </span>
        ))}
      </div>
    </div>
  );
}
