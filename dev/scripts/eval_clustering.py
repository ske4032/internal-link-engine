"""
Leiden vs HDBSCAN, scored against the planted ground truth.

Runs both on the same corpus and reports where each one wins. The corpus is
built so they cannot tie: it plants hierarchy, per-cluster density variation,
and genuine noise pages that are nonetheless linked.

    uv run python scripts/eval_clustering.py
    uv run python scripts/eval_clustering.py --umap        # HDBSCAN after UMAP
    uv run python scripts/eval_clustering.py --stability 3 # rerun N times

Reads:  Neo4j (link graph, embeddings), Mongo (_topic / _subtopic / _planted)
Writes: nothing. Read-only by design, so it can be run repeatedly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from neo4j import GraphDatabase
from pymongo import MongoClient
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ── loading ───────────────────────────────────────────────────────────────

def load(neo_uri, neo_user, neo_pwd, mongo_uri):
    drv = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pwd))
    with drv.session() as s:
        rows = s.run("""
            MATCH (p:Page)
            RETURN p.url AS url, p.content_embedding AS emb,
                   p.pageType AS page_type
            ORDER BY p.url
        """).data()
        edges = s.run("""
            MATCH (a:Page)-[:LINKS_TO]->(b:Page)
            RETURN a.url AS src, b.url AS tgt
        """).data()
    drv.close()

    db = MongoClient(mongo_uri).get_database("linking_engine")
    meta = {d["url"]: d for d in db.pages.find(
        {"tenantId": "demo"}, {"url": 1, "_topic": 1, "_subtopic": 1, "_planted": 1})}
    queries = list(db.gsc_queries.find({"tenantId": "demo"}))

    urls = [r["url"] for r in rows]
    emb = np.array([r["emb"] for r in rows], dtype=np.float32)
    truth_topic = [meta[u].get("_topic") for u in urls]
    truth_sub = [f"{meta[u].get('_topic')}/{meta[u].get('_subtopic')}"
                 if meta[u].get("_topic") else None for u in urls]
    planted = {u: set(meta[u].get("_planted", [])) for u in urls}
    return urls, emb, edges, truth_topic, truth_sub, planted, queries


# ── clustering ────────────────────────────────────────────────────────────

def run_leiden(neo_uri, neo_user, neo_pwd, urls, seed=42):
    from graphdatascience import GraphDataScience
    gds = GraphDataScience(neo_uri, auth=(neo_user, neo_pwd))
    if gds.graph.exists("eval-graph")["exists"]:
        gds.graph.drop(gds.graph.get("eval-graph"))
    g, _ = gds.graph.project("eval-graph", "Page",
                             {"LINKS_TO": {"orientation": "UNDIRECTED"}})
    try:
        res = gds.leiden.stream(g, concurrency=4, randomSeed=seed)
        algo = "leiden"
    except Exception:
        res = gds.louvain.stream(g, concurrency=4)
        algo = "louvain (leiden unavailable)"

    node_ids = {int(r["nodeId"]): int(r["communityId"]) for _, r in res.iterrows()}
    lookup = gds.run_cypher(
        "MATCH (p:Page) RETURN id(p) AS nid, p.url AS url")
    url_to_comm = {row["url"]: node_ids.get(row["nid"], -1)
                   for _, row in lookup.iterrows()}
    gds.graph.drop(g)
    gds.close()
    return algo, [url_to_comm.get(u, -1) for u in urls]


def run_hdbscan(emb: np.ndarray, use_umap: bool, min_cluster_size: int,
                min_samples: int | None, seed: int = 42):
    import hdbscan
    X = emb
    if use_umap:
        import umap
        X = umap.UMAP(n_neighbors=15, n_components=10, min_dist=0.0,
                      metric="cosine", random_state=seed).fit_transform(emb)
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples,
                         metric="euclidean" if use_umap else "euclidean",
                         cluster_selection_method="eom",
                         prediction_data=True)
    labels = cl.fit_predict(X.astype(np.float64))
    return labels, cl


# ── scoring ───────────────────────────────────────────────────────────────

def score(name, labels, truth_topic, truth_sub, planted, urls):
    # ARI/NMI need a label for every point; treat true-noise as its own class
    t_topic = [t if t else "__noise__" for t in truth_topic]
    t_sub = [t if t else "__noise__" for t in truth_sub]

    noise_urls = {u for u in urls if "TRUE_NOISE" in planted[u]}
    lab = dict(zip(urls, labels))

    n_clusters = len({l for l in labels if l != -1})
    n_unclustered = sum(1 for l in labels if l == -1)

    # Did the algorithm isolate the planted noise?
    noise_as_noise = sum(1 for u in noise_urls if lab[u] == -1)
    noise_recall = noise_as_noise / max(1, len(noise_urls))
    # ...without discarding real pages
    real_urls = [u for u in urls if u not in noise_urls]
    real_discarded = sum(1 for u in real_urls if lab[u] == -1)
    noise_precision = noise_as_noise / max(1, noise_as_noise + real_discarded)

    # Per-topic purity, to expose density sensitivity
    by_topic = defaultdict(Counter)
    for u, t in zip(urls, truth_topic):
        if t:
            by_topic[t][lab[u]] += 1
    purity = {t: round(c.most_common(1)[0][1] / sum(c.values()), 3)
              for t, c in by_topic.items()}

    return {
        "algorithm": name,
        "clusters_found": n_clusters,
        "unclustered": n_unclustered,
        "ari_topic": round(adjusted_rand_score(t_topic, labels), 3),
        "nmi_topic": round(normalized_mutual_info_score(t_topic, labels), 3),
        "ari_subtopic": round(adjusted_rand_score(t_sub, labels), 3),
        "noise_recall": round(noise_recall, 3),
        "noise_precision": round(noise_precision, 3),
        "purity_by_topic": purity,
    }


def hub_bridges(urls, emb, labels, edges, queries, top_n=5):
    """Rank cluster pairs by (semantic + query overlap) minus link density."""
    lab = dict(zip(urls, labels))
    idx = {u: i for i, u in enumerate(urls)}
    clusters = defaultdict(list)
    for u, l in lab.items():
        if l != -1:
            clusters[l].append(u)
    if len(clusters) < 2:
        return []

    centroid = {c: emb[[idx[u] for u in us]].mean(axis=0) for c, us in clusters.items()}
    for c in centroid:
        centroid[c] /= np.linalg.norm(centroid[c])

    q_by_cluster = defaultdict(set)
    for q in queries:
        l = lab.get(q["url"])
        if l is not None and l != -1:
            q_by_cluster[l].add(q["query"])

    link_ct = Counter()
    for e in edges:
        a, b = lab.get(e["src"]), lab.get(e["tgt"])
        if a is not None and b is not None and a != b and a != -1 and b != -1:
            link_ct[tuple(sorted((a, b)))] += 1

    out = []
    cs = sorted(clusters)
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            sim = float(centroid[a] @ centroid[b])
            qa, qb = q_by_cluster[a], q_by_cluster[b]
            jac = len(qa & qb) / max(1, len(qa | qb))
            density = link_ct[(a, b)] / max(1, len(clusters[a]) * len(clusters[b]))
            out.append({
                "pair": [int(a), int(b)],
                "sizes": [len(clusters[a]), len(clusters[b])],
                "hub_similarity": round(sim, 3),
                "query_jaccard": round(jac, 3),
                "link_density": round(density, 5),
                "bridge_gap": round(0.4 * sim + 0.6 * jac - density * 50, 3),
            })
    return sorted(out, key=lambda r: -r["bridge_gap"])[:top_n]


def pillar_check(urls, emb, labels, planted):
    """Is the declared pillar the page nearest its cluster centroid?"""
    lab = dict(zip(urls, labels))
    idx = {u: i for i, u in enumerate(urls)}
    clusters = defaultdict(list)
    for u, l in lab.items():
        if l != -1:
            clusters[l].append(u)

    out = []
    for c, us in clusters.items():
        cen = emb[[idx[u] for u in us]].mean(axis=0)
        cen /= np.linalg.norm(cen)
        sims = sorted(((float(emb[idx[u]] @ cen), u) for u in us), reverse=True)
        nearest = sims[0][1]
        declared = [u for u in us if "DECLARED_PILLAR" in planted[u]]
        if declared:
            out.append({
                "cluster": int(c),
                "declared_pillar": declared[0],
                "centroid_nearest": nearest,
                "agree": declared[0] == nearest,
                "mismatch_planted": "PILLAR_MISMATCH" in planted[declared[0]],
            })
    return out


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="localdevpassword")
    ap.add_argument("--mongo-uri",
                    default="mongodb://localhost:27017/?directConnection=true")
    ap.add_argument("--umap", action="store_true",
                    help="reduce to 10d with UMAP before HDBSCAN")
    ap.add_argument("--min-cluster-size", type=int, default=15)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--stability", type=int, default=0,
                    help="rerun N times and report label stability")
    a = ap.parse_args()

    urls, emb, edges, t_topic, t_sub, planted, queries = load(
        a.neo4j_uri, a.neo4j_user, a.neo4j_password, a.mongo_uri)
    print(f"loaded {len(urls)} pages, {len(edges)} edges, {len(queries)} queries\n")

    results = []

    leiden_name, leiden_labels = run_leiden(
        a.neo4j_uri, a.neo4j_user, a.neo4j_password, urls)
    results.append(score(leiden_name, leiden_labels, t_topic, t_sub, planted, urls))

    hdb_labels, model = run_hdbscan(emb, a.umap, a.min_cluster_size, a.min_samples)
    hdb_name = f"hdbscan{'+umap' if a.umap else ''} (mcs={a.min_cluster_size})"
    results.append(score(hdb_name, hdb_labels, t_topic, t_sub, planted, urls))

    print(json.dumps(results, indent=2))

    print("\n── hierarchy (hdbscan condensed tree) ──")
    try:
        tree = model.condensed_tree_.to_pandas()
        levels = tree[tree.child_size > 1]
        print(f"internal nodes: {len(levels)}  "
              f"selected clusters: {len({l for l in hdb_labels if l != -1})}")
    except Exception as e:
        print(f"unavailable: {e}")

    print("\n── hub bridges (hdbscan clusters) ──")
    print(json.dumps(hub_bridges(urls, emb, hdb_labels, edges, queries), indent=2))

    print("\n── pillar check (hdbscan clusters) ──")
    print(json.dumps(pillar_check(urls, emb, hdb_labels, planted), indent=2))

    if a.stability:
        print(f"\n── stability over {a.stability} runs ──")
        runs = [run_hdbscan(emb, a.umap, a.min_cluster_size, a.min_samples,
                            seed=100 + i)[0] for i in range(a.stability)]
        pairs = [(adjusted_rand_score(runs[i], runs[j]), i, j)
                 for i in range(len(runs)) for j in range(i + 1, len(runs))]
        print(json.dumps({"pairwise_ari": [round(p[0], 3) for p in pairs],
                          "mean": round(float(np.mean([p[0] for p in pairs])), 3)},
                         indent=2))


if __name__ == "__main__":
    main()
