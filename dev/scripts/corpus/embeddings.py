"""
Embeddings for the corpus.

Two backends, and the choice decides what the corpus can prove.

  synthetic  deterministic per-topic base vector plus anisotropic noise. Pages
             in a topic land close, topics land apart. Enough to build retrieval
             and clustering against; proves the plumbing works and nothing else.

  voyage     real voyage-4-large over real page text. Required before trusting
             any retrieval quality number — which is why the LLM content backend
             exists in the first place.

Anisotropy matters. Real embeddings do not form spherical clusters, and a
spherical synthetic space would flatter density-based clustering unfairly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from corpus.structure import Page
from corpus.taxonomy import DIM, TOPICS

CACHE_DIR = Path(".cache/embeddings")

# voyage-4-large limits. NOT 1M — that is the lite models.
MAX_TOKENS_PER_REQUEST = 110_000   # headroom under the 120K ceiling
MAX_ITEMS_PER_REQUEST = 1_000


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ── synthetic ───────────────────────────────────────────────────────────────

def build_space(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    space: dict = {"topics": {}, "noise_basis": _unit(rng.normal(size=DIM))}
    for name, t in TOPICS.items():
        centroid = _unit(rng.normal(size=DIM))
        axes = np.stack([_unit(rng.normal(size=DIM)) for _ in range(8)])
        scales = rng.uniform(0.4, 1.6, size=8)
        subs = {
            sub: _unit(centroid + _unit(rng.normal(size=DIM)) * 0.35)
            for sub in t["subtopics"]
        }
        space["topics"][name] = {
            "centroid": centroid, "subs": subs,
            "axes": axes, "scales": scales, "spread": t["spread"],
        }
    return space


def embed_synthetic(
    rng: np.random.Generator, space: dict,
    topic: str | None, subtopic: str | None, eccentric: float = 0.0,
) -> np.ndarray:
    """`eccentric` > 0 displaces a page along the cluster's highest-variance
    axis, so it stays inside the cluster but sits far from the centroid. Scaled
    by spread, because a diffuse cluster needs a larger displacement before its
    member becomes an outlier."""
    if topic is None:
        return _unit(space["noise_basis"] * 0.3 + rng.normal(scale=1.0, size=DIM))

    t = space["topics"][topic]
    base = t["subs"][subtopic] if subtopic else t["centroid"]
    coeffs = rng.normal(scale=t["spread"], size=len(t["axes"])) * t["scales"]
    noise = (coeffs[:, None] * t["axes"]).sum(axis=0)
    noise += rng.normal(scale=t["spread"] * 0.25, size=DIM)
    v = base + noise
    if eccentric:
        v = v + t["axes"][0] * t["scales"][0] * eccentric * (1 + 4 * t["spread"])
    return _unit(v)


def embed_query_synthetic(
    rng: np.random.Generator, space: dict, topic: str, subtopic: str | None
) -> np.ndarray:
    """Queries sit tighter than pages: short text, less drift."""
    t = space["topics"][topic]
    base = t["subs"][subtopic] if subtopic else t["centroid"]
    return _unit(base + rng.normal(scale=0.16, size=DIM))


def apply_synthetic(pages: list[Page], seed: int) -> None:
    from corpus.taxonomy import PILLAR_MISMATCH

    space = build_space(seed)
    rng = np.random.default_rng(seed + 1)
    for p in pages:
        ecc = 1.9 if (p.topic in PILLAR_MISMATCH
                      and "DECLARED_PILLAR" in p.planted) else 0.0
        p.embedding = embed_synthetic(rng, space, p.topic, p.subtopic, ecc).tolist()


# ── voyage ──────────────────────────────────────────────────────────────────

def _token_batches(client, texts: list[str]) -> list[list[int]]:
    """Batch by TOKEN COUNT, never by list length. A fixed batch size fails on
    long pages and wastes headroom on short ones."""
    counts = client.count_tokens(texts)
    batches, cur, total = [], [], 0
    for i, n in enumerate(counts):
        if cur and (total + n > MAX_TOKENS_PER_REQUEST
                    or len(cur) >= MAX_ITEMS_PER_REQUEST):
            batches.append(cur)
            cur, total = [], 0
        cur.append(i)
        total += n
    if cur:
        batches.append(cur)
    return batches


def _cache_key(model: str, dim: int, text: str) -> str:
    """Keyed by model AND dimension. Keying on text alone would silently serve
    vectors from a different space after a model change."""
    h = hashlib.sha256(text.encode()).hexdigest()[:24]
    return f"{model}-{dim}-{h}"


def embed_voyage(
    texts: list[str], model: str = "voyage-4-large", dim: int = DIM,
    input_type: str = "document", use_cache: bool = True,
) -> list[list[float]]:
    import voyageai

    client = voyageai.Client()
    out: list[list[float] | None] = [None] * len(texts)
    pending: list[int] = []

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(texts):
            f = CACHE_DIR / f"{_cache_key(model, dim, t)}.npy"
            if f.exists():
                out[i] = np.load(f).tolist()
            else:
                pending.append(i)
    else:
        pending = list(range(len(texts)))

    if pending:
        pending_texts = [texts[i] for i in pending]
        for batch in _token_batches(client, pending_texts):
            resp = client.embed(
                [pending_texts[j] for j in batch],
                model=model, input_type=input_type, output_dimension=dim,
            )
            for j, vec in zip(batch, resp.embeddings, strict=True):
                v = np.asarray(vec, dtype=np.float32)
                v = v / np.linalg.norm(v)   # Neo4j cosine index assumes unit
                idx = pending[j]
                out[idx] = v.tolist()
                if use_cache:
                    np.save(CACHE_DIR / f"{_cache_key(model, dim, texts[idx])}.npy", v)

    assert all(o is not None for o in out), "embedding gap"
    return out  # type: ignore[return-value]


def apply_voyage(pages: list[Page], queries: list[dict], dim: int = DIM) -> dict:
    """Everything here is a document — page bodies and query strings alike are
    corpus text being matched, not search queries. There is no query side in
    this pipeline."""
    page_texts = [p.body_text for p in pages]
    vecs = embed_voyage(page_texts, dim=dim, input_type="document")
    for p, v in zip(pages, vecs, strict=True):
        p.embedding = v

    unique_q = sorted({q["query"] for q in queries})
    qvecs = dict(zip(unique_q, embed_voyage(unique_q, dim=dim,
                                            input_type="document"), strict=True))
    for q in queries:
        q["embedding"] = qvecs[q["query"]]

    return {"pages_embedded": len(pages), "unique_queries": len(unique_q)}
