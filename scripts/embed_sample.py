"""Embed three action1 pages (shortest, median, longest body) with Voyage; vectors are never stored.

uv run --env-file .env python scripts/embed_sample.py [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from itertools import combinations
from typing import Final

import numpy as np
from pymongo import AsyncMongoClient
from structlog.testing import capture_logs

from linking_engine.embedding.voyage_client import VoyageClient, VoyageSettings
from linking_engine.errors import EmbeddingError
from linking_engine.models import PageText, TenantConfig

DATABASE: Final = "linking_engine"
COLLECTION: Final = "pages"
TENANT: Final = "action1"
MIN_WORDS: Final = 50
LONGEST_CANDIDATES: Final = 5

Document = dict[str, object]


async def select_pages(client: VoyageClient) -> list[tuple[str, PageText]]:
    mongo: AsyncMongoClient[Document] = AsyncMongoClient(
        os.environ["MONGO_URI"], serverSelectionTimeoutMS=5000
    )
    try:
        pages = mongo[DATABASE][COLLECTION]
        match: Document = {
            "tenantId": TENANT,
            "statusCode": 200,
            "usable": True,
            "wordCount": {"$gte": MIN_WORDS},
        }
        cursor = await pages.aggregate(
            [
                {"$match": match},
                {"$project": {"_id": 0, "url": 1, "chars": {"$strLenCP": "$bodyText"}}},
                {"$sort": {"chars": 1, "url": 1}},
            ]
        )
        lengths = [(str(doc["url"]), int(str(doc["chars"]))) for doc in await cursor.to_list()]
        if len(lengths) < 3:
            sys.exit(f"only {len(lengths)} eligible {TENANT} pages; need 3")
        print(f"eligible pages (status 200, usable, >= {MIN_WORDS} words): {len(lengths)}")

        longest_by_chars = [url for url, _ in lengths[-LONGEST_CANDIDATES:]]
        wanted = {lengths[0][0], lengths[len(lengths) // 2][0], *longest_by_chars}
        found = await pages.find(
            {"tenantId": TENANT, "url": {"$in": sorted(wanted)}},
            {"_id": 0, "url": 1, "bodyText": 1},
        ).to_list()
        bodies = {str(doc["url"]): str(doc["bodyText"]) for doc in found}
    finally:
        await mongo.close()

    counts = client.count_tokens([bodies[url] for url in longest_by_chars])
    longest = max(zip(counts, longest_by_chars, strict=True))[1]
    chosen = [
        ("short", lengths[0][0]),
        ("median", lengths[len(lengths) // 2][0]),
        ("long", longest),
    ]
    return [(label, PageText(url=url, text=bodies[url])) for label, url in chosen]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="select and count tokens only")
    args = ap.parse_args()

    tenant = TenantConfig(tenant_id=TENANT)
    settings = VoyageSettings()
    client = VoyageClient(
        settings, model=tenant.embedding_model, dimension=tenant.embedding_dimensions
    )
    limit = settings.context_tokens

    try:
        sample = await select_pages(client)
        counts = client.count_tokens([page.text for _, page in sample])
    except EmbeddingError as error:
        sys.exit(f"embedding error: {error}")

    print(f"model {tenant.embedding_model}, {tenant.embedding_dimensions}d, text limit {limit}")
    for (label, page), count in zip(sample, counts, strict=True):
        print(f"{label:>6}  chars={len(page.text):>7}  tokens={count:>6}  {page.url}")
    if counts[-1] > limit:
        print(f"longest page exceeds the context limit: {counts[-1]} > {limit} tokens")
    else:
        print(f"NO page exceeds the context limit: longest is {counts[-1]} <= {limit} tokens")
    if args.dry_run:
        return

    try:
        with capture_logs() as events:
            embedded = await client.embed([page for _, page in sample])
    except EmbeddingError as error:
        sys.exit(f"embedding error: {error}")

    batches = [event for event in events if event["event"] == "embedding.batch"]
    if len(batches) != 1:
        sys.exit(f"expected one embedding.batch event, got {len(batches)}")
    batch = batches[0]
    print(
        f"\nbatch: items={batch['items']} latency_ms={batch['latency_ms']} retries={batch['retries']}"
    )
    for event in events:
        if event["event"] == "embedding.truncated":
            print(f"truncated: {event['url']} {event['original_tokens']} -> limit {event['limit']}")

    vectors: dict[str, np.ndarray[tuple[int], np.dtype[np.float64]]] = {}
    for (label, _), result in zip(sample, embedded, strict=True):
        vector = np.asarray(result.vector, dtype=np.float64)
        vectors[label] = vector
        print(
            f"{label:>6}  original_tokens={result.original_tokens:>6}  sent={result.tokens:>6}  "
            f"truncated={result.truncated!s:<5}  dims={vector.size}  "
            f"l2={np.linalg.norm(vector):.9f}  {result.url}"
        )

    print(f"\ninfo only: voyage-reported api_tokens={batch['api_tokens']}")
    print("\ncosine similarity:")
    for a, b in combinations(vectors, 2):
        va, vb = vectors[a], vectors[b]
        cosine = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
        print(f"  {a}-{b}: {cosine:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
