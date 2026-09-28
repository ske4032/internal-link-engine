"""Rank one tenant's candidate pairs as the rank-pairs Prefect flow: the production ranker's
scores, else the baseline scorer's with the reason, written as Parquet under the cache
directory; nothing is written to the stores.

uv run --env-file .env python scripts/rank_pairs.py --tenant <tenant> [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import rank_pairs_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    report, path = asyncio.run(rank_pairs_flow(args.tenant, args.cache_dir))
    version = f" version {report.model_version}" if report.model_version else ""
    print(f"{report.pairs} pairs ranked by {report.scorer.value}{version} in {report.seconds:.1f}s")
    if report.fallback_reason:
        print(f"baseline because: {report.fallback_reason}")
    print(f"ranked pairs {path}")


if __name__ == "__main__":
    main()
