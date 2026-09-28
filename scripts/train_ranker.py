"""Train one tenant's learned ranker as the train-ranker Prefect flow: held-out rounds of its
own body links as proxy labels, LambdaMART against the baseline and the production model, the
run and the model logged to MLflow. The production alias moves only when RANKER_PROMOTION is
"allowed"; nothing is written to the stores. Without a Voyage key the semantic rung is skipped.

uv run --env-file .env python scripts/train_ranker.py --tenant <tenant> [--rounds 10] [--share 0.10] [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from linking_engine.ml.ranking import summarise_ranker
from linking_engine.models import HeldOutSettings
from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import train_ranker_flow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--rounds", type=int, default=HeldOutSettings().rounds)
    ap.add_argument("--share", type=float, default=HeldOutSettings().share)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    report = asyncio.run(train_ranker_flow(args.tenant, args.rounds, args.share, args.cache_dir))
    print(summarise_ranker(report))


if __name__ == "__main__":
    main()
