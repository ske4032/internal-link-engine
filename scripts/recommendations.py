"""Assemble a tenant's served output as the recommendations Prefect flow: new links per source
page with their anchors or content gaps, the link audit's verdicts, page profiles, hubs,
bridges, duplicates, pairs without an anchor and target fixes, written as a new run that
replaces the previous one once complete; the run is logged to MLflow without urls. Run after
rank-pairs, anchor-selection, hub-bridges and link-audit.

uv run --env-file .env python scripts/recommendations.py --tenant <tenant>
"""

from __future__ import annotations

import argparse
import asyncio

from linking_engine.pipeline.flows import recommendations_flow
from linking_engine.pipeline.recommendations import summarise_recommendations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()
    report, mlflow_run = asyncio.run(recommendations_flow(args.tenant))
    print(summarise_recommendations(report))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
