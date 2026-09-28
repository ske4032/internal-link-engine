"""Score every existing body link of a tenant as the score-links Prefect flow: context
relevance and anchor-target fit are written on the LINKS_TO edges and their distributions are
logged to MLflow. Run after embed-links.

uv run --env-file .env python scripts/score_links.py --tenant <tenant>
"""

from __future__ import annotations

import argparse
import asyncio

from linking_engine.pipeline.flows import score_links_flow
from linking_engine.pipeline.link_relevance import summarise_link_relevance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()
    report, mlflow_run = asyncio.run(score_links_flow(args.tenant))
    print(summarise_link_relevance(report))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
