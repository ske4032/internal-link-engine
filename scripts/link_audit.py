"""Audit every existing body link of a tenant as the link-audit Prefect flow: scores, issue
flags and a FIX / REANCHOR / REMOVE verdict with reasons, replacing the previous run in
link_audit and written onto the LINKS_TO edges; the run is logged to MLflow without urls. Run
after graph analytics, and after score-links for A2.

uv run --env-file .env python scripts/link_audit.py --tenant <tenant>
"""

from __future__ import annotations

import argparse
import asyncio

from linking_engine.pipeline.flows import link_audit_flow
from linking_engine.pipeline.link_audit import summarise_link_audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tenant", required=True)
    args = ap.parse_args()
    report, mlflow_run = asyncio.run(link_audit_flow(args.tenant))
    print(summarise_link_audit(report))
    print(f"mlflow run {mlflow_run}")


if __name__ == "__main__":
    main()
