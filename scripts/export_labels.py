"""Export a tenant's candidate pairs for hand labelling, as the export-labels Prefect flow.

Source pages are drawn at random from those with enough anchored candidates, and each page's
candidates are sampled from its best scores to its worst. The file holds no score or rank, its
rows are shuffled, and it holds the tenant's urls and copy, so it is written outside the
repository only. Label each row accept, modify (with the anchor you would use in anchor_used)
or dismiss, add the reviewer, then import it with scripts/import_labels.py.

uv run --env-file .env python scripts/export_labels.py --tenant <tenant> --out <folder> \\
    [--pages 20] [--pairs-per-page 10] [--seed N] [--cache-dir .cache/features]
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
from pathlib import Path

from linking_engine.models import LabelSettings
from linking_engine.pipeline.features import CACHE_DIR
from linking_engine.pipeline.flows import export_labels_flow

REPOSITORY = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--out", type=Path, required=True, help="folder for the label file")
    ap.add_argument("--pages", type=int, help="source pages to sample")
    ap.add_argument("--pairs-per-page", type=int, help="candidate pairs of each page")
    ap.add_argument("--seed", type=int, help="draw the same sample again")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out.is_relative_to(REPOSITORY):
        ap.error("--out must be outside the repository: the file holds the tenant's pages")
    given = {"pages": args.pages, "pairs_per_page": args.pairs_per_page}
    settings = LabelSettings(
        **{name: value for name, value in given.items() if value is not None},
        seed=secrets.randbelow(2**32) if args.seed is None else args.seed,
    )
    export, path = asyncio.run(export_labels_flow(args.tenant, out, settings, args.cache_dir))
    model = f" version {export.model_version}" if export.model_version else ""
    print(
        f"{export.pairs} pairs from {export.pages} source pages, {export.pairs_per_page} each; "
        f"seed {export.seed}"
    )
    print(
        f"drawn from {export.eligible_pages} pages with {export.pairs_per_page} or more anchored "
        f"candidates ({export.eligible_pairs} pairs); {export.small_pages} pages had fewer"
    )
    print(f"ranked by {export.scorer.value}{model}, weights {export.weights_version}")
    print(f"export {export.export_id}")
    print(f"label file {path}")


if __name__ == "__main__":
    main()
