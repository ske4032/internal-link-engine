"""Check a labelled file and import its labels into anchor_feedback, as the import-labels
Prefect flow. A file with any problem is refused whole and every problem is listed; --check
lists them, or the counts, without importing. Rows left without a label stay unlabelled, and a
pair labelled again replaces its earlier label.

uv run --env-file .env python scripts/import_labels.py --tenant <tenant> <file> [--check]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from linking_engine.pipeline.flows import import_labels_flow
from linking_engine.pipeline.labels import LabelFileError


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tenant", required=True)
    ap.add_argument("file", type=Path)
    ap.add_argument("--check", action="store_true", help="check the file without importing")
    args = ap.parse_args()
    path = args.file.expanduser().resolve()
    if not path.is_file():
        ap.error(f"no file {path}")
    try:
        report = asyncio.run(import_labels_flow(args.tenant, path, args.check))
    except LabelFileError as error:
        sys.exit(str(error))
    counts = ", ".join(
        f"{status.value.lower()} {n}" for status, n in sorted(report.by_status.items())
    )
    print(f"{report.labelled} of {report.rows} rows labelled: {counts}")
    print(f"export {report.export_id}")
    if report.import_id is None:
        print("checked only: nothing imported")
    else:
        print(f"imported as {report.import_id}")


if __name__ == "__main__":
    main()
