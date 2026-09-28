from __future__ import annotations

import sys
from pathlib import Path

# quality_factories lives beside the model tests; pytest only puts a test file's own directory
# on sys.path.
_FACTORIES_DIR = str(Path(__file__).resolve().parent.parent / "models")
if _FACTORIES_DIR not in sys.path:
    sys.path.insert(0, _FACTORIES_DIR)
