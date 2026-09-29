from __future__ import annotations

import sys
from pathlib import Path

# The two-tenant output fixture lives beside the output tests; pytest only puts a test file's
# own directory on sys.path.
_OUTPUT = str(Path(__file__).resolve().parent.parent / "output")
if _OUTPUT not in sys.path:
    sys.path.insert(0, _OUTPUT)
