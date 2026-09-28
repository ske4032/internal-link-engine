from __future__ import annotations

import sys
from pathlib import Path

# The factories live beside the model tests, the planted ranking tenant beside the stage tests
# and the Voyage fakes it uses beside the client tests; pytest only puts a test file's own
# directory on sys.path.
for _name in ("models", "pipeline", "embedding"):
    _dir = str(Path(__file__).resolve().parent.parent / _name)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
