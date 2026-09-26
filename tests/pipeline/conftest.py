from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# voyage_fakes lives beside the client tests; pytest only puts a test file's own directory on sys.path.
_FAKES_DIR = str(Path(__file__).resolve().parent.parent / "embedding")
if _FAKES_DIR not in sys.path:
    sys.path.insert(0, _FAKES_DIR)


@pytest.fixture(autouse=True)
def isolated_voyage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A VOYAGE_* variable in the developer's shell must not feed VoyageSettings."""
    for var in list(os.environ):
        if var.upper().startswith("VOYAGE_"):
            monkeypatch.delenv(var, raising=False)
