from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_voyage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A VOYAGE_* variable in the developer's shell must not feed VoyageSettings."""
    for var in list(os.environ):
        if var.upper().startswith("VOYAGE_"):
            monkeypatch.delenv(var, raising=False)
