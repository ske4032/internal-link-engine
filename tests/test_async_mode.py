"""Issue #1 Gotcha: `asyncio_mode = "auto"` means an async test needs no decorator.

The failure mode is silent. Under strict mode an undecorated `async def` test is not
run at all — pytest warns that the coroutine was never awaited and moves on, and the
suite stays green while testing nothing. So one async test records that it ran and a
following sync test asserts the record, which turns a silent skip into a red test.

These two must be run as a module; `-k` down to the second one alone will fail by
design, because in that case nothing has proved the first one executes.
"""

from __future__ import annotations

import asyncio

_EXECUTED: list[str] = []


def test_asyncio_mode_is_auto(pytestconfig) -> None:
    assert pytestconfig.getini("asyncio_mode") == "auto", (
        "pyproject.toml must set asyncio_mode = 'auto'; under strict mode every "
        "undecorated async test in this repo becomes a silent skip"
    )


async def test_async_test_bodies_execute() -> None:
    await asyncio.sleep(0)
    loop = asyncio.get_running_loop()
    assert loop.is_running()
    _EXECUTED.append("test_async_test_bodies_execute")


def test_the_async_test_above_was_not_silently_skipped() -> None:
    assert _EXECUTED == ["test_async_test_bodies_execute"], (
        "the async test in this module did not execute: pytest-asyncio is not "
        "installed, or asyncio_mode is not 'auto'"
    )
