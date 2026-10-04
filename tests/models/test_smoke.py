"""Model smoke checks on the Mac (``uv run pytest -m model``): every model loads and answers a known case.

Each check runs in its own process, so only one model is in memory at a time. Golden tests with
fixtures and thresholds arrive with M2–M7.
"""

from __future__ import annotations

import pytest

from granit.models import smoke


@pytest.mark.model
@pytest.mark.parametrize("key", list(smoke.CHECKS))
def test_smoke(key: str) -> None:
    result = smoke.run_isolated(key)
    assert result.ok, result.error
