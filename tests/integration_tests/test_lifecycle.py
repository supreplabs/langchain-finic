"""Separately authorized lifecycle, never selected by the ordinary live marker."""

import asyncio
import os
import runpy
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.lifecycle,
    pytest.mark.skipif(
        os.getenv("FINIC_LIFECYCLE_TESTS") != "1" or not os.getenv("FINIC_API_KEY"),
        reason="Set FINIC_LIFECYCLE_TESTS=1 and FINIC_API_KEY to create a synthetic server-side model",
    ),
]


def test_synthetic_model_lifecycle(tmp_path):
    runner = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/live_lifecycle.py"))
    output = Path(os.getenv("FINIC_LIFECYCLE_OUTPUT", str(tmp_path / "lifecycle-evidence.json")))
    result = asyncio.run(runner["run_live"](output))
    assert result["status"] == "passed"
