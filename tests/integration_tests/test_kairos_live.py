"""Read-only, opt-in contract checks against the live Kairos API."""

import os

import pytest
from kairos import Kairos

pytestmark = [
    pytest.mark.live,
    pytest.mark.kairos_live,
    pytest.mark.skipif(
        not os.getenv("KAIROS_API_KEY"),
        reason="Set KAIROS_API_KEY to run the live Kairos contract check",
    ),
]


def test_live_client_settings_authentication_and_response() -> None:
    """Authenticate with Kairos and retrieve non-mutating client settings."""
    client = Kairos(
        api_key=os.environ["KAIROS_API_KEY"],
        flush_interval=0,
        max_retries=0,
    )
    try:
        settings = client.client_settings()
    finally:
        client.shutdown()

    assert isinstance(settings, dict)
    assert settings
