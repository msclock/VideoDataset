"""Test for settings."""

import os

from fast_context_queue.settings import global_settings, settings


def test_settings() -> None:
    """Test for settings."""
    assert settings.logging_level == os.getenv(
        "FAST_CONTEXT_QUEUE_LOGGING_LEVEL",
        "INFO",
    )
    assert str(global_settings.ci).lower() == os.getenv("CI", "False").lower()
