from __future__ import annotations

import logging

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class TestSettings(BaseSettings):
    """Settings class for datasets used in setting."""

    batch_size: int = 16
    num_workers: int = 2

    model_config = {
        "env_file": ".env.test",
        "env_prefix": "TEST_",
        "case_sensitive": False,
    }


training_settings = TestSettings()
