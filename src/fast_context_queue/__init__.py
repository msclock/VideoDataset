"""Init for the project."""

from __future__ import annotations

import importlib
import os
import platform


def _setup_environment() -> None:
    """Setup environment variables and paths."""
    if platform.system() == "Linux":
        # Linux: Update LD_LIBRARY_PATH
        lib_paths: list[str] = []

        # Add torch library path for _decoder extension
        try:
            torch = importlib.import_module("torch")
            lib_paths.append(torch.__path__[0] + "/lib")
        except ImportError as e:
            err_msg = "Unable to import torch. Please ensure torch is installed."
            raise ImportError(err_msg) from e

        if "LD_LIBRARY_PATH" in os.environ:
            lib_paths.extend(os.environ["LD_LIBRARY_PATH"].split(":"))

        os.environ["LD_LIBRARY_PATH"] = ":".join(filter(None, lib_paths))


_setup_environment()

from fast_context_queue.context import get_context  # noqa: E402
from fast_context_queue.queue import Queue  # noqa: E402

__all__ = ["Queue", "get_context"]
