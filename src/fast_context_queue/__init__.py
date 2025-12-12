"""Init for the project."""

from __future__ import annotations

import importlib
import os
import platform
from multiprocessing import reduction
from typing import Any

import torch
from torch.multiprocessing import reductions

from fast_context_queue._core import POOL_SIZE_DEFAULT, TorchSegment, handle_t


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

segment = TorchSegment(pool_size=POOL_SIZE_DEFAULT * 10)


def rebuild_tensor_cpu(handle: handle_t) -> torch.Tensor:
    """Rebuild tensor from handle."""
    return segment.restore_tensor(handle)


def reduce_tensor_cpu(tensor: torch.Tensor) -> tuple:
    """Reduce tensors on device cpu."""
    h = segment.save_tensor(tensor)
    return rebuild_tensor_cpu, (h,)


def _reduce_tensor(tensor: Any) -> Any:
    """Reduce tensors on device cpu or other devices."""
    if tensor.device.type == "cpu":
        return reduce_tensor_cpu(tensor)
    return reductions.reduce_tensor(tensor)


reduction.register(torch.Tensor, _reduce_tensor)
