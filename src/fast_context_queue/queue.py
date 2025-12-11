"""Implementation of a fast context queue using a shared memory pool."""
# mypy: allow-untyped-defs

import multiprocessing.queues
from typing import Any

import torch
from torch.multiprocessing.queue import ConnectionWrapper

from fast_context_queue._core import POOL_SIZE_DEFAULT, TorchSegment, handle_t


class _Queue(multiprocessing.queues.Queue):
    """A fast context queue using a shared memory pool."""

    def __init__(self, maxsize=0, *, ctx):
        """Use the same init method as multiprocessing.Queue."""
        super().__init__(maxsize, ctx=ctx)
        self.segment = TorchSegment(POOL_SIZE_DEFAULT * 10)
        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv

    def __getstate__(self):
        """Save the segment and the state of the queue."""
        return (
            self.segment,
            super().__getstate__(),
        )

    def __setstate__(self, state):
        """Load the segment and the state of the queue."""
        (
            self.segment,
            super_state,
        ) = state
        super().__setstate__(super_state)  # type: ignore[misc]

    def put(self, obj: Any, block: bool = True, timeout: float | None = None) -> None:
        """Fast context queue put method for dicts of tensors."""
        super().put(self.put_preprocess(obj), block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Fast context queue get method for dicts of tensors."""
        obj = super().get(block=block, timeout=timeout)
        if isinstance(obj, handle_t):
            return self.segment.restore_tensor(obj)
        else:
            return self.get_postprocess(obj)

    def put_preprocess(self, obj: Any) -> Any:
        """Convert tensors to handles from a tensor or a dict of tensors."""
        if isinstance(obj, torch.Tensor):
            return self.segment.save_tensor(obj)
        elif isinstance(obj, dict):
            return {k: self.put_preprocess(v) for k, v in obj.items()}
        else:
            return obj

    def get_postprocess(self, obj: Any) -> Any:
        """Convert handles to tensors from a handle or a dict of handles."""
        if isinstance(obj, handle_t):
            return self.segment.restore_tensor(obj)
        elif isinstance(obj, dict):
            return {k: self.get_postprocess(v) for k, v in obj.items()}
        else:
            return obj


def Queue(maxsize=0, *, ctx=None) -> _Queue:
    """Create a fast context queue using a circular buffer."""
    if ctx is None:
        ctx = torch.multiprocessing.get_context()
    return _Queue(maxsize, ctx=ctx)
