"""Implementation of a fast context queue using a circular buffer."""
# mypy: allow-attr-def, allow-untyped-defs

import multiprocessing.queues
from typing import Any

import torch

from fast_context_queue._core import TorchSegment, handle_t


class Queue(multiprocessing.queues.Queue):
    """A fast context queue using a circular buffer."""

    def __init__(self, *args, **kwargs):
        """Use the same init method as multiprocessing.Queue."""
        super().__init__(*args, **kwargs)
        self.segment = TorchSegment()

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
        super().__setstate__(super_state)

    def put(self, obj: Any, block: bool = True, timeout: float | None = None) -> None:
        """Fast context queue put method for dicts of tensors."""
        super().put(self.put_preprocess(obj), block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Fast context queue get method for dicts of tensors."""
        obj = super().get(block=block, timeout=timeout)
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
        """Convert handles to tensors from a tensor or a dict of tensors."""
        if isinstance(obj, handle_t):
            return self.segment.restore_tensor(obj)
        elif isinstance(obj, dict):
            return {k: self.get_postprocess(v) for k, v in obj.items()}
        else:
            return obj
