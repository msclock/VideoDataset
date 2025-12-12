"""Test."""

import cProfile
import logging
import multiprocessing as mp
from multiprocessing import context

_ForkingPickler = context.reduction.ForkingPickler
import pstats
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.multiprocessing as mp

import fast_context_queue

fcq = fast_context_queue

logger = logging.getLogger(__name__)


@contextmanager
def cprofile_context(
    sort: str = "tottime", max_line: int = 10
) -> Generator[None, Any, None]:
    """Context manager to profile code using cProfile."""
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()

    pstats.Stats(pr).strip_dirs().sort_stats(sort).print_stats(max_line)


if __name__ == "__main__":
    num_workers = 1
    batch_size = 2
    num_tensors = 100
    tensors = [torch.rand(100, 1, 1280, 720) for _ in range(num_tensors)]
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("spawn"),
    )

    count = 0
    with cprofile_context():
        for _ in data_loader:
            count += 1
    if count != num_tensors:
        logger.error(f"Count {count} does not match num_tensors {num_tensors}")
