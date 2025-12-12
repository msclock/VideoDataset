"""Test multiprocessing.Queue to send and receive a tensor from a worker process."""

import cProfile
import logging
import multiprocessing as mp
import pickle
import pstats
from collections.abc import Callable, Generator
from contextlib import contextmanager
from multiprocessing.reduction import ForkingPickler
from multiprocessing.synchronize import Event
from time import sleep
from typing import Any

import pytest
import torch
from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-untyped]
    LeRobotDataset,
)

import fast_context_queue

_reduction = fast_context_queue

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


def write_tensor_worker(
    exit_event: Event,
    _writer_send: Callable[[bytes | torch.Tensor], None],
    num_tensors: int = 200,
    serialize: bool = True,
) -> None:
    """Worker process that sends a tensor to the main process."""
    for _ in range(num_tensors):
        shared_tensor = torch.rand(1, 1280, 720)
        if serialize:
            _writer_send(
                ForkingPickler.dumps(
                    shared_tensor, protocol=pickle.HIGHEST_PROTOCOL
                ).tobytes()
            )
        else:
            _writer_send(shared_tensor)
    while not exit_event.is_set():
        continue


@pytest.mark.parametrize("queue_type", ["mp", "torch"])
def test_queue_single_tensor(queue_type: str) -> None:
    """Test different types of queues to send and receive a tensor from a worker."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 2000
    exit_event = mp.Event()
    if queue_type == "mp":
        q: mp.Queue = mp.Queue()
    elif queue_type == "torch":
        q = torch.multiprocessing.Queue()
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")

    sub_process = mp.Process(
        target=write_tensor_worker,
        args=(exit_event, q.put, num_tensors, False),
    )
    sub_process.start()
    sleep(3)
    with cprofile_context():
        for _ in range(num_tensors):
            q.get()
    exit_event.set()
    sub_process.join()


@pytest.mark.parametrize("queue_type", ["mp", "torch"])
def test_queue_context_with_lerobot(queue_type: str) -> None:
    """Profile the performance of the subprocess queue with DataLoader."""
    num_workers = 2
    if queue_type == "mp":
        get_context = mp.get_context
    elif queue_type == "torch":
        get_context = torch.multiprocessing.get_context
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")
    dataset = LeRobotDataset(
        repo_id=None,
        root="/mnt/public/fengli/lerobot/ucsd_kitchen_dataset",
        # root="/mnt/public/qiuying/iros/v30/task_2666",
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        multiprocessing_context=get_context("spawn"),
    )

    with cprofile_context():
        for _ in data_loader:
            pass
