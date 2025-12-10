"""Test multiprocessing.Queue to send and receive a tensor from a worker process."""

import cProfile
import logging
import pickle
from collections.abc import Callable
from multiprocessing.reduction import ForkingPickler
from multiprocessing.synchronize import Event
from time import sleep

import pytest
import torch
import torch.multiprocessing as mp
from lerobot.datasets.lerobot_dataset import (  # type: ignore[import-untyped]
    LeRobotDataset,
)

import fast_context_queue as fq

logger = logging.getLogger(__name__)


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
                    (True, shared_tensor), protocol=pickle.HIGHEST_PROTOCOL
                ).tobytes()
            )
        else:
            _writer_send(shared_tensor)
    while not exit_event.is_set():
        continue


@pytest.mark.parametrize("queue_type", ["mp", "fq", "torch", "manager"])
def test_queue_single_tensor(queue_type: str) -> None:
    """Test different types of queues to send and receive a tensor from a worker."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 1000
    exit_event = mp.Event()
    if queue_type == "mp":
        q: mp.Queue = mp.Queue()
    elif queue_type == "fq":
        q = fq.Queue()
    elif queue_type == "torch":
        q = torch.multiprocessing.Queue()
    elif queue_type == "manager":
        q = mp.Manager().Queue()  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")

    sub_process = mp.Process(
        target=write_tensor_worker,
        args=(exit_event, q.put_nowait, num_tensors, False),
    )
    sub_process.start()
    sleep(3)
    with cProfile.Profile() as pr:
        for _ in range(num_tensors):
            q.get()
        pr.print_stats(sort="tottime")
    exit_event.set()
    sub_process.join()


@pytest.mark.parametrize("queue_type", ["mp", "fq", "torch"])
def test_queue_context_in_dataloader(queue_type: str) -> None:
    """Test different types of queues to send and receive a tensor from a worker."""
    num_workers = 1
    num_tensors = 100
    if queue_type == "mp":
        get_context = mp.get_context
    elif queue_type == "fq":
        get_context = fq.get_context  # type: ignore[assignment]
    elif queue_type == "torch":
        get_context = torch.multiprocessing.get_context
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")
    tensors = [torch.rand(10, 1280, 720) for _ in range(num_tensors)]
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        num_workers=num_workers,
        multiprocessing_context=get_context("spawn"),
    )
    with cProfile.Profile() as pr:
        data_loader_iter = iter(data_loader)
        count = 0
        for _ in next(data_loader_iter):
            count += 1
        assert count == num_tensors
        pr.print_stats(sort="tottime")


@pytest.mark.parametrize("queue_type", ["mp", "fq", "torch"])
def test_queue_context_with_lerobot(queue_type: str) -> None:
    """Profile the performance of the subprocess queue with DataLoader."""
    num_workers = 1
    if queue_type == "mp":
        get_context = mp.get_context
    elif queue_type == "fq":
        get_context = fq.get_context  # type: ignore[assignment]
    elif queue_type == "torch":
        get_context = torch.multiprocessing.get_context
    else:
        raise ValueError(f"Unknown queue type: {queue_type}")
    dataset = LeRobotDataset(
        repo_id=None,
        root="/mnt/public/qiuying/iros/task_2666",
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        multiprocessing_context=get_context("spawn"),
    )

    with cProfile.Profile() as pr:
        data_loader_iter = iter(data_loader)
        for _ in next(data_loader_iter):
            pass
        pr.print_stats(sort="tottime")
