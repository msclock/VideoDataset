"""Test multiprocessing.Queue to send and receive a tensor from a worker process."""

import cProfile
import logging
import pickle
import queue
from collections.abc import Callable
from multiprocessing import connection
from multiprocessing.reduction import ForkingPickler
from multiprocessing.synchronize import Event
from time import sleep

import pytest
import torch
import torch.multiprocessing as mp
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import fast_context_queue.queue as fq

logger = logging.getLogger(__name__)


def generate_tensor(result_queue: mp.Queue, interval: float = 0.01) -> None:
    """Generate 100 tensors and put them into result_queue."""
    for _ in range(100):
        tensor = torch.rand(1, 1280, 720)
        result_queue.put(tensor)


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
    num_tensors = 2000
    exit_event = mp.Event()
    if queue_type == "mp":
        q: mp.Queue = mp.Queue()
    elif queue_type == "fq":
        q = fq.Queue()
    elif queue_type == "torch":
        q = torch.multiprocessing.Queue()
    elif queue_type == "manager":
        q: queue.Queue = mp.Manager().Queue()
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


def test_worker_tensor_send_recv() -> None:
    """Test ForkingPickler to send and receive a tensor from a worker process."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 200
    exit_event = mp.Event()
    _reader, _writer = connection.Pipe(duplex=False)
    sub_process = mp.Process(
        target=write_tensor_worker,
        args=(
            exit_event,
            _writer.send_bytes,
            num_tensors,
        ),
    )
    sub_process.start()
    sleep(3)
    for _ in range(num_tensors):
        ForkingPickler.loads(_reader.recv_bytes())
    exit_event.set()
    sub_process.join()


def test_dataloader_mp_queue() -> None:
    """Profile the performance of the subprocess queue with DataLoader."""
    num_workers = 1
    num_tensors = 100
    tensors = [torch.rand(10, 1280, 720) for _ in range(num_tensors)]
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        num_workers=num_workers,
        multiprocessing_context=mp.get_context("spawn"),
    )

    data_loader_iter = iter(data_loader)
    count = 0
    for _ in next(data_loader_iter):
        count += 1
    assert count == num_tensors


def test_dataloader_fast_queue() -> None:
    """Profile the performance of the fast queue with DataLoader."""
    from fast_context_queue.context import get_context

    num_workers = 1
    num_tensors = 100
    tensors = [torch.rand(10, 1280, 720) for _ in range(num_tensors)]
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        num_workers=num_workers,
        multiprocessing_context=get_context("spawn"),
    )

    data_loader_iter = iter(data_loader)
    count = 0
    for _ in next(data_loader_iter):
        count += 1

    assert count == num_tensors


def test_dataloader_torch_queue() -> None:
    """Profile the performance of the subprocess queue with DataLoader."""
    num_workers = 1
    num_tensors = 100
    tensors = [torch.rand(10, 1280, 720) for _ in range(num_tensors)]
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*tensors),
        num_workers=num_workers,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )

    data_loader_iter = iter(data_loader)
    count = 0
    for _ in next(data_loader_iter):
        count += 1
    assert count == num_tensors


def test_lerobot_dataloader_torch_queue() -> None:
    """Profile the performance of the subprocess queue with DataLoader."""
    num_workers = 1
    dataset = LeRobotDataset(
        repo_id=None,
        root="/mnt/public/fengli/lerobot/ucsd_kitchen_dataset",
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )

    data_loader_iter = iter(data_loader)
    count = 0
    for _ in next(data_loader_iter):
        count += 1
    logger.info(f"count: {count}")
