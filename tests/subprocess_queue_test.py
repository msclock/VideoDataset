"""Test multiprocessing.Queue to send and receive a tensor from a worker process."""

import cProfile
import logging
import pickle
import queue
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import connection
from multiprocessing.reduction import ForkingPickler
from multiprocessing.synchronize import Event
from time import sleep

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


def test_subprocess_queue() -> None:
    """Test multiprocessing.Queue to send and receive a tensor from a worker process."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 2000
    exit_event = mp.Event()
    q: mp.Queue = mp.Queue()
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


def test_fast_queue() -> None:
    """Test multiprocessing.Queue to send and receive a tensor from a worker process."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 2000
    exit_event = mp.Event()
    q: fq.Queue = fq.Queue()
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


def test_torch_queue() -> None:
    """Test torch.multiprocessing.Queue."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 2000
    exit_event = mp.Event()
    q: torch.multiprocessing.Queue = torch.multiprocessing.Queue()
    sub_process = mp.Process(
        target=write_tensor_worker,
        args=(exit_event, q.put_nowait, num_tensors, False),
    )
    sub_process.start()
    sleep(3)
    for _ in range(num_tensors):
        q.get()
    exit_event.set()
    sub_process.join()


def test_manager_queue() -> None:
    """Test torch.multiprocessing.Queue."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 2000
    exit_event = mp.Event()
    q: queue.Queue = mp.Manager().Queue()
    sub_process = mp.Process(
        target=write_tensor_worker,
        args=(exit_event, q.put_nowait, num_tensors, False),
    )
    sub_process.start()
    sleep(3)
    for _ in range(num_tensors):
        q.get()
    exit_event.set()
    sub_process.join()


def test_worker_tensor_send_recv() -> None:
    """Test ForkingPickler to send and receive a tensor from a worker process."""
    mp.set_start_method("spawn", force=True)
    num_tensors = 2000
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


def test_subprocess_queue_process_pool() -> None:
    """This test is used to profile the performance of the subprocess queue.

    4 worker processes are used to generate configurable number of tensors to result
    queue in parallel and main process waits for the queue and receives the results.
    """
    num_workers = 4
    manager = mp.Manager()
    result_queues: list = [manager.Queue() for _ in range(num_workers)]
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                generate_tensor,
                result_queue,
            )
            for result_queue in result_queues
        ]
        for future in futures:
            future.result()

    count = 0
    for result_queue in result_queues:
        while not result_queue.empty():
            result_queue.get()
            count += 1

    assert count == 400


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
