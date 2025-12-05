"""Test fast queue with objects and tensors."""

import logging

import numpy as np
import torch

import fast_context_queue.queue as fq

logger = logging.getLogger(__name__)


def test_fast_queue_tensors() -> None:
    """Test fast queue with tensors."""
    q = fq.Queue()
    tensor = torch.rand(1, 1280, 720)
    q.put(tensor)
    assert torch.equal(q.get(), tensor)


def test_fast_queue_objects() -> None:
    """Test fast queue with objects."""
    objs = [
        None,
        1,
        "2",
        b"3",
        [4, 5],
        {"6": 7},
        tuple([8, 9]),
        set([10, 11]),
        {},
    ]
    q = fq.Queue()
    for obj in objs:
        q.put(obj)
        assert q.get() == obj


def test_fast_queue_numpy() -> None:
    """Test fast queue with numpy arrays."""
    q = fq.Queue()
    arr = np.random.rand(10, 10)
    q.put(arr)
    assert np.array_equal(q.get(), arr)


def test_fast_queue_tuple() -> None:
    """Test fast queue with tuple."""
    t = (1, 2, 3)
    q = fq.Queue()
    q.put(t)
    assert q.get() == t

    q.put((1, 2))
    assert q.get() == (1, 2)

    q.put((1,))
    assert q.get() == (1,)
