"""Test fast queue with objects and tensors."""

import numpy as np
import torch

import fast_context_queue.queue as fq


def test_fast_queue_tensors() -> None:
    """Test fast queue with tensors."""
    q = fq.Queue()
    tensor = torch.rand(1, 1280, 720)
    q.put(tensor)
    assert torch.equal(q.get(), tensor)


def test_fast_queue_objects() -> None:
    """Test fast queue with objects."""
    objs: list = [
        None,
        1,
        "2",
        b"3",
        {},
        [4, 5],
        {"6": 7},
        (8, 9),
        set([10, 11]),
        [[12, 13], [14, 15]],
        {16: [17, 18]},
        {19: {20: 21}},
        {22: (23, 24)},
        {25: [26, 27]},
        {28: {29: (30, 31)}},
        {32: [33, 34]},
        {35: {36: [37, 38]}},
        {39: {40: {41: [42, 43]}}},
        {44: {45: {46: {47: [48, 49]}}}},
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
    q = fq.Queue()
    t2 = (1, 2)
    q.put(t2)
    assert q.get() == t2
