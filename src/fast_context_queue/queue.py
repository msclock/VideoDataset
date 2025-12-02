"""Main module to define FastContextQueue class."""

import io
import multiprocessing
import pickle
from multiprocessing.reduction import ForkingPickler
from typing import Any

_ForkingPickler = ForkingPickler


class SharedRingBufferWrapper:
    """This class is used to manage a circular buffer of fixed size on shared memory.

    It provides a inter-process communication mechanism between processes to exchange
    data in a safe and efficient way.
    """

    def __init__(self, size: int):
        """SharedRingBuffer impl in cpp."""
        pass

    def send(self, obj: Any) -> None:
        """Use send_bytes to send data in SharedRingBuffer impl in cpp."""
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        # self.send_bytes(buf.getvalue())

    def recv(self) -> Any:
        """Use recv_bytes to recv data in SharedRingBuffer impl in cpp."""
        # buf = self.recv_bytes()
        # return pickle.loads(buf)
        pass


class FastContextQueue(multiprocessing.Queue):
    """This class is a faster implementation of multiprocessing.Queue.

    By use of ring buffer based on shared-memory, it provides a high-throughput
    and low-latency communication mechanism between processes.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Override _send and _recv to use SharedRingBuffer."""
        super().__init__(*args, **kwargs)
        buffer = SharedRingBufferWrapper(1_000_000)
        self._reader: SharedRingBufferWrapper = buffer
        self._writer: SharedRingBufferWrapper = buffer
        self._send = self._writer.send
        self._recv = self._reader.recv
