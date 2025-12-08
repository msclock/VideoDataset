"""Implementation of a fast context queue using a circular buffer."""

import pickle
from collections.abc import Callable
from multiprocessing.reduction import ForkingPickler
from queue import Empty, Full
from typing import Any

from fast_context_queue._core import (
    Q_EMPTY,
    Q_FULL,
    Q_SUCCESS,
    CircularBuffer,
)

_ForkingPickler = ForkingPickler


DEFAULT_TIMEOUT = float(10)
DEFAULT_CIRCULAR_BUFFER_SIZE = 1000 * 1000  # 1 Mb
INITIAL_RECV_BUFFER_SIZE = 5000


class QueueError(Exception):
    pass


class Queue:
    """A fast context queue using a circular buffer."""

    def __init__(
        self,
        max_buffer_size: int = DEFAULT_CIRCULAR_BUFFER_SIZE,
        maxsize: int = int(1e9),
        loads: Callable[[bytes], Any] | None = None,
        dumps: Callable[[Any], bytes] | None = None,
    ):
        """Initialize the queue.

        Args:
            max_buffer_size: Maximum size of the circular buffer in bytes.
            maxsize: Maximum size of the queue.
            loads: Function to load objects from bytes.
            dumps: Function to dump objects to bytes.
        """
        self.max_buffer_size = max_buffer_size
        self.maxsize = maxsize
        self.buffer = CircularBuffer(max_buffer_size, maxsize, auto_unlink=True)

        if loads is not None:
            self.loads = loads  # type: ignore[assignment, method-assign]
        if dumps is not None:
            self.dumps = dumps  # type: ignore[assignment, method-assign]

        self.last_error: str | None = None

    def _error(self, message: str) -> None:
        """Set last_error and raise QueueError."""
        self.last_error = message
        raise QueueError(message)

    def loads(self, msg_bytes: bytes) -> Any:
        """Loads."""
        return _ForkingPickler.loads(msg_bytes)

    def dumps(self, obj: Any) -> bytes:
        """Dumps."""
        return _ForkingPickler.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL).tobytes()

    def put(
        self,
        x: Any,
        block: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """Put an objects into the queue."""
        status = self.buffer.write(self.dumps(x), block, timeout)
        if status == Q_SUCCESS:
            pass
        elif status == Q_FULL:
            raise Full()
        else:
            raise Exception(f"Unexpected queue error {status}")

    def put_nowait(self, x: Any) -> None:
        """Put an object into the queue without blocking."""
        return self.put(x, block=False)

    def get(
        self,
        block: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Any:
        """Get an object from the queue."""
        msg_bytes, status = self.buffer.read(
            block,
            timeout,
        )

        if status == Q_SUCCESS:
            return self.loads(msg_bytes)
        elif status == Q_EMPTY:
            raise Empty()
        else:
            raise Exception(f"Unexpected queue error {status}")

    def get_nowait(self) -> Any:
        """Get a object from the queue without blocking."""
        return self.get(block=False)

    def qsize(self) -> int:
        """Return the number of messages in the queue."""
        return self.buffer.get_queue_size()

    def data_size(self) -> int:
        """Return the total number of bytes in the queue."""
        return self.buffer.get_data_size()

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise."""
        return self.buffer.is_queue_full()

    def join_thread(self) -> None:
        """Placeholder."""
        pass

    def cancel_join_thread(self) -> None:
        """Placeholder."""
        pass

    def close(self) -> None:
        """Placeholder."""
        pass
