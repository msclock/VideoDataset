"""Implementation of a fast context queue using a circular buffer."""

import ctypes
import threading
from ctypes import c_size_t
from multiprocessing.context import reduction
from queue import Empty, Full

from fast_context_queue._core import (
    Q_EMPTY,
    Q_FULL,
    Q_MSG_BUFFER_TOO_SMALL,
    Q_SUCCESS,
    CircularBuffer,
)

_ForkingPickler = reduction.ForkingPickler


DEFAULT_TIMEOUT = float(10)
DEFAULT_CIRCULAR_BUFFER_SIZE = 1000 * 1000  # 1 Mb
INITIAL_RECV_BUFFER_SIZE = 5000


def caddr(buf):
    """Return the address of the first byte in a buffer."""
    buf_ptr = ctypes.addressof(buf)
    return buf_ptr


def msg_buf_addr(q):
    """Return the address of the first byte in the message buffer of a queue."""
    return caddr(q.message_buffer.val)


def bytes_to_ptr(b):
    """Return the address of the first byte in a bytes object."""
    ptr = ctypes.cast(b, ctypes.POINTER(ctypes.c_byte))
    return ctypes.addressof(ptr.contents)


class QueueError(Exception):
    pass


class TLSBuffer(threading.local):
    """Used for recv message buffers, prevents race condition in multithreading (not a problem with multiprocessing)."""

    def __init__(self, v=None):
        self.val = v

    def __getstate__(self):
        message_buffer_size = 0 if self.val is None else len(self.val)
        return message_buffer_size

    def __setstate__(self, message_buffer_size):
        if message_buffer_size == 0:
            self.valmessage_buffer = None
        else:
            self.val = (ctypes.c_ubyte * message_buffer_size)()


class Queue:
    def __init__(
        self,
        max_size_bytes=DEFAULT_CIRCULAR_BUFFER_SIZE,
        maxsize=int(1e9),
        loads=None,
        dumps=None,
    ):
        self.max_size_bytes = max_size_bytes
        self.maxsize = maxsize  # default maxsize
        self.max_bytes_to_read = (
            self.max_size_bytes
        )  # by default, read the whole queue if necessary

        self.Q = CircularBuffer(max_size_bytes, maxsize)

        # allow per-instance serializer overriding
        if loads is not None:
            self.loads = loads
        if dumps is not None:
            self.dumps = dumps

        self.message_buffer: TLSBuffer = TLSBuffer(None)

        self.last_error: str | None = None

    def _error(self, message):
        self.last_error = message
        raise QueueError(message)

    # allow class level serializers
    def loads(self, msg_bytes):
        return _ForkingPickler.loads(msg_bytes)

    def dumps(self, obj):
        return _ForkingPickler.dumps(obj).tobytes()

    def close(self):
        """This is not atomic by any means, but using locks is expensive. So this should be preferably called by
        only one process, e.g. main process.
        """
        self.closed.value = True

    def is_closed(self):
        """This 'closed' variable is not atomic, so changes may not immediately propagate between processes.
        This should be okay for most usecases, but if 100% reliability is required perhaps another mechanism is needed.
        """
        return self.closed.value

    def put_many(self, xs, block=True, timeout=DEFAULT_TIMEOUT):
        if not isinstance(xs, (list, tuple)):
            self._error(f"put_many() expects a list or tuple, got {type(xs)}")

        xs = [self.dumps(ele) for ele in xs]

        _len = len
        msgs_buf = (c_size_t * _len(xs))()
        size_buf = (c_size_t * _len(xs))()

        for i, ele in enumerate(xs):
            msgs_buf[i] = bytes_to_ptr(ele)
            size_buf[i] = _len(ele)

        # explicitly convert all function parameters to corresponding C-types
        c_msgs_buf_addr = caddr(msgs_buf)
        c_size_buff_addr = caddr(size_buf)

        c_len_x = _len(xs)
        c_block = block
        c_timeout = timeout

        c_status = 0

        c_status = self.Q.queue_put(
            c_msgs_buf_addr,
            c_size_buff_addr,
            c_len_x,
            c_block,
            c_timeout,
        )

        status = c_status

        if status == Q_SUCCESS:
            pass
        elif status == Q_FULL:
            raise Full()
        else:
            raise Exception(f"Unexpected queue error {status}")

    def put(self, x, block=True, timeout=DEFAULT_TIMEOUT):
        status = self.put_many([x], block, timeout)
        if status == Q_FULL:
            raise Full()
        return status

    def put_many_nowait(self, xs):
        status = self.put_many(xs, block=False)
        if status == Q_FULL:
            raise Full()
        return status

    def put_nowait(self, x):
        status = self.put_many_nowait([x])
        if status == Q_FULL:
            raise Full()
        return status

    def get_many(
        self, block=True, timeout=DEFAULT_TIMEOUT, max_messages_to_get=int(1e9)
    ):
        if self.message_buffer.val is None:
            # initialize a small buffer at first, it will be increased later if needed
            self.reallocate_msg_buffer(INITIAL_RECV_BUFFER_SIZE)

        messages_read = ctypes.c_size_t(0)
        messages_read_ptr = ctypes.addressof(messages_read)

        bytes_read = ctypes.c_size_t(0)
        bytes_read_ptr = ctypes.addressof(bytes_read)

        messages_size = ctypes.c_size_t(
            0
        )  # this is how much memory we need to allocate to read more messages
        messages_size_ptr = ctypes.addressof(messages_size)

        # explicitly convert all function parameters to corresponding C-types
        c_msg_buf_addr = msg_buf_addr(self)

        c_block = block
        c_timeout = timeout
        c_max_messages_to_get = max_messages_to_get
        c_max_bytes_to_read = self.max_bytes_to_read
        c_len_message_buffer = len(self.message_buffer.val)

        c_status = 0

        c_status = self.Q.queue_get(
            c_msg_buf_addr,
            c_len_message_buffer,
            c_max_messages_to_get,
            c_max_bytes_to_read,
            messages_read_ptr,
            bytes_read_ptr,
            messages_size_ptr,
            c_block,
            c_timeout,
        )

        status = c_status

        if status == Q_MSG_BUFFER_TOO_SMALL and messages_read.value <= 0:
            # could not read any messages because msg buffer was too small
            # reallocate the buffer and try again
            self.reallocate_msg_buffer(int(messages_size.value * 1.5))
            return self.get_many_nowait(max_messages_to_get)
        elif status == Q_SUCCESS or status == Q_MSG_BUFFER_TOO_SMALL:
            # we definitely managed to read something!
            if messages_read.value <= 0 or bytes_read.value <= 0:
                self._error(
                    f"Expected to read at least 1 message, but got {messages_read.value} messages and {bytes_read.value} bytes"
                )
            messages = self.parse_messages(
                messages_read.value, bytes_read.value, self.message_buffer
            )

            if status == Q_MSG_BUFFER_TOO_SMALL:
                # we could not read as many messages as we wanted
                # allocate a bigger buffer so next time we can read more
                self.reallocate_msg_buffer(int(messages_size.value * 1.5))

            return messages

        elif status == Q_EMPTY:
            raise Empty()
        else:
            raise Exception(f"Unexpected queue error {status}")

    def get_many_nowait(self, max_messages_to_get=int(1e9)):
        return self.get_many(block=False, max_messages_to_get=max_messages_to_get)

    def get(self, block=True, timeout=DEFAULT_TIMEOUT):
        return self.get_many(block=block, timeout=timeout, max_messages_to_get=1)[0]

    def get_nowait(self):
        return self.get(block=False)

    def parse_messages(self, num_messages, total_bytes, msg_buffer):
        messages = [None] * num_messages

        offset = 0
        for msg_idx in range(num_messages):
            msg_size = c_size_t.from_buffer(msg_buffer.val, offset)
            offset += ctypes.sizeof(c_size_t)

            msg_bytes = memoryview(msg_buffer.val)[offset : offset + msg_size.value]
            # msg_bytes = msg_buffer.val[offset,msg_size.value] #memoryview(msg_buffer.val)[offset:offset + msg_size.value]
            offset += msg_size.value
            msg = self.loads(msg_bytes)
            messages[msg_idx] = msg

        if offset != total_bytes:
            self._error(f"Expected to read {total_bytes} bytes, but got {offset} bytes")
        return messages

    def reallocate_msg_buffer(self, new_size):
        new_size = max(INITIAL_RECV_BUFFER_SIZE, new_size)
        self.message_buffer.val = (ctypes.c_ubyte * new_size)()

    def qsize(self):
        """Return the number of messages in the queue."""
        return self.Q.get_queue_size()

    def data_size(self):
        """Return the total number of bytes in the queue."""
        return self.Q.get_data_size()

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def full(self):
        """Return True if the queue is full, False otherwise."""
        return self.Q.is_queue_full()
