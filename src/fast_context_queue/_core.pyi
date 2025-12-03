Q_EMPTY: int
Q_FULL: int
Q_MSG_BUFFER_TOO_SMALL: int
Q_SUCCESS: int

class CircularBuffer:
    """A circular buffer for process communication."""
    def __init__(
        self,
        max_byte_size: int = ...,
        max_size: int = ...,
        name: str = ...,
        create: bool = ...,
        auto_unlink: bool = ...,
    ) -> None:
        """__init__(self: _core.CircularBuffer, max_byte_size: int = 10000000, max_size: int = 1000000000, name: str = '', create: bool = True, auto_unlink: bool = True) -> None

        Create a circular buffer.

        Args:
            max_byte_size (int): Maximum size of the buffer in bytes.
            max_size (int): Maximum number of elements in the buffer.
            name (str): shared memory name.
            create (bool): whether to create first.
            auto_unlink (bool): whether to unlink the shared memory when the buffer is destroyed.
        """
    def get_data_size(self) -> int:
        """get_data_size(self: _core.CircularBuffer) -> int"""
    def get_queue_size(self) -> int:
        """get_queue_size(self: _core.CircularBuffer) -> int"""
    def is_queue_full(self) -> bool:
        """is_queue_full(self: _core.CircularBuffer) -> bool"""
    def queue_get(
        self,
        msg_buffer: Buffer,
        msg_buffer_size: int,
        max_messages_to_get: int,
        max_bytes_to_get: int,
        message_read: Buffer,
        bytes_read: Buffer,
        messages_size: Buffer,
        block: int,
        timeout: float,
    ) -> int:
        """queue_get(self: _core.CircularBuffer, msg_buffer: Buffer, msg_buffer_size: int, max_messages_to_get: int, max_bytes_to_get: int, message_read: Buffer, bytes_read: Buffer, messages_size: Buffer, block: int, timeout: float) -> int

        Get messages from the buffer.
        """
    def queue_put(
        self,
        msgs_data: Buffer,
        msg_sizes: Buffer,
        num_msgs: int,
        block: int,
        timeout: float,
    ) -> None:
        """queue_put(self: _core.CircularBuffer, msgs_data: Buffer, msg_sizes: Buffer, num_msgs: int, block: int, timeout: float) -> None

        Put messages into the buffer.
        """
