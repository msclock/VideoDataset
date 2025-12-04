"""Init for the project."""

import multiprocessing
from multiprocessing import *

from fast_context_queue import queue

# Queue = queue.Queue
__all__ = multiprocessing.__all__

SpawnContext = get_context("spawn")
SpawnContext.Queue = queue.Queue
