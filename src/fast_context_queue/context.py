"""multiprocessing queue context for torch.utils.data.DataLoader."""

import multiprocessing
from multiprocessing import *

from fast_context_queue import queue

__all__ = multiprocessing.__all__

SpawnContext = get_context("spawn")
SpawnContext.Queue = queue.Queue
