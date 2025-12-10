"""multiprocessing queue context for torch.utils.data.DataLoader."""

from multiprocessing.context import _default_context


def Queue(self, maxsize: int = 0) -> fq.Queue:
    """Returns a queue object."""
    return Queue(maxsize, ctx=self.get_context())


def get_context(self, method=None):
    """Returns a context object."""
    ctx = _default_context.get_context(method)
    ctx.Queue = Queue
    return ctx
