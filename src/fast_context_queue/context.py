"""multiprocessing queue context for torch.utils.data.DataLoader."""
# mypy: allow-untyped-defs

from multiprocessing.context import BaseContext, _default_context

import fast_context_queue.queue as fq


def _get_context(method: str | None = None) -> BaseContext:
    """Returns a context object."""
    ctx = _default_context.get_context(method)

    def queue_factory(maxsize: int = 0, ctx: BaseContext = ctx):
        return fq.Queue(maxsize, ctx=ctx)

    ctx.Queue = queue_factory  # type: ignore[method-assign]
    return ctx


get_context = _get_context
