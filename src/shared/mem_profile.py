from __future__ import annotations

import contextlib
import logging
import os
import tracemalloc
from typing import Generator

log = logging.getLogger(__name__)

# Truthy values that enable memory_profiler decorators.
_TRUTHY = {"1", "true", "yes", "on"}
_PROFILE_ACTIVE = os.environ.get("PROFILE_MEMORY", "0").strip().lower() in _TRUTHY

try:
    from memory_profiler import profile as _mp_profile  # type: ignore[import-untyped]
except ImportError:
    _mp_profile = None  # package not installed — profile_memory becomes a no-op


@contextlib.contextmanager
def tracemalloc_snapshot(label: str, top_n: int = 10) -> Generator[None, None, None]:
    """Context manager: capture heap allocation delta around a code block.

    Only active when ``PROFILE_MEMORY=1``.  Otherwise a transparent no-op.
    """
    if not _PROFILE_ACTIVE:
        yield
        return

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start(25)

    snapshot_before = tracemalloc.take_snapshot()
    stats_before = snapshot_before.statistics("lineno")
    mem_before = sum(s.size for s in stats_before)

    try:
        yield
    finally:
        snapshot_after = tracemalloc.take_snapshot()
        _log_memory_delta(label, snapshot_before, snapshot_after, mem_before, top_n)
        if not already_tracing:
            tracemalloc.stop()


def _log_memory_delta(label, snapshot_before, snapshot_after, mem_before, top_n):
    """Log heap memory change and top allocation sites."""
    stats_after = snapshot_after.statistics("lineno")
    mem_after = sum(s.size for s in stats_after)

    diff = mem_after - mem_before
    sign = "+" if diff >= 0 else ""
    log.info(
        "[mem] %s: %s%d KB  (%.2f MB → %.2f MB)",
        label, sign, diff // 1024,
        mem_before / 1024 / 1024, mem_after / 1024 / 1024,
    )

    top = snapshot_after.compare_to(snapshot_before, "lineno")[:top_n]
    for rank, stat in enumerate(top, 1):
        if stat.size_diff == 0:
            continue
        site = str(stat.traceback[0]) if stat.traceback else "<unknown>"
        log.debug(
            "[mem]  #%-2d  %+8.1f KB  count %+d  |  %s",
            rank,
            stat.size_diff / 1024,
                stat.count_diff,
                site,
            )


def profile_memory(fn):
    """Decorator: line-level RAM profiling when ``PROFILE_MEMORY=1``.

    When the env-var is **not** set (default) this is a transparent no-op and
    adds zero runtime overhead.

    When active, wraps the function with ``memory_profiler.profile`` which
    prints a per-line memory table to stdout after the function returns.

    Requirements::

        pip install memory-profiler

    If the package is missing and ``PROFILE_MEMORY=1`` is set, a single
    warning is logged and the function runs unwrapped.

    Example::

        @profile_memory
        def run_cinematic(self, duration, fps, ...):
            ...

    Output example (only when PROFILE_MEMORY=1)::

        Line #    Mem usage    Increment   Line Contents
        ================================================
           235   245.3 MiB    245.3 MiB   def run_cinematic(self, ...):
           ...
    """
    if not _PROFILE_ACTIVE:
        return fn

    if _mp_profile is None:
        log.warning(
            "[mem] PROFILE_MEMORY=1 but 'memory-profiler' is not installed. "
            "Run:  pip install memory-profiler"
        )
        return fn

    decorated = _mp_profile(fn)
    log.debug(
        "[mem] memory_profiler active → %s.%s",
        fn.__module__,
        fn.__qualname__,
    )
    return decorated
