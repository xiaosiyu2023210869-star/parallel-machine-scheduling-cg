"""Shared parameter validation helpers for experiment entry points."""

from __future__ import annotations


def effective_machine_count(requested_m: int | float, job_count: int | float) -> int:
    """Return a valid machine count for an instance.

    The experiments sometimes sweep a requested machine count such as
    ``m in {3, 6, 9}``.  A scheduling instance with ``n`` jobs cannot use more
    than ``n`` nonempty single-machine schedules in the set-partitioning master,
    so the runnable value is ``min(requested_m, n)``.
    """

    requested = int(requested_m)
    n = int(job_count)
    if requested <= 0:
        raise ValueError("machine count m must be positive.")
    if n <= 0:
        raise ValueError("job count n must be positive.")
    return min(requested, n)


def machine_count_adjusted(requested_m: int | float, actual_m: int | float) -> bool:
    return int(requested_m) != int(actual_m)
