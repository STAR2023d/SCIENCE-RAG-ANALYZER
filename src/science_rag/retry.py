"""Retry with linear backoff for rate limits (429) and transient server errors (503)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def with_backoff[T](
    fn: Callable[[], T],
    *,
    retry_on: tuple[type[BaseException], ...],
    attempts: int = 5,
    base_delay: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(); on a retryable error wait base_delay * attempt seconds and try again.
    The last error is re-raised once attempts are exhausted."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = base_delay * attempt
            logger.warning(
                "%s on attempt %d/%d, retrying in %.0fs",
                type(exc).__name__,
                attempt,
                attempts,
                delay,
            )
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
