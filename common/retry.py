"""
Retry utilities with exponential backoff for API calls
"""

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional

logger = logging.getLogger('osmid.retry')


def retry_with_backoff(
    max_retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: Optional[logging.Logger] = None
):
    """
    Decorator for exponential backoff retry logic

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        logger_instance: Logger to use for retry messages

    Usage:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def my_api_call():
            # API call that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger_instance or logger

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        log.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    log.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

            # Should never reach here
            return None

        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiter for API calls

    Usage:
        limiter = RateLimiter(calls_per_second=2.0)

        for item in items:
            with limiter:
                api_call(item)
    """

    def __init__(self, calls_per_second: float):
        """
        Initialize rate limiter

        Args:
            calls_per_second: Maximum number of calls per second
        """
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0

    def __enter__(self):
        """Wait if necessary before allowing call"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass
