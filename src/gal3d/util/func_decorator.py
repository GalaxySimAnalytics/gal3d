import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["timer"]

# Define type variables for better type checking
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")
LoggerType = Any  # Ideally would be logging.Logger, but avoiding import


def timer(logger: LoggerType) -> Callable[[F], F]:
    """
    Decorator to measure the execution time of a function and log it.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance used to log the timing information.

    Returns
    -------
    _timer : callable
        A decorator that wraps the target function to log its execution time.

    Examples
    --------
    >>> import logging
    >>> logger = logging.getLogger("example")
    >>> @timer(logger)
    ... def slow_function():
    ...     import time
    ...     time.sleep(1)
    >>> slow_function()
    # Logger will record: "Slow function: 1.00000 sec"
    """

    def _timer(fun: F) -> F:
        name = fun.__name__

        @functools.wraps(fun)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = fun(*args, **kwargs)
                end_time = time.time()
                logname = name.replace("_", " ").capitalize()
                usetime = (end_time - start_time)
                logger.info("%s: %.2f sec", logname, usetime)
                return result
            except Exception as e:
                end_time = time.time()
                logger.exception(
                    "Error in %s after %.5f sec: %s",
                    name,
                    end_time - start_time,
                    repr(e),
                )
                raise RuntimeError(
                    f"Error in {name} after {end_time - start_time:.5f} sec: {repr(e)}"
                ) from e

        return wrapper  # type: ignore

    return _timer
