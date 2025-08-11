import functools
import json
import time
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

import numpy as np

__all__ = ['timer', 'classproperty', 'lru_cache']

# Define type variables for better type checking
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')
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
    logger = logger

    def _timer(fun: F) -> F:
        name = fun.__name__

        @functools.wraps(fun)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = fun(*args, **kwargs)
                end_time = time.time()
                logger.info(name.replace('_', ' ').capitalize() + ': ' + f"{(end_time-start_time):.5f} sec")
                return result
            except Exception as e:
                end_time = time.time()
                logger.error(f"Error in {name}: {str(e)} after {(end_time-start_time):.5f} sec")
                raise
        
        return wrapper  # type: ignore

    return _timer
