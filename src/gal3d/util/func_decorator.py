import functools
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, overload

__all__ = ["timer"]

# Define type variables for better type checking
F = TypeVar("F", bound=Callable[..., Any])
if TYPE_CHECKING:
    from logging import Logger


def timer(logger: "Logger") -> Callable[[F], F]:
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

@overload
def deprecated(func: F) -> F: ...
@overload
def deprecated(func: str) -> Callable[[F], F]: ...
def deprecated(func: F | str, message: str | None = None) -> F | Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.

    Parameters
    ----------
    func : callable
        The function to be marked as deprecated.
    message : str, optional
        Additional message to include in the deprecation warning.

    Returns
    -------
    callable
        A wrapper function that raises a DeprecationWarning when called.

    Examples
    --------
    >>> @deprecated
    ... def old_function():
    ...     pass
    >>> old_function()
    # Raises DeprecationWarning: Call to deprecated function old_function.
    """
    if isinstance(func, str):
        return functools.partial(deprecated, message=func)

    if message is None:
        message = f"Call to deprecated function {func.__name__}."

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(message, category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper

class DevelopmentWarning(UserWarning):
    """Warning for modules or features that are still under development."""

@overload
def development_warning(func: F) -> F: ...
@overload
def development_warning(func: str) -> Callable[[F], F]: ...
def development_warning(func: F | str, message: str | None = None) -> F | Callable[[F], F]:
    """
    Decorator to mark functions as under development.

    Parameters
    ----------
    func : callable
        The function to be marked as under development.
    message : str, optional
        Additional message to include in the development warning.

    Returns
    -------
    callable
        A wrapper function that raises a UserWarning when called.

    Examples
    --------
    >>> @development_warning
    ... def new_function():
    ...     pass
    >>> new_function()
    # Raises UserWarning: Call to function new_function which is under development.
    """
    if isinstance(func, str):
        return functools.partial(development_warning, message=func)

    if message is None:
        message = f"Call to function {func.__name__} which is under development."

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(message, category=DevelopmentWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper
