import functools
import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

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
    ...
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
                clean = name.lstrip("_").replace("_", " ").strip()
                logname = clean.capitalize() if clean else name
                usetime = end_time - start_time
                logger.info("%s: %.2f sec", logname, usetime)
                return result
            except Exception as e:
                end_time = time.time()
                logger.exception("Error in %s after %.5f sec: %s", name, end_time - start_time, repr(e))
                raise RuntimeError(f"Error in {name} after {end_time - start_time:.5f} sec: {repr(e)}") from e

        return wrapper  # type: ignore

    return _timer


AnyFunc = Callable[..., Any]


def _build_warning_wrapper(func: AnyFunc, message: str, category: type[Warning]) -> AnyFunc:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(message, category=category, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper


def _rewrap_descriptor(original: Any, wrapped: AnyFunc) -> Any:
    if isinstance(original, classmethod):
        return classmethod(wrapped)
    if isinstance(original, staticmethod):
        return staticmethod(wrapped)
    return wrapped


def _make_warning_decorator(
    category: type[Warning], default_message_factory: Callable[[str], str]
) -> Callable[[Any, str | None], Any]:
    def decorator(func: Any, message: str | None = None) -> Any:
        if isinstance(func, str):

            def apply_to(target: F) -> F:
                return cast("F", decorator(target, message=func))

            return apply_to

        target: AnyFunc = func.__func__ if isinstance(func, (classmethod, staticmethod)) else func

        if message is None:
            name = getattr(target, "__name__", target.__class__.__name__)
            message = default_message_factory(name)

        wrapped = _build_warning_wrapper(target, message, category)
        return _rewrap_descriptor(func, wrapped)

    return decorator


_deprecated_impl = _make_warning_decorator(DeprecationWarning, lambda name: f"Call to deprecated function {name}.")


class DevelopmentWarning(UserWarning):
    """Warning for modules or features that are still under development."""


_development_warning_impl = _make_warning_decorator(
    DevelopmentWarning, lambda name: f"Call to function {name} which is under development."
)


@overload
def deprecated(func: F) -> F: ...
@overload
def deprecated(func: str) -> Callable[[F], F]: ...
def deprecated(func: F | str, message: str | None = None) -> F | Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.
    """
    return cast("F | Callable[[F], F]", _deprecated_impl(func, message))


@overload
def development_warning(func: F) -> F: ...
@overload
def development_warning(func: str) -> Callable[[F], F]: ...
def development_warning(func: F | str, message: str | None = None) -> F | Callable[[F], F]:
    """
    Decorator to mark functions as under development.
    """
    return cast("F | Callable[[F], F]", _development_warning_impl(func, message))
