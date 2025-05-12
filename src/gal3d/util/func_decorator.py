import functools
import json
import time
from typing import Any, Callable, TypeVar, Dict, Union, List, Set, Optional

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


class classproperty:
    """
    Descriptor for creating read-only class-level properties.
    
    Similar to @property but works on classes instead of instances.
    
    Examples
    --------
    >>> class Example:
    ...     _value = 42
    ...     @classproperty
    ...     def value(cls):
    ...         return cls._value
    >>> Example.value
    42
    """
    def __init__(self, f: Callable[[Any], T]) -> None:
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> T:
        return self.f(owner)


HashDict = type('HashDict', (dict,), {'__hash__': lambda self: hash(json_hash(self))})
HashList = type('HashList', (list,), {'__hash__': lambda self: hash(json_hash(self))})
HashSet = type('HashSet', (set,), {'__hash__': lambda self: hash(json_hash(self))})


def json_hash(o: Any, *args: Any, **kwargs: Any) -> str:
    """
    Generate a JSON-based hash for an object.

    Parameters
    ----------
    o : object
        The object to be hashed.
    *args :
        Positional arguments passed to `json.dumps`.
    **kwargs :
        Keyword arguments passed to `json.dumps`.

    Returns
    -------
    str
        JSON string representation of the object used for hashing.
        
    Notes
    -----
    This function creates a deterministic string representation of objects
    for consistent hashing. For custom objects, it falls back to repr().
    """
    return json.dumps(o, *args, **dict({"sort_keys": True, "default": repr}, **kwargs))


@functools.singledispatch
def to_hash_object(obj: Any) -> Any:
    """
    Convert an object to a hashable form.

    Parameters
    ----------
    obj : object
        The input object.

    Returns
    -------
    object
        The object wrapped in a hashable container if applicable.
        
    Notes
    -----
    This is the default implementation which returns the object unchanged.
    Specialized implementations are registered for specific types.
    """
    return obj


@to_hash_object.register(dict)
def _(obj: dict) -> HashDict:
    """Convert a dictionary to a hashable form."""
    return HashDict(obj)


@to_hash_object.register(list)
def _(obj: list) -> HashList:
    """Convert a list to a hashable form."""
    return HashList(obj)


@to_hash_object.register(set)
def _(obj: set) -> HashSet:
    """Convert a set to a hashable form."""
    return HashSet(obj)


@to_hash_object.register(np.ndarray)
def _(obj: np.ndarray) -> bytes:
    """
    Convert a NumPy array to a hashable form.
    
    Parameters
    ----------
    obj : numpy.ndarray
        The NumPy array to make hashable.
        
    Returns
    -------
    bytes
        A bytes representation of the array that can be hashed.
        
    Notes
    -----
    This creates a hash that is sensitive to both the array's data content
    and its shape, dtype, etc.
    """
    # Get a consistent byte representation of the array
    # Include shape, dtype, and data for a complete representation
    try:
        array_bytes = bytes([
            hash(str(obj.shape)),
            hash(str(obj.dtype)),
            hash(obj.tobytes())
        ])
        return array_bytes
    except Exception:
        # Fallback for arrays that can't be directly converted to bytes
        return str(obj).encode('utf-8')


# https://luoruiqing.github.io/blog/2021/12/09/Python%E7%BC%93%E5%AD%98%E5%87%BD%E6%95%B0/
def lru_cache(*lru_args, **lru_kwargs):
    """
    A variant of `functools.lru_cache` that supports unhashable types by
    converting them into hashable forms.

    Parameters
    ----------
    *lru_args :
        Arguments passed to `functools.lru_cache`, such as `maxsize`.
    **lru_kwargs :
        Keyword arguments passed to `functools.lru_cache`.

    Returns
    -------
    callable
        A decorator that wraps a function with an LRU cache supporting
        dict, list, set, and NumPy array types.
    """
    def wrapper_cache(func):
        func = functools.lru_cache(*lru_args, **lru_kwargs)(func)

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(
                *map(to_hash_object, args),
                **{k: to_hash_object(v) for k, v in kwargs.items()},
            )

        wrapped_func.cache_info = func.cache_info
        wrapped_func.cache_clear = func.cache_clear
        return wrapped_func

    return wrapper_cache
