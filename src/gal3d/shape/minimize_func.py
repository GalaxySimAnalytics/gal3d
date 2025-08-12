import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("gal3d.shape.minimize_func")


class MinimizeFunc:
    """
    A registry class for storing and registering functions to be used with minimization routines.

    This class allows users to register custom functions that are intended to be minimized
    during model fitting. Registered functions are stored in the `minimize_fn` dictionary,
    with their names as keys.

    Attributes
    ----------
    minimize_fn : dict
        A dictionary mapping function names to callable objects.
    """

    minimize_fn: dict[str, Callable] = {}

    @staticmethod
    def fn_registry(fn: str | Callable[..., Any]) -> Callable:
        """
        Register a function to the minimization function registry.

        This method can be used as a decorator or called directly with a function
        or a name and function pair.

        Parameters
        ----------
        fn : str or Callable
            If a callable is provided, it will be registered with its `__name__`.
            If a string is provided, a decorator will be returned to register a function
            under the given name.

        Returns
        -------
        Callable
            The original function if `fn` is callable, or a decorator function if `fn` is a string.

        Raises
        ------
        TypeError
            If used as a decorator with a non-callable object.

        Examples
        --------
        Register function directly:
        >>> def my_min_func(x):
        ...     return x**2
        >>> MinimizeFunc.fn_registry(my_min_func)  # Returns my_min_func

        Register with custom name:
        >>> @MinimizeFunc.fn_registry("custom_name")
        ... def my_other_func(x):
        ...     return abs(x)
        """
        try:
            # Case 1: Direct registration of a callable
            if callable(fn):
                func_name = getattr(fn, "__name__", str(fn))
                MinimizeFunc.minimize_fn[func_name] = fn
                logger.debug("Registered function '%s' to minimization registry",
                             func_name)
                return fn  # Return the function unchanged

            # Case 2: String name provided, return a decorator
            fn_name = str(fn)  # Ensure fn_name is a string

            def decorator(func: Callable) -> Callable:
                if not callable(func):
                    raise TypeError("Cannot register %s as '%s', object is not callable", func, fn_name)
                MinimizeFunc.minimize_fn[fn_name] = func
                logger.debug("Registered function '%s' as '%s' to minimization registry",
                             func.__name__, fn_name)
                return func

            return decorator

        except Exception as e:
            logger.exception(
                "Error registering function to minimization registry: %s",
                repr(e),
            )
            raise RuntimeError(
                f"Error registering function to minimization registry: {repr(e)}"
            ) from e
