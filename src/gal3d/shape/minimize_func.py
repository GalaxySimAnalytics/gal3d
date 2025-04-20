from typing import Callable


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

    minimize_fn = {}

    @staticmethod
    def fn_registry(fn: str | Callable) -> Callable:
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
            The original function, unchanged.

        Raises
        ------
        TypeError
            If used as a decorator with a non-callable object.
        """
        if callable(fn):
            MinimizeFunc.minimize_fn[fn.__name__] = fn
            return fn

        fn_name = fn

        def decorator(fn: Callable) -> Callable:
            if callable(fn):
                MinimizeFunc.minimize_fn[fn_name] = fn
                return fn
            raise TypeError(f"try register {fn} as {fn_name}, but {fn} is not callable")

        return decorator


from .fns import *
