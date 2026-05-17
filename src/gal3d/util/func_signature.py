import inspect
import logging
from collections.abc import Callable
from functools import cached_property
from typing import Any, TypeVar

logger = logging.getLogger("gal3d.util.func_signature")

T = TypeVar("T", bound=Callable[..., Any])


class MySignature(inspect.Signature):
    """
    A custom signature class that extends `inspect.Signature` to provide additional functionality
    for analyzing function parameters.

    This class provides methods to determine whether a function accepts keyword arguments,
    positional arguments, and to filter parameters based on their kind and default values.

    Attributes
    ----------
    params : dict
        A dictionary mapping parameter names to their kind and default values.
    kwargs : bool
        True if the function accepts keyword arguments, False otherwise.
    args : bool
        True if the function accepts positional arguments, False otherwise.

    Methods
    -------
    get_params(positional=0, keyword=0, empty=0)
        Filters and returns parameters based on their kind and default values.
    """

    @cached_property
    def params(self) -> dict[str, tuple]:
        """
        Returns a dictionary mapping parameter names to their kind and default values.

        Returns
        -------
        dict
            A dictionary where keys are parameter names and values are tuples of
            (parameter kind, default value).
        """
        try:
            return {i: (self.parameters[i].kind, self.parameters[i].default) for i in self.parameters}
        except Exception as e:
            logger.error("Error retrieving parameter information")
            raise ValueError("Failed to retrieve parameter information") from e

    @cached_property
    def kwargs(self) -> bool:
        """
        Determines whether the function accepts keyword arguments.

        Returns
        -------
        bool
            True if the function accepts keyword arguments, False otherwise.

        Notes
        -----
        This checks for VAR_KEYWORD parameter kind (kind=4),
        which represents **kwargs style parameters.
        """
        try:
            for param_name in self.params:
                if self.params[param_name][0] == inspect.Parameter.VAR_KEYWORD:  # 4
                    return True
            return False
        except Exception:
            logger.error("Error checking for kwargs parameter")
            return False

    @cached_property
    def args(self) -> bool:
        """
        Determines whether the function accepts positional arguments.

        Returns
        -------
        bool
            True if the function accepts positional arguments, False otherwise.

        Notes
        -----
        This checks for VAR_POSITIONAL parameter kind (kind=2),
        which represents *args style parameters.
        """
        try:
            for param_name in self.params:
                if self.params[param_name][0] == inspect.Parameter.VAR_POSITIONAL:  # 2
                    return True
            return False
        except Exception:
            logger.error("Error checking for args parameter")
            return False

    def get_params(self, positional: int = 0, keyword: int = 0, empty: int = 0) -> dict[str, Any]:
        """
        Filters and returns parameters based on their kind and default values.

        Parameters
        ----------
        positional : int, optional
            Specifies the type of positional parameters to include:
            - 0: Include all parameters (default).
            - 1: Include parameters that can be positional.
            - 2: Include only strictly positional parameters.
        keyword : int, optional
            Specifies the type of keyword parameters to include:
            - 0: Include all parameters (default).
            - 1: Include parameters that can be keyword.
            - 2: Include only strictly keyword parameters.
        empty : int, optional
            Specifies the type of default values to include:
            - 0: Include all parameters (default).
            - 1: Include parameters with no default value.
            - 2: Include parameters with a default value.

        Returns
        -------
        dict
            A dictionary where keys are parameter names and values are their default values.

        Examples
        --------
        To get parameters with no default value:
            - For positional parameters: `get_params(positional=2, keyword=0, empty=1)`
            - For keyword parameters: `get_params(positional=0, keyword=1, empty=1)`

        Notes
        -----
        Parameter kinds:
            - POSITIONAL_ONLY: 0
            - POSITIONAL_OR_KEYWORD: 1
            - VAR_POSITIONAL: 2
            - KEYWORD_ONLY: 3
            - VAR_KEYWORD: 4

        Raises
        ------
        ValueError
            If any of the provided filter values are invalid.
        """
        try:
            # Available parameter kinds to filter
            available_kinds = [
                inspect.Parameter.POSITIONAL_ONLY,  # 0
                inspect.Parameter.POSITIONAL_OR_KEYWORD,  # 1
                inspect.Parameter.KEYWORD_ONLY,  # 3
            ]

            valid_filter_levels = [0, 1, 2]

            # Validate input parameters
            if positional not in valid_filter_levels:
                raise ValueError(f"'positional' = {positional}, valid values are {valid_filter_levels}")
            if keyword not in valid_filter_levels:
                raise ValueError(f"'keyword' = {keyword}, valid values are {valid_filter_levels}")
            if empty not in valid_filter_levels:
                raise ValueError(f"'empty' = {empty}, valid values are {valid_filter_levels}")

            # Apply positional filter
            if positional == 1:
                # Parameters that can be positional (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD)
                available_kinds = [k for k in available_kinds if k < inspect.Parameter.VAR_POSITIONAL]
            elif positional == 2:
                # Only strictly positional parameters
                available_kinds = [inspect.Parameter.POSITIONAL_ONLY]

            # Apply keyword filter
            if keyword == 1:
                # Parameters that can be keyword (POSITIONAL_OR_KEYWORD, KEYWORD_ONLY)
                available_kinds = [
                    k
                    for k in available_kinds
                    if k in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                ]
            elif keyword == 2:
                # Only strictly keyword parameters
                available_kinds = [inspect.Parameter.KEYWORD_ONLY]

            # Create filter function based on empty parameter
            if empty == 0:
                # Include all parameters regardless of default value
                def empty_filter(x):
                    return True
            elif empty == 1:
                # Only parameters with no default value
                def empty_filter(x):
                    return x == inspect.Parameter.empty
            else:  # empty == 2
                # Only parameters with a default value
                def empty_filter(x):
                    return x != inspect.Parameter.empty

            # Apply filters and build result
            params_result = {}
            for param_name in self.params:
                param_kind, param_default = self.params[param_name]
                if param_kind in available_kinds and empty_filter(param_default):
                    params_result[param_name] = param_default

            return params_result

        except ValueError:
            # Re-raise validation errors
            logger.exception("Invalid parameter for get_params")
            raise
        except Exception as e:
            # Handle unexpected errors
            err_msg = f"Error filtering parameters: {e}"
            logger.exception(err_msg)
            raise RuntimeError(err_msg) from e


def update_dict_value(origin: dict, other: dict, **kwargs: dict) -> dict:
    """
    Updates the values in `origin` dictionary with values from `other` dictionary and `kwargs`.

    Parameters
    ----------
    origin : dict
        The original dictionary to be updated.
    other : dict
        A dictionary containing values to update `origin`.
    **kwargs : dict
        Additional key-value pairs to update `origin`.

    Returns
    -------
    dict
        A new dictionary with updated values, merging `origin`, `other`, and `kwargs`.

    Raises
    ------
    TypeError
        If `origin` or `other` is not a dictionary.
    RuntimeError
        If an error occurs during the update operation.

    Examples
    --------
    >>> original = {"a": 1, "b": 2}
    >>> updated = update_dict_value(original, {"b": 3}, c=4)
    >>> updated
    {'a': 1, 'b': 3, 'c': 4}
    """
    if not isinstance(origin, dict) or not isinstance(other, dict):
        raise TypeError("Both 'origin' and 'other' must be dictionaries.")
    try:
        # Create a copy of the original dictionary to avoid modifying it
        result_dict = origin.copy()

        # Update values from 'other' dictionary
        common_keys = result_dict.keys() & other.keys()
        for key in common_keys:
            result_dict[key] = other[key]

        # Update values from kwargs
        common_keys = result_dict.keys() & kwargs.keys()
        for key in common_keys:
            result_dict[key] = kwargs[key]

        return result_dict
    except Exception as e:
        error_msg = f"Failed to update dictionary: {e}"
        logger.exception("Failed to update dictionary: %s", origin)
        raise RuntimeError(error_msg) from e


# Helper functions to extract parameters from callable objects
def func_optional_key(x):
    return MySignature.from_callable(x).get_params(positional=0, keyword=1, empty=2)


def func_required_key(x):
    return MySignature.from_callable(x).get_params(positional=0, keyword=0, empty=1)
