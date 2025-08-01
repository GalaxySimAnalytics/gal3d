"""
Utility module for enhanced error handling and reporting in gal3d.

This module provides decorators and utility functions to standardize error handling
across the package, making debugging easier and error messages more informative.
"""

import functools
import inspect
import logging
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger("gal3d.util.error_handling")

F = TypeVar('F', bound=Callable[..., Any])

def capture_context(func: F) -> F:
    """
    Decorator that captures execution context when an error occurs.
    
    Wraps a function to catch exceptions and log them with additional context
    information, helping with debugging and error tracking.

    Parameters
    ----------
    func : Callable
        The function to wrap with enhanced error handling.

    Returns
    -------
    Callable
        The wrapped function with enhanced error handling.

    Examples
    --------
    >>> @capture_context
    ... def process_data(data):
    ...     # function implementation
    ...     return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get the function signature to understand what parameters were expected
            sig = inspect.signature(func)
            
            # Get the source file and line number
            frame = inspect.currentframe().f_back
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            
            # Build contextual error information
            context = {
                'function': func.__name__,
                'module': func.__module__,
                'file': filename,
                'line': lineno,
                'signature': str(sig),
                'traceback': traceback.format_exc()
            }
            
            # Try to safely add argument information (avoiding large data dumps)
            arg_context = {}
            for i, arg in enumerate(args):
                if isinstance(arg, (int, float, str, bool)) or arg is None:
                    arg_context[f'arg_{i}'] = arg
                else:
                    try:
                        arg_context[f'arg_{i}_type'] = type(arg).__name__
                        if hasattr(arg, 'shape'):  # For numpy arrays and similar
                            arg_context[f'arg_{i}_shape'] = str(arg.shape)
                    except:
                        arg_context[f'arg_{i}_type'] = 'unknown'
            
            # Add kwargs info similarly
            for k, v in kwargs.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    arg_context[f'kwarg_{k}'] = v
                else:
                    try:
                        arg_context[f'kwarg_{k}_type'] = type(v).__name__
                        if hasattr(v, 'shape'):
                            arg_context[f'kwarg_{k}_shape'] = str(v.shape)
                    except:
                        arg_context[f'kwarg_{k}_type'] = 'unknown'
            
            context['args_info'] = arg_context
            
            # Log the detailed error with context
            logger.error(
                f"Error in {func.__name__}: {type(e).__name__}: {str(e)}\n"
                f"Context: {context}",
                exc_info=True
            )
            
            # Re-raise the original exception with the same traceback
            raise
    
    return wrapper

def validate_inputs(constraints: Dict[str, Callable[[Any], bool]], error_msgs: Optional[Dict[str, str]] = None) -> Callable:
    """
    Decorator that validates function inputs against specified constraints.
    
    Parameters
    ----------
    constraints : Dict[str, Callable]
        Dictionary mapping parameter names to validation functions.
        Each validation function should return True if the input is valid.
    error_msgs : Dict[str, str], optional
        Custom error messages for each parameter. If not provided, a default message is used.

    Returns
    -------
    Callable
        Decorator function that validates inputs.

    Examples
    --------
    >>> @validate_inputs({'a': lambda x: x > 0, 'b': lambda x: isinstance(x, str)})
    ... def my_function(a, b):
    ...     return a + len(b)
    """
    if error_msgs is None:
        error_msgs = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind the arguments to parameter names
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Check each constrained parameter
            for param_name, constraint in constraints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not constraint(value):
                        error_msg = error_msgs.get(
                            param_name, 
                            f"Invalid value for parameter '{param_name}': {value}"
                        )
                        logger.error(f"{error_msg} in function {func.__name__}")
                        raise ValueError(error_msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
