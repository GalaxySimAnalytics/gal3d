import inspect
import logging
import textwrap
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

logger = logging.getLogger("gal3d.util.func_signature")

T = TypeVar('T', bound=Callable[..., Any])

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
    def params(self) -> Dict[str, tuple]:
        """
        Returns a dictionary mapping parameter names to their kind and default values.

        Returns
        -------
        dict
            A dictionary where keys are parameter names and values are tuples of
            (parameter kind, default value).
        """
        try:
            return {
                i: (self.parameters[i].kind, self.parameters[i].default)
                for i in self.parameters
            }
        except Exception as e:
            logger.error(f"Error retrieving parameter information: {e}")
            raise ValueError(f"Failed to retrieve parameter information: {e}") from e

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
        except Exception as e:
            logger.error(f"Error checking for kwargs parameter: {e}")
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
        except Exception as e:
            logger.error(f"Error checking for args parameter: {e}")
            return False

    def get_params(self, positional: int = 0, keyword: int = 0, empty: int = 0) -> Dict[str, Any]:
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
                inspect.Parameter.POSITIONAL_ONLY,        # 0
                inspect.Parameter.POSITIONAL_OR_KEYWORD,  # 1
                inspect.Parameter.KEYWORD_ONLY            # 3
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
                    k for k in available_kinds if 
                    k == inspect.Parameter.POSITIONAL_OR_KEYWORD or k == inspect.Parameter.KEYWORD_ONLY
                ]
            elif keyword == 2:
                # Only strictly keyword parameters
                available_kinds = [inspect.Parameter.KEYWORD_ONLY]

            # Create filter function based on empty parameter
            if empty == 0:
                # Include all parameters regardless of default value
                empty_filter = lambda x: True
            elif empty == 1:
                # Only parameters with no default value
                empty_filter = lambda x: x == inspect.Parameter.empty
            else:  # empty == 2
                # Only parameters with a default value
                empty_filter = lambda x: x != inspect.Parameter.empty

            # Apply filters and build result
            params_result = {}
            for param_name in self.params:
                param_kind, param_default = self.params[param_name]
                if param_kind in available_kinds and empty_filter(param_default):
                    params_result[param_name] = param_default

            return params_result
            
        except ValueError as e:
            # Re-raise validation errors
            logger.error(f"Invalid parameter for get_params: {e}")
            raise
        except Exception as e:
            # Handle unexpected errors
            err_msg = f"Error filtering parameters: {e}"
            logger.error(err_msg, exc_info=True)
            raise RuntimeError(err_msg) from e


def update_dict_value(origin: dict, other: dict, **kwargs) -> dict:
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
    >>> original = {'a': 1, 'b': 2}
    >>> updated = update_dict_value(original, {'b': 3}, c=4)
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
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


# Helper functions to extract parameters from callable objects
func_optional_key = lambda x: MySignature.from_callable(x).get_params(
    positional=0,
    keyword=1,
    empty=2,
)
func_required_key = lambda x: MySignature.from_callable(x).get_params(
    positional=0,
    keyword=0,
    empty=1,
)


def format_signature(func: Callable) -> str:
    """
    Returns the formatted string representation of a function's signature.

    Parameters
    ----------
    func : callable
        The function whose signature is to be formatted.

    Returns
    -------
    str
        A string representing the function name and its signature.
        
    Raises
    ------
    ValueError
        If unable to retrieve the function's signature.
    """
    try:
        sig = inspect.signature(func)
        return f"{func.__name__}{sig}"
    except Exception as e:
        error_msg = f"Failed to format signature for {func.__name__}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def format_docstring(docstring: str, indent: int = 4) -> str:
    """
    Formats a docstring with the specified indentation.

    Parameters
    ----------
    docstring : str
        The docstring to be formatted.
    indent : int, optional
        The number of spaces to indent the docstring, by default 4.

    Returns
    -------
    str
        The formatted docstring with the specified indentation.
        
    Raises
    ------
    RuntimeError
        If the docstring formatting process fails.
    ValueError
        If the indent value is negative.
    """
    if not docstring:
        return ""
    
    if indent < 0:
        raise ValueError("Indent value must be non-negative")
        
    try:
        # Add triple quotes and indent the docstring
        lines = textwrap.indent('"""\n' + docstring.strip() + '\n"""', " " * indent)
        return lines
    except Exception as e:
        error_msg = f"Failed to format docstring: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def is_static_or_class_method(cls: Type, attr_name: str) -> Optional[str]:
    """
    Determines if the given attribute of the class is a static method or class method.

    Parameters
    ----------
    cls : type
        The class to check.
    attr_name : str
        The attribute name to check within the class.

    Returns
    -------
    str or None
        Returns '@staticmethod' or '@classmethod' if the attribute is a static or class method,
        respectively. Returns None otherwise.
        
    Notes
    -----
    This function inspects the class's __dict__ attribute to determine the type of method.
    It does not handle inherited methods or descriptors other than staticmethod/classmethod.
    """
    try:
        # Get the attribute from the class's __dict__ (not from parent classes)
        attr = cls.__dict__.get(attr_name)
        
        # Check if it's a staticmethod or classmethod
        if isinstance(attr, staticmethod):
            return "@staticmethod"
        elif isinstance(attr, classmethod):
            return "@classmethod"
        
        # Additional checks could be implemented for property or other descriptor types
        return None
    except Exception as e:
        logger.warning(f"Error checking method type for {cls.__name__}.{attr_name}: {e}")
        return None


def generate_plugin_stub(
    base: Type, 
    abc: Type, 
    plugins: Dict[str, Type], 
    output_path: str
) -> None:
    """
    Generates a plugin stub that includes the base class, abstract class, and plugin classes.

    The generated code includes method stubs with docstrings for the methods in the base and abstract
    classes, and overloads for the `get_plugin` method for each plugin class.

    Parameters
    ----------
    base : type
        The base class for the plugin system.
    abc : type
        The abstract base class for the plugin system.
    plugins : dict
        A dictionary where keys are plugin names and values are plugin classes.
    output_path : str
        The path to save the generated plugin stub file.

    Returns
    -------
    None
        The function writes the generated plugin stub to the specified `output_path`.
        
    Raises
    ------
    IOError
        If there's an error writing to the output file.
    ValueError
        If required method information can't be extracted.
    """
    try:
        # Initialize lines list to store generated code
        stub_lines = [
            "import typing",
            "from typing import overload, Type, Literal, List, NoReturn, Union, Any, Sequence",
            "import numpy",
            "import gal3d",
            f"from {abc.__module__} import {abc.__name__}",
            *[f"from {cls.__module__} import {cls.__name__}" for cls in plugins.values()],
            "",
        ]
        
        # Generate abstract base class stub
        logger.debug(f"Generating stub for abstract class {abc.__name__}")
        stub_lines.append(f"class {abc.__name__}:")
        stub_lines.append("")
        
        # Process methods in abstract class
        for name, func in abc.__dict__.items():
            # Skip non-function attributes
            if isinstance(func, (staticmethod, classmethod)):
                func = func.__func__
            elif inspect.isfunction(func):
                pass
            else:
                continue
                
            # Add appropriate decorator if method is static or class method
            decorator = is_static_or_class_method(abc, name)
            if decorator:
                stub_lines.append(f"    {decorator}")
                
            # Extract function information
            docstring = inspect.getdoc(func)
            sig = inspect.signature(func)
            ret = func.__annotations__.get('return', 'None')
            ret_type = ret.__name__ if hasattr(ret, '__name__') else str(ret)
            
            # Generate function stub with docstring
            if docstring:
                if '->' in str(sig):
                    stub_lines.append(f"    def {name}{sig}:")
                else:
                    stub_lines.append(f"    def {name}{sig} -> None:")
                stub_lines.append(format_docstring(docstring, indent=8))
                stub_lines.append(f"        ...")
            else:
                if '->' in str(sig):
                    stub_lines.append(f"    def {name}{sig}: ...")
                else:
                    stub_lines.append(f"    def {name}{sig} -> None: ...")
            stub_lines.append("")
        
        # Generate base class stub
        logger.debug(f"Generating stub for base class {base.__name__}")
        stub_lines.append(f"class {base.__name__}:")
        stub_lines.append("")
        
        # Store get_plugin information for later use
        get_plugin_info = None
        
        # Process methods in base class
        for name, func in base.__dict__.items():
            if isinstance(func, (staticmethod, classmethod)):
                func = func.__func__
            elif inspect.isfunction(func):
                pass
            else:
                continue
                
            # Handle get_plugin method separately
            if name == "get_plugin":
                get_plugin_deco = is_static_or_class_method(base, name)
                get_plugin_sig = inspect.signature(func)
                get_plugin_docstring = inspect.getdoc(func)
                get_plugin_info = (get_plugin_deco, get_plugin_sig, get_plugin_docstring)
                continue
                
            # Process other methods
            decorator = is_static_or_class_method(base, name)
            if decorator:
                stub_lines.append(f"    {decorator}")
                
            docstring = inspect.getdoc(func)
            sig = inspect.signature(func)
            ret = func.__annotations__.get('return', 'None')
            ret_type = ret.__name__ if hasattr(ret, '__name__') else str(ret)
            
            # Generate function stub
            if docstring:
                stub_lines.append(f"    def {name}{sig} -> {ret_type}:")
                stub_lines.append(format_docstring(docstring, indent=8))
                stub_lines.append(f"        ...")
            else:
                stub_lines.append(f"    def {name}{sig} -> {ret_type}: ...")
            stub_lines.append("")
        
        # Generate get_plugin overloads
        if get_plugin_info:
            get_plugin_deco, get_plugin_sig, get_plugin_docstring = get_plugin_info
            
            # Add decorator if needed
            if get_plugin_deco:
                stub_lines.append(f"    {get_plugin_deco}")
                
            # Add base overload for None input
            stub_lines.append(f"    @overload")
            if get_plugin_docstring:
                stub_lines.append(f"    def get_plugin(plugin: None) -> {abc.__name__}:")
                stub_lines.append(format_docstring(get_plugin_docstring, indent=8))
                stub_lines.append(f"        ...")
            else:
                stub_lines.append(f"    def get_plugin(plugin: None) -> {abc.__name__}:...")
            
            stub_lines.append("")
            
            # Add overloads for each plugin
            for plugin_key, plugin_cls in plugins.items():
                if get_plugin_deco:
                    stub_lines.append(f"    {get_plugin_deco}")
                    
                plugin_name = plugin_cls.__name__
                stub_lines.append(f"    @overload")
                stub_lines.append(
                    f"    def get_plugin(plugin: Literal['{plugin_key}']) -> Type[{plugin_name}]:..."
                )
                stub_lines.append("")
        
        # Write to output file
        logger.info(f"Writing plugin stub to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(stub_lines))
            
        logger.info(f"Successfully generated plugin stub with {len(plugins)} plugins")
            
    except IOError as e:
        error_msg = f"Failed to write stub to {output_path}: {e}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error generating plugin stub: {e}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg) from e


# Fix typo in function name but keep the old one for backward compatibility
def fromat_signature(func: Callable) -> str:
    """
    Returns the formatted string representation of a function's signature.
    
    This function is a deprecated alias for `format_signature` with a typo in the name.
    Please use `format_signature` instead.

    Parameters
    ----------
    func : callable
        The function whose signature is to be formatted.

    Returns
    -------
    str
        A string representing the function name and its signature.
    """
    logger.warning(
        "The function 'fromat_signature' is deprecated due to a typo. "
        "Please use 'format_signature' instead."
    )
    return format_signature(func)
