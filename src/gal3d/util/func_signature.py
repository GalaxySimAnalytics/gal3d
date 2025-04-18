import inspect
import textwrap
from typing import Dict, Type
from collections.abc import Callable
from functools import cached_property

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
    def params(self) -> dict:
        """
        Returns a dictionary mapping parameter names to their kind and default values.

        Returns
        -------
        dict
            A dictionary where keys are parameter names and values are tuples of
            (parameter kind, default value).
        """
        return {i: (self.parameters[i].kind,self.parameters[i].default) for i in self.parameters}
    
    @cached_property
    def kwargs(self) -> bool:
        """
        Determines whether the function accepts keyword arguments.

        Returns
        -------
        bool
            True if the function accepts keyword arguments, False otherwise.
        """
        for i in self.params:
            if self.params[i][0] == 4:
                return True
        return False
    
    @cached_property
    def args(self) -> bool:
        """
        Determines whether the function accepts positional arguments.

        Returns
        -------
        bool
            True if the function accepts positional arguments, False otherwise.
        """
        for i in self.params:
            if self.params[i][0] == 2:
                return True
        return False
    
    def get_params(self, positional: int = 0,keyword: int = 0, empty: int = 0) -> dict:
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
        """
        avaiable = [0,1,3]
        
        levels = [0,1,2]
        
        if positional not in levels:
            raise ValueError(f"'positional' = {positional}, this is not a valid value")
        if keyword not in levels:
            raise ValueError(f"'keyword' = {keyword}, this is not a valid value")
        if empty not in levels:
            raise ValueError(f"'positional' = {empty}, this is not a valid value")
        
        
        if positional == 1:
            avaiable = list(filter(lambda x: x<2, avaiable))
        if positional == 2:
            avaiable = list(filter(lambda x: x==0, avaiable))
        if keyword ==1 :
            avaiable = list(filter(lambda x: (x==1 or x==3), avaiable))
        if keyword ==2:
            avaiable = list(filter(lambda x: (x==3),avaiable))
        
        if empty == 0:
            emptysel = lambda x: True
        else:
            emptysel = (lambda x: x==inspect.Parameter.empty) if empty==1 else (lambda x: x!=inspect.Parameter.empty)
        
        params = {}
        for i in self.params:
            if self.params[i][0] in avaiable:
                if emptysel(self.params[i][1]):
                    params[i] = self.params[i][1]
                
        return params
    
    

def update_dict_value(origin: dict, other: dict, **kwargs)->dict:
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
        A new dictionary with updated values.
    """
    ret = origin.copy()
    same_key = ret.keys() & other.keys()
    for i in same_key:
        ret[i] = other[i]
    same_key = ret.keys() & kwargs.keys()
    for i in same_key:
        ret[i] = kwargs[i]
    return ret
    
func_optional_key = lambda x: MySignature.from_callable(x).get_params(positional=0,keyword=1,empty=2,)
func_required_key = lambda x: MySignature.from_callable(x).get_params(positional=0,keyword=0,empty=1,)




def fromat_signature(func):
    sig = inspect.signature(func)
    return f"{func.__name__}{sig}"

def format_docstring(docstring,indent=4):
    if not docstring:
        return ""
    lines = textwrap.indent('"""\n'+docstring.strip()+'\n"""'," "*indent)
    return lines

def is_static_or_class_method(cls,attr_name):
    attr = cls.__dict__.get(attr_name)
    if isinstance(attr,staticmethod):
        return "@staticmethod"
    elif isinstance(attr,classmethod):
        return "@classmethod"
    return None

def generate_plugin_stub(base,abc, plugins: Dict[str, Type], output_path: str):
    lines = [
        "import typing",
        "from typing import overload, Type, Literal, List, NoReturn, Union, Any",
        "import numpy",
        f"from {abc.__module__} import {abc.__name__}",
        *[
            f"from {cls.__module__} import {cls.__name__}"
            for cls in plugins.values()
        ],
        "",]
    
    
    lines.append(f"class {abc.__name__}:")
    lines.append("")
    for name, func in abc.__dict__.items():
        if isinstance(func,(staticmethod,classmethod)):
            func = func.__func__
        elif inspect.isfunction(func):
            pass
        else:
            continue
        deco = is_static_or_class_method(abc,name)
        if not deco is None:
            lines.append(f"    {deco}")
        docstring = inspect.getdoc(func)
        sig = inspect.signature(func)
        ret = func.__annotations__.get('return','None')
        ret_type = ret.__name__ if hasattr(ret,'__name__') else str(ret)
        if docstring:
            if '->' in str(sig):
                lines.append(f"    def {name}{sig}:")
            else:
                lines.append(f"    def {name}{sig} -> None:")
            lines.append(format_docstring(docstring,indent=8))
            lines.append(f"        ...")
        else:
            if '->' in str(sig):
                lines.append(f"    def {name}{sig}: ...")
            else:
                lines.append(f"    def {name}{sig} -> None: ...")
        lines.append("")
    
    lines.append(f"class {base.__name__}:")
    lines.append("")
    for name, func in base.__dict__.items():
        if isinstance(func,(staticmethod,classmethod)):
            func = func.__func__
        elif inspect.isfunction(func):
            pass
        else:
            continue
        if name == "get_plugin":
            get_plugin_deco = is_static_or_class_method(base,name)

            get_plugin_sig = inspect.signature(func)
            get_plugin_docstring = inspect.getdoc(func)
            continue
        deco = is_static_or_class_method(base,name)
        if not deco is None:
            lines.append(f"    {deco}")
        docstring = inspect.getdoc(func)
        sig = inspect.signature(func)
        ret = func.__annotations__.get('return','None')
        ret_type = ret.__name__ if hasattr(ret,'__name__') else str(ret)
        if docstring:
            lines.append(f"    def {name}{sig} -> {ret_type}:")
            lines.append(format_docstring(docstring,indent=8))
            lines.append(f"        ...")
        else:
            lines.append(f"    def {name}{sig} -> {ret_type}: ...")
        lines.append("")


    if not get_plugin_deco is None:
            lines.append(f"    {get_plugin_deco}")
    lines.append(f"    @overload")
    if get_plugin_docstring:
        lines.append(f"    def get_plugin(plugin: None) -> {abc.__name__}:")
        lines.append(format_docstring(get_plugin_docstring,indent=8))
        lines.append(f"        ...")
    else:
        lines.append(f"    def get_plugin(plugin: None) -> {abc.__name__}:...")
        
    lines.append("")
    for plugin_key, plugin_cls in plugins.items():
        if not get_plugin_deco is None:
            lines.append(f"    {get_plugin_deco}")
        plugin_name = plugin_cls.__name__
        lines.append(f"    @overload")
        lines.append(f"    def get_plugin(plugin: Literal['{plugin_key}']) -> Type[{plugin_name}]:...")
        lines.append("")
    

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))