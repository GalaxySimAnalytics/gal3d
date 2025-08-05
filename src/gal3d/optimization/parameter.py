import copy
import logging
from functools import wraps
from typing import Union, Optional, KeysView, Dict, Mapping,Iterable, Tuple, overload, Any, TypeVar,Self

import numpy as np
from scipy import optimize

from .util import truncate

# from optimagic import Bounds

_ParamDict = TypeVar("_ParamDict", bound="ParameterDict")

logger = logging.getLogger("gal3d.optimization.parameter")


__all__ = ['Parameters']


class Parameter(float):
    """
    A class representing a parameter with lower and upper bounds.

    This class extends the built-in `float` type to include lower (`lb`) and upper (`ub`) bounds.
    It is used to define parameters that can be fitted within a specified range.

    Attributes
    ----------
    _lb : float
        The lower bound of the parameter.
    _ub : float
        The upper bound of the parameter.

    Methods
    -------
    __new__(cls, value, **kwargs)
        Creates a new instance of the Parameter class.
    assign_value(value)
        Assigns a new value to the parameter while preserving the bounds.
    lb
        Property getter for the lower bound.
    ub
        Property getter for the upper bound.
    lb.setter
        Property setter for the lower bound.
    ub.setter
        Property setter for the upper bound.
    """

    __slots__ = ["_lb", "_ub"]

    def __new__(cls, value: Union[float, "Parameter"], lb: Optional[float] = None, ub: Optional[float] = None):
        """
        Creates a new instance of the Parameter class.

        Parameters
        ----------
        value : float or Parameter
            The initial value of the parameter. If `value` is an instance of `Parameter`,
            the new instance will inherit the bounds from `value` unless overridden by `lb` or `ub`.
        lb : float, optional
            The lower bound of the parameter. Default is -np.inf.
        ub : float, optional
            The upper bound of the parameter. Default is np.inf.

        Returns
        -------
        Parameter
            A new instance of the Parameter class.
        """
        instance = float.__new__(cls, value)

        if isinstance(value, Parameter):
            object.__setattr__(instance, '_lb', float(lb) if lb is not None else value.lb)
            object.__setattr__(instance, '_ub', float(ub) if ub is not None else value.ub)
            return instance
        object.__setattr__(instance, '_lb', float(lb) if lb is not None else -np.inf)
        object.__setattr__(instance, '_ub', float(ub) if ub is not None else np.inf)
        return instance

    def assign_value(self, value: float):
        """
        Assigns a new value to the parameter while preserving the bounds.

        Parameters
        ----------
        value : float
            The new value to assign to the parameter.

        Returns
        -------
        Parameter
            A new instance of the Parameter class with the updated value and the same bounds.

        Examples
        --------
        >>> param = Parameter(1.0)
        >>> param = param.assign_value(2.0)
        >>> param
        2.0
        """
        inst = self.__class__.__new__(self.__class__, value, lb=self.lb, ub=self.ub)
        return inst

    @property
    def lb(self) -> float:
        """
        Get the lower bound of the parameter.

        Returns
        -------
        float
            The lower bound of the parameter.
        """
        return self._lb
    
    @lb.setter
    def lb(self, value : float):
        """
        Set the lower bound of the parameter.

        Parameters
        ----------
        value : float
            The new lower bound of the parameter.
        """
        self._lb = float(value)

    @property
    def ub(self) -> float:
        """
        Get the upper bound of the parameter.

        Returns
        -------
        float
            The upper bound of the parameter.
        """
        return self._ub

    @ub.setter
    def ub(self, value : float):
        """
        Set the upper bound of the parameter.

        Parameters
        ----------
        value : float
            The new upper bound of the parameter.
        """
        self._ub = float(value)

class ParameterDict(dict):
    """
    A dictionary-like class that enforces all values to be of Parameter type.
    
    This class extends the built-in `dict` type to ensure all values are instances
    of the `Parameter` class. When assigning a value that is not a `Parameter`, 
    it automatically converts it to a `Parameter` instance.
    
    Attributes
    ----------
    None
    
    Methods
    -------
    __setitem__(key, value)
        Sets the value for a key, converting non-Parameter values to Parameter.
    deepcopy()
        Creates a deep copy of the ParameterDict.
    update(*args, **kwargs)
        Updates the dictionary, ensuring all values are converted to Parameter.
    get_rounded(n: int = 3, round_value: bool = True, round_bound: bool = False) -> 'ParameterDict'
        Returns a new ParameterDict with rounded values and/or bounds.
    lb
        Property getter for the lower bounds of all parameters.
    ub
        Property getter for the upper bounds of all parameters.
    set_lb(**kwargs)
        Sets the lower bounds of multiple parameters.
    set_ub(**kwargs)
        Sets the upper bounds of multiple parameters.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a new ParameterDict.
        
        Parameters
        ----------
        *args : dict or iterable
            Dictionary or iterable of key-value pairs to initialize with.
        **kwargs : dict
            Additional key-value pairs to initialize with.
        
        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params = ParameterDict({'a': 1.0, 'b': 2.0})
        """
        super().__init__()
        self.update(*args, **kwargs)

    def __getitem__(self, key: str) -> Parameter:
        
        return super().__getitem__(key)
    
    def __setitem__(self, key: str, value: Union[float, Parameter]) -> None:
        """
        Set a key-value pair, ensuring the value is a Parameter.
        
        Parameters
        ----------
        key : str
            The key to set.
        value : float or Parameter
            The value to set. If not a Parameter, it will be converted to one.
        """
        if isinstance(value, Parameter):
            super().__setitem__(key, Parameter(value))
        else:
            if key in self:
                param = super().__getitem__(key)
                value = Parameter(value, lb=param.lb, ub=param.ub)
                super().__setitem__(key, value)
            else:
                super().__setitem__(key, Parameter(value))
    
    
    def update(self, *args, **kwargs: float) -> None:
        """
        Update the dictionary with new key-value pairs.
        
        Parameters
        ----------
        *args : dict or iterable
            Dictionary or iterable of key-value pairs to update with.
        **kwargs : dict
            Additional key-value pairs to update with.
            
        Examples
        --------
        >>> params = ParameterDict()
        >>> params.update({'a': 1.0, 'b': 2.0})
        >>> params.update(c=3.0, d=4.0)
        """
        # Handle positional argument (dictionary or iterable of key-value pairs)
        if args:
            for other in args:
                if isinstance(other, Mapping):
                    for key in other:
                        self[key] = other[key]
                else:
                    # Assume it's an iterable of key-value pairs
                    for key, value in other:
                        self[key] = value
        # Handle keyword arguments
        for key, value in kwargs.items():
            self[key] = value
    
    def get_rounded(self:_ParamDict, n: int = 3, round_value: bool = True, round_bound: bool = False) -> 'ParameterDict':
        """
        Returns a new ParameterDict with rounded values and/or bounds.

        Parameters
        ----------
        n : int, optional
            Number of decimal places to round to. Default is 3.
        round_value : bool, optional
            Whether to round the parameter values. Default is True.
        round_bound : bool, optional
            Whether to round the parameter bounds (lb/ub). Default is False.

        Returns
        -------
        ParameterDict
            A new ParameterDict with rounded values and/or bounds.
        """
        if n < 0:
            raise ValueError("Decimal places must be non-negative.")
        return ParameterDict({
            k: Parameter(
                truncate(v, n) if round_value else v,
                lb=truncate(v.lb, n) if round_bound else v.lb,
                ub=truncate(v.ub, n) if round_bound else v.ub
            )
            for k, v in self.items()
        })
    @property
    def value(self) -> Dict[str, float]:
        """
        Get the values of all parameters.

        Returns
        -------
        dict
            A dictionary of parameter names and their values.

        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.value
        {'a': 1.0, 'b': 2.0}
        """
        return {key: value.value for key, value in self.items()}
    
    @property
    def lb(self) -> Dict[str, float]:
        """
        Get the lower bounds of all parameters.
        
        Returns
        -------
        dict
            A dictionary of parameter names and their lower bounds.
            
        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.lb
        {'a': -inf, 'b': -inf}
        """
        return {key: value.lb for key, value in self.items()}
    
    @property
    def ub(self) -> Dict[str, float]:
        """
        Get the upper bounds of all parameters.
        
        Returns
        -------
        dict
            A dictionary of parameter names and their upper bounds.
            
        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.ub
        {'a': inf, 'b': inf}
        """
        return {key: value.ub for key, value in self.items()}
    
    def set_value(self, **kwargs: float) -> Self:
        """
        Sets the values of multiple parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their new values.

        Returns
        -------
        ParameterDict
            The updated ParameterDict instance.

        Raises
        ------
        KeyError
            If a parameter name does not exist in the dictionary.
        """
        for key, value in kwargs.items():
            if key not in self:
                raise KeyError(f"Parameter '{key}' does not exist")
            self[key].assign_value(value)
        return self

    def set_lb(self, **kwargs: float) -> Self:
        """
        Sets the lower bounds of multiple parameters.
        
        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their new lower bounds.
            
        Returns
        -------
        ParameterDict
            The updated ParameterDict instance.
            
        Raises
        ------
        KeyError
            If a parameter name does not exist in the dictionary.
            
        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.set_lb(a=0.0, b=1.0)
        """
        for key, value in kwargs.items():
            if key not in self:
                raise KeyError(f"Parameter '{key}' does not exist")
            if not np.isfinite(value) and value != -np.inf:
                self[key].lb = -np.inf
            else:
                self[key].lb = value
        return self
    
    def set_ub(self, **kwargs) -> Self:
        """
        Sets the upper bounds of multiple parameters.
        
        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their new upper bounds.
            
        Returns
        -------
        ParameterDict
            The updated ParameterDict instance.
            
        Raises
        ------
        KeyError
            If a parameter name does not exist in the dictionary.
        ValueError
            If an upper bound is less than the corresponding lower bound.
            
        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.set_ub(a=2.0, b=3.0)
        """
        for key, value in kwargs.items():
            if key not in self:
                raise KeyError(f"Parameter '{key}' does not exist")
            if self[key].lb > value:
                raise ValueError(f"Upper bound ({value}) for '{key}' cannot be less than lower bound ({self[key].lb})")
            if not np.isfinite(value) and value != np.inf:
                self[key].ub = np.inf
            else:
                self[key].ub = value
        return self
    
    
    def scipy_bounds(self) -> optimize.Bounds:
        """
        Get the bounds in a format suitable for scipy.optimize.
        
        Returns
        -------
        scipy.optimize.Bounds
            A Bounds object containing the lower and upper bounds of all parameters.
            
        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.set_lb(a=0.0, b=1.0)
        >>> params.set_ub(a=2.0, b=3.0)
        >>> bounds = params.scipy_bounds()
        """
        param_keys = list(self.keys())
        return optimize.Bounds(
            lb=[self[key].lb for key in param_keys],
            ub=[self[key].ub for key in param_keys],
        )
    
    def values_list(self) -> list[Parameter]:
        """
        Get parameter values as a list, useful for optimization functions.
        
        Returns
        -------
        list
            A list of parameter values in the order of keys.
            
        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.values_list()
        [1.0, 2.0]
        """
        return list(self.values())

    def copy(self) -> Self:
        """Create a shallow copy of the parameter dictionary."""
        from copy import copy
        return copy(self)

    def deepcopy(self) -> Self:
        """Create a deep copy of the parameter dictionary."""
        from copy import deepcopy
        return deepcopy(self)
    
    def available_keys(self):
        """
        Get all available parameter keys, including derived and info keys.
        """
        return self.keys() 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()[1:-1]})"

class ConstrainedParameterDict(ParameterDict):
    """
    A dictionary-like class that extends ParameterDict with parameter constraint capabilities.
    
    This class allows defining equality constraints between parameters, where some parameters
    are derived from others according to specified functions.
    
    Attributes
    ----------
    _equal_constraints : dict
        Dictionary mapping parameter names to constraint functions
    _constraints_parameters : dict
        Dictionary storing original Parameter objects before they became constrained
    _parameter_names : list
        List of all parameter names including constrained ones
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a new ConstrainedParameterDict.
        
        Parameters
        ----------
        *args : dict or iterable
            Dictionary or iterable of key-value pairs to initialize with.
        **kwargs : dict
            Additional key-value pairs to initialize with.
        """
        self._equal_constraints = {}
        self._constraints_parameters = {}
        self._parameter_names = []
        super().__init__(*args, **kwargs)
        self._parameter_names = list(self.keys())
        
    def add_equal_constraints(self, **kwargs) -> None:
        """
        Add equality constraints to parameters.
        
        Parameters
        ----------
        **kwargs : dict
            A dictionary where keys are parameter names and values are constraint functions.
            Each constraint function takes a dictionary of parameters and returns the constrained value.
            
        Raises
        ------
        ValueError
            If any of the constraints is not callable.
            
        Examples
        --------
        >>> def constrain_c(params):
        ...     return params['a'] * params['b']  # c = a*b
        >>> params = ConstrainedParameterDict(a=2.0, b=3.0, c=6.0)
        >>> params.add_equal_constraints(c=constrain_c)
        """
        for i in kwargs:
            if callable(kwargs[i]):
                if i not in self.parameter_names():
                    raise ValueError(f"Parameter '{i}' is not the original parameter")
                if i not in self._equal_constraints:
                    # Store original parameter before removing it from direct parameters
                    self._constraints_parameters[i] = self.pop(i)
                    logger.debug(f"Added constraint on parameter '{i}', removed from direct parameters")
                self._equal_constraints[i] = kwargs[i]
                logger.debug(f"Set constraint function for parameter '{i}'")
            else:
                error_msg = f"The constraint of '{i}' must be callable, got {type(kwargs[i])}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def del_equal_constraints(self, name: str) -> None:
        """
        Remove an equality constraint from a parameter.
        
        Parameters
        ----------
        name : str
            The name of the parameter to remove constraint from.
            
        Raises
        ------
        ValueError
            If there is no constraint on the parameter or if the parameter 
            was not properly recorded in _constraints_parameters.
            
        Examples
        --------
        >>> params.del_equal_constraints('c')  # Remove constraint on c
        """
        if name not in self._parameter_names:
            raise ValueError(f"The parameter '{name}' was not found in original parameters")
        if name not in self._equal_constraints:
            raise ValueError(f"No constraints on the parameter '{name}'")
        if name not in self._constraints_parameters:
            raise ValueError(f"The parameter '{name}' was not recorded by _constraints_parameters")
        if name in self and name in self._constraints_parameters:
            raise ValueError(f"The parameter '{name}' was both a direct parameter and had a constraint")

        param = self._constraints_parameters.pop(name)
        self._equal_constraints.pop(name)
        
        idx = self._parameter_names.index(name)
        items = list(self.items())
        items.insert(idx, (name, param))
        self.clear()
        self.update(items)

            
    def __getitem__(self, key):
        """
        Get a parameter value, applying constraints if necessary.
        
        Parameters
        ----------
        key : str
            The parameter name to retrieve
            
        Returns
        -------
        Parameter or float
            The parameter value, possibly computed from a constraint
            
        Raises
        ------
        KeyError
            If the key is not found in parameters or constraints
        """
        if key in self._equal_constraints:
            return self._equal_constraints[key](self)
        return super().__getitem__(key)
            
    def decorate_func_constraints(self, function):
        """
        Decorates a function to apply equality constraints to parameters.

        If equality constraints are defined, this method modifies the input parameters
        of the function to include the constrained values based on the constraints.

        Parameters
        ----------
        function : callable
            The function to be decorated. It should accept a list of parameter values
            as its first argument.

        Returns
        -------
        callable
            A wrapped version of the input function that applies the equality constraints
            to the parameters before calling the original function.
            
        Examples
        --------
        >>> def objective(params, *args):
        ...     # params is a list of parameter values
        ...     return some_calculation(params)
        >>> params = ConstrainedParameterDict(a=1.0, b=2.0)
        >>> decorated_objective = params.decorate_func_constraints(objective)
        """
        if self._equal_constraints:
            @wraps(function)
            def wrapper(params, *args, **kwargs):
                input_dict = dict(zip(list(self.keys()), params))
                new_params = [
                    (
                        input_dict[i]
                        if i in input_dict
                        else self._equal_constraints[i](input_dict)
                    )
                    for i in self._parameter_names
                ]

                result = function(new_params, *args, **kwargs)
                return result

            return wrapper
        else:
            return function
        
    def constraint_keys(self):
        """
        Returns all constraint keys.
        """
        return self._equal_constraints.keys()

    def available_keys(self):
        """
        Returns all available keys, including constrained parameters.
        
        Returns
        -------
        set
            A set of all available parameter keys
        """
        return super().available_keys() | self.constraint_keys()
    
    def parameter_names(self):
        """
        Returns the original parameter names in the order they were added.
        """
        return list(self._parameter_names)

    def parameter_dict(self):
        """
        Returns a new ParameterDict with keys ordered according to parameter_names().
        """
        result = ParameterDict()
        for k in self.parameter_names():
            if k in self._equal_constraints:
                v = self[k]
                result[k] = Parameter(v, lb=v, ub=v)
            else:
                result[k] = self[k]
        return result
    

    def update(self, *args, **kwargs: float) -> None:
        """
        Update the dictionary with another dict, ParameterDict, or ConstrainedParameterDict.

        Parameters
        ----------
        other : dict, ParameterDict, or ConstrainedParameterDict
            The object to update from.
        """
        for other in args:
            if isinstance(other, ConstrainedParameterDict):
                # First update the base dict
                super().update(other)
                # Merge constraints and related attributes
                self._equal_constraints.update(other._equal_constraints)
                self._constraints_parameters.update(other._constraints_parameters)
                # Merge parameter names, preserving order and uniqueness
                for name in other._parameter_names:
                    if name not in self._parameter_names:
                        self._parameter_names.append(name)
            else:
                # dict or ParameterDict
                super().update(other)
        if kwargs:
            super().update(kwargs)
            
        for i in self:
            if i not in self._parameter_names:
                self._parameter_names.append(i)
            
class RichParameterDict(ParameterDict):
    """
    A ParameterDict with support for derived parameters and additional info.

    Attributes
    ----------
    _derived : dict
        Functions that compute derived parameters.
    _info : dict
        Additional information associated with the parameters.

    Methods
    -------
    add_derived(name, func)
        Add a derived parameter function.
    add_info(**kwargs)
        Add additional info.
    get_derived(name)
        Get the value of a derived parameter.
    get_info(name)
        Get info by key.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._derived = {}
        self._info = {}

    def add_derived(self, name: str, func):
        """Add a derived parameter function."""
        self._derived[name] = func

    def add_info(self, **kwargs):
        """Add additional info."""
        self._info.update(kwargs)

    def get_derived(self, name: str):
        """Get the value of a derived parameter."""
        if name in self._derived:
            return self._derived[name](self)
        raise KeyError(f"Derived parameter '{name}' not found.")

    def get_info(self, name: str):
        """Get info by key."""
        return self._info.get(name)

    def derived_keys(self):
        """Get all derived parameter keys."""
        return self._derived.keys()
    
    def info_keys(self):
        """Get all info keys."""
        return self._info.keys()
    
    def available_keys(self):
        """
        Get all available parameter keys, including derived and info keys.
        """
        return super().available_keys() | self.derived_keys() | self.info_keys()

    def __getitem__(self, key: str):
        if key in self:
            return super().__getitem__(key)
        if key in self._derived:
            return self.get_derived(key)
        if key in self._info:
            return self.get_info(key)
        raise KeyError(key)
    
    def update(self, *args, **kwargs: float):
        """
        Update the dictionary with another dict, ParameterDict, or RichParameterDict.

        Parameters
        ----------
        other : dict, ParameterDict, or RichParameterDict
            The object to update from.
        """
        for other in args:
            if isinstance(other, RichParameterDict):
                # First update the base dict
                super().update(other)
                # Merge constraints and related attributes
                self._derived.update(other._derived)
                self._info.update(other._info)
            else:
                # dict or ParameterDict
                super().update(other)
        if kwargs:
            super().update(kwargs)

class NewParameters(ConstrainedParameterDict, RichParameterDict):
    """
    A class that combines ConstrainedParameterDict and RichParameterDict.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def new(self,*args,**kwargs) :
        """
        Create a new instance of the Parameters class.
        """
        new = self.deepcopy()
        if args:
            params = dict(zip(self.keys(), *args))
            new.set_value(**params)
        if kwargs:
            new.set_value(**kwargs)
        return new
    @property
    def structure_parameters(self):
        return self.parameter_dict()
    
    def __add__(self, other: Dict | "NewParameters") -> "NewParameters":
        h1 = self.deepcopy()
        h1.update(other)
        return h1

class Parameters:
    """
    A class representing a collection of parameters with bounds and derived values.

    This class is used to manage a set of parameters, each of which is an instance of the `Parameter` class.
    It also supports derived parameters and additional information associated with the parameters.

    Attributes
    ----------
    __parameters : dict
        A dictionary of `Parameter` instances.
    _derived : dict
        A dictionary of derived parameters, where keys are function names and values are functions.
    _info : dict
        A dictionary of additional information associated with the parameters.

    Methods
    -------
    __init__(**kwargs)
        Initializes the Parameters class with a set of parameters.
    new(*args, **kwargs)
        Creates a new instance of Parameters with updated values and bounds.
    get_rounded_values_dict(n=3)
        Rounds the parameter values to `n` decimal places.
    available_keys()
        Returns a set of all available keys, including parameter keys, derived keys, and info keys.
    keys()
        Returns the keys of the parameters.
    derived(f)
        Adds a derived parameter function to the collection.
    __len__()
        Returns the number of parameters.
    __contains__(key)
        Checks if a key is in the parameters.
    __repr__()
        Returns a string representation of the Parameters instance.
    __setitem__(key, value)
        Sets the value of a parameter.
    __getitem__(k)
        Gets the value of a parameter or derived parameter.
    __copy__()
        Creates a shallow copy of the Parameters instance.
    update(other)
        Updates the parameters with values from another Parameters instance or dictionary.
    set_value(*args, **kwargs)
        Sets the values of multiple parameters.
    set_ub(**kwargs)
        Sets the upper bounds of multiple parameters.
    set_lb(**kwargs)
        Sets the lower bounds of multiple parameters.
    lb
        Property getter for the lower bounds of all parameters.
    ub
        Property getter for the upper bounds of all parameters.
    __add__(other)
        Merges two Parameters instances or a Parameters instance with a dictionary.
    scipy_bounds
        Property getter for the bounds in a format suitable for scipy.optimize.
    add_info(**kwargs)
        Adds additional information to the Parameters instance.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Parameters class with a set of parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their initial values.
        """
        self.__parameters = dict(
            **{key: Parameter(value) for key, value in kwargs.items()}
        )
        self._derived = {}
        self._info = {}

        self.__equal_constraints = {}
        self.__constraints_parameters = {}

        self._parameter_names = list(self.__parameters.keys())

    def new(self, *args, **kwargs):
        """
        Creates a new instance of Parameters with updated values and bounds.

        Parameters
        ----------
        *args : list
            A list of values to set for the parameters, in the order of the keys.
        **kwargs : dict
            A dictionary of parameter names and their new values.

        Returns
        -------
        Parameters
            A new instance of Parameters with updated values and bounds.
        """
        new = Parameters(**self)

        new._derived = self._derived.copy()
        new._info = self._info.copy()
        new.__equal_constraints = self.__equal_constraints.copy()

        new._parameter_names = self._parameter_names.copy()
        new.set_lb(**self.lb)
        new.set_ub(**self.ub)
        if args:
            params = dict(zip(self.keys(), *args))
            new.set_value(**params)
        if kwargs:
            new.set_value(**kwargs)

        return new

    def get_rounded_values_dict(self, n: int = 3) -> dict:
        """
        Rounds the parameter values to `n` decimal places.

        Parameters
        ----------
        n : int, optional
            The number of decimal places to round to. Default is 3.

        Returns
        -------
        dict
            A dictionary of parameter names and their rounded values.

        Raises
        ------
        ValueError
            If `n` is negative.
        """
        if n < 0:
            logger.error("Negative decimal places are not allowed: n=%d", n)
            raise ValueError("Decimal places must be non-negative.")
        try:
            return {i: truncate(self.__parameters[i], n) for i in self.__parameters}
        except Exception as e:
            logger.error("Failed to truncate dictionary: %s", e, exc_info=True)
            raise

    def available_keys(self) -> KeysView[str]:
        """
        Returns a set of all available keys, including parameter keys, derived keys, and info keys.

        Returns
        -------
        set
            A set of all available keys.
        """
        return (
            self.keys()
            | self.__equal_constraints.keys()
            | self._derived.keys()
            | self._info.keys()
        )

    def keys(self) -> KeysView[str]:
        """
        Returns the keys of the parameters.

        Returns
        -------
        dict_keys
            The keys of the parameters.
        """
        return self.__parameters.keys()

    def derived(self, f):
        """
        Adds a derived parameter function to the collection.

        Parameters
        ----------
        f : function
            A function that computes a derived parameter.
        """
        self._derived[f.__name__] = f

    def __len__(self):
        """
        Returns the number of parameters.

        Returns
        -------
        int
            The number of parameters.
        """
        return len(self.__parameters)

    def __contains__(self, key):
        """
        Checks if a key is in the parameters.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        bool
            True if the key is in the parameters, False otherwise.
        """
        return key in self.__parameters

    def __repr__(self):
        """
        Returns a string representation of the Parameters instance.

        Returns
        -------
        str
            A string representation of the Parameters instance.
        """
        dict_repr = repr(self.__parameters)

        return "Parameters( " + dict_repr[1:-1] + " )"

    def __setitem__(self, key, value):
        """
        Sets the value of a parameter.

        Parameters
        ----------
        key : str
            The name of the parameter.
        value : float or Parameter
            The new value of the parameter.

        Raises
        ------
        KeyError
            If the key is not in the parameters.
        """
        if key in self.__parameters:
            if isinstance(value, Parameter):
                self.__parameters[key] = value
                return
            else:
                self.__parameters[key] = self.__parameters[key].assign_value(value)
                return
        raise KeyError(f"Only Parameter: {list(self.keys())} can be set values")

    def __getitem__(self, k):
        """
        Gets the value of a parameter or derived parameter.

        Parameters
        ----------
        k : str
            The name of the parameter or derived parameter.

        Returns
        -------
        float or any
            The value of the parameter or derived parameter.

        Raises
        ------
        KeyError
            If the key is not found in the parameters, derived parameters, or info.
        """
        if k in self.__parameters:
            return self.__parameters[k]
        elif k in self.__equal_constraints:
            return self.__equal_constraints[k](self.__parameters)
        elif k in self._derived:
            return self._derived[k](self)
        elif k in self._info:
            return self._info[k]
        else:
            raise KeyError(k)

    def __copy__(self):
        """
        Creates a shallow copy of the Parameters instance.

        Returns
        -------
        Parameters
            A shallow copy of the Parameters instance.
        """
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        return inst

    def update(self, other):
        """
        Updates the parameters with values from another Parameters instance or dictionary.

        Parameters
        ----------
        other : Parameters or dict
            Another Parameters instance or a dictionary of parameter names and values.

        Returns
        -------
        None
        """
        if isinstance(other, Parameters):
            return self.__parameters.update(other)
        else:
            for i in other:
                if i in self.__parameters:
                    self.__parameters[i] = self.__parameters[i].assign_value(other[i])
                else:
                    self.__parameters[i] = Parameter(other[i])
                    self._parameter_names.append(i)
            return

    def set_value(self, *args, **kwargs):
        """
        Sets the values of multiple parameters.

        Parameters
        ----------
        *args : list
            A list of values to set for the parameters, in the order of the keys.
        **kwargs : dict
            A dictionary of parameter names and their new values.

        Returns
        -------
        Parameters
            The updated Parameters instance.
        """
        if args:
            params = dict(zip(self.__parameters.keys(), *args))
            for i in params:
                if i not in self.__parameters:
                    logger.warning(
                        f"Assigning value failed. {i} is not a parameter name"
                    )
                else:
                    self.__parameters[i] = self.__parameters[i].assign_value(params[i])

        for i in kwargs:
            if i not in self.__parameters:
                logger.warning(f"Assigning value failed. {i} is not a parameter name")
            else:
                self.__parameters[i] = self.__parameters[i].assign_value(kwargs[i])

        return self

    def set_ub(self, **kwargs) -> 'Parameters':
        """
        Sets the upper bounds of multiple parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their new upper bounds.

        Returns
        -------
        Parameters
            The updated Parameters instance.
            
        Examples
        --------
        >>> params = Parameters(a=1.0, b=2.0)
        >>> params.set_ub(a=2.0, b=3.0)
        Parameters( a=1.0, b=2.0 )  # with upper bounds set
        """
        try:
            for i in kwargs:

                if i not in self.__parameters:
                    logger.warning(f"Setting upper bound failed. '{i}' is not a parameter name.")

                elif self.__parameters[i].lb > kwargs[i]:
                    msg = f"Upper bound ({kwargs[i]}) for '{i}' cannot be less than lower bound ({self.__parameters[i].lb})"
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    # Validate the bound value
                    if not np.isfinite(kwargs[i]) and kwargs[i] != np.inf:
                        logger.warning(f"Non-finite upper bound for '{i}': {kwargs[i]}, using inf")
                        self.__parameters[i].ub = np.inf
                    else:
                        self.__parameters[i].ub = kwargs[i]
                        logger.debug(f"Set upper bound for '{i}' to {kwargs[i]}")
            return self
        except Exception as e:
            logger.error(f"Error setting upper bounds: {e}", exc_info=True)
            raise

    def set_lb(self, **kwargs) -> 'Parameters':
        """
        Sets the lower bounds of multiple parameters.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameter names and their new lower bounds.

        Returns
        -------
        Parameters
            The updated Parameters instance.
            
        Examples
        --------
        >>> params = Parameters(a=1.0, b=2.0)
        >>> params.set_lb(a=0.5, b=1.0)
        Parameters( a=1.0, b=2.0 )  # with lower bounds set
        """
        try:
            for i in kwargs:
                if i not in self.__parameters:
                    logger.warning(
                        f"Setting lower bound failed. '{i}' is not a parameter name."
                    )
                else:
                    # Validate the bound value
                    if not np.isfinite(kwargs[i]) and kwargs[i] != -np.inf:
                        logger.warning(f"Non-finite lower bound for '{i}': {kwargs[i]}, using -inf")
                        self.__parameters[i].lb = -np.inf
                    else:
                        self.__parameters[i].lb = kwargs[i]
                        logger.debug(f"Set lower bound for '{i}' to {kwargs[i]}")
            return self
        except Exception as e:
            logger.error(f"Error setting lower bounds: {e}", exc_info=True)
            raise

    @property
    def lb(self):
        """
        Get the lower bounds of all parameters.

        Returns
        -------
        dict
            A dictionary of parameter names and their lower bounds.
        """
        return {i: j.lb for i, j in self.__parameters.items()}

    @property
    def ub(self):
        """
        Get the upper bounds of all parameters.

        Returns
        -------
        dict
            A dictionary of parameter names and their upper bounds.
        """
        return {i: j.ub for i, j in self.__parameters.items()}

    def __add__(self, other):
        """
        Merges two Parameters instances or a Parameters instance with a dictionary.

        Parameters
        ----------
        other : Parameters or dict
            Another Parameters instance or a dictionary of parameter names and values.

        Returns
        -------
        Parameters
            The merged Parameters instance.

        Raises
        ------
        TypeError
            If `other` is not a dictionary or Parameters instance.
        """
        if isinstance(other, Parameters):
            logger.debug(f"merge {self} and {other}")

            h1 = copy.copy(self)
            h2 = copy.copy(other)
            h1.update(h2)
            h1._derived.update(h2._derived)
            h1._info.update(h2._info)
            h1.__equal_constraints.update(h2.__equal_constraints)
            h1._parameter_names = h1._parameter_names + h2._parameter_names
            return h1
        if isinstance(other, dict):
            logger.debug(f"merge {self} and {other}")

            h1 = copy.copy(self)
            h1.update(other)
            h1._parameter_names = h1._parameter_names + list(other.keys())
            return h1
        logger.error(f"{other} is not a dict")
        raise TypeError("Must be dict")

    @property
    def scipy_bounds(self):
        """
        Get the bounds in a format suitable for scipy.optimize.

        Returns
        -------
        scipy.optimize.Bounds
            A Bounds object containing the lower and upper bounds of all parameters.
        """
        return optimize.Bounds(
            lb=[i.lb for i in self.__parameters.values()],
            ub=[i.ub for i in self.__parameters.values()],
        )

    def add_info(self, **kwargs):
        """
        Adds additional information to the Parameters instance.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of additional information to add to the Parameters instance.
        """
        self._info.update(kwargs)

    @property
    def structure_parameters(self) -> dict:
        """
        Get a dictionary of the parameters needed for structure initialization.
        
        Returns
        -------
        dict
            A dictionary containing the parameter names and values needed for structure initialization.
            
        Notes
        -----
        This is used to extract parameters that should be passed to Structure3D instances.
        """
        try:
            return {i: self[i] for i in self._parameter_names}
        except Exception as e:
            logger.error(f"Failed to retrieve structure parameters: {e}", exc_info=True)
            raise

    def add_equal_constraints(self, **kwargs) -> None:
        """
        Add equality constraints to parameters.
        
        Parameters
        ----------
        **kwargs : dict
            A dictionary where keys are parameter names and values are constraint functions.
            Each constraint function takes a dictionary of parameters and returns the constrained value.
            
        Raises
        ------
        ValueError
            If any of the constraints is not callable.
            
        Examples
        --------
        >>> def constrain_c(params):
        ...     return params['a'] * params['b']  # c = a*b
        >>> params = Parameters(a=2.0, b=3.0, c=6.0)
        >>> params.add_equal_constraints(c=constrain_c)
        """
        for i in kwargs:
            if callable(kwargs[i]):
                if i not in self.__equal_constraints:
                    # Store original parameter before removing it from __parameters
                    self.__constraints_parameters[i] = self.__parameters.pop(i)
                    logger.debug(f"Added constraint on parameter '{i}', removed from direct parameters")
                self.__equal_constraints[i] = kwargs[i]
                logger.debug(f"Set constraint function for parameter '{i}'")
            else:
                error_msg = f"The constraint of '{i}' must be callable, got {type(kwargs[i])}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def del_equal_constraints(self, name: str) -> None:
        """
        Remove an equality constraint from a parameter.
        
        Parameters
        ----------
        name : str
            The name of the parameter to remove constraint from.
            
        Raises
        ------
        ValueError
            If there is no constraint on the parameter or if the parameter 
            was not properly recorded in __constraints_parameters.
            
        Examples
        --------
        >>> params.del_equal_constraints('c')  # Remove constraint on c
        """
        if name in self.__equal_constraints:
            if name in self.__constraints_parameters:
                # Restore the parameter to __parameters
                self.__parameters[name] = self.__constraints_parameters.pop(name)
                self.__equal_constraints.pop(name)
                logger.debug(f"Removed constraint on parameter '{name}', restored to direct parameters")

                # Retain the parameters order
                ordername = list(
                    filter(lambda x: x in self.__parameters, self._parameter_names)
                )
                self.__parameters = {i: self.__parameters[i] for i in ordername}
            else:
                error_msg = f"The parameter '{name}' was not recorded by __constraints_parameters"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"No constraints on the parameter '{name}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def decorate_func_constraints(self, function):
        """
        Decorates a function to apply equality constraints to parameters.

        If equality constraints are defined, this method modifies the input parameters
        of the function to include the constrained values based on the constraints.

        Parameters
        ----------
        function : callable
            The function to be decorated. It should accept a list of parameter values
            as its first argument.

        Returns
        -------
        callable
            A wrapped version of the input function that applies the equality constraints
            to the parameters before calling the original function.
        """

        if self.__equal_constraints:
            wraps(function)

            def wrapper(params, *args, **kwargs):
                input_dict = dict(zip(list(self.keys()), params))
                new_params = [
                    (
                        input_dict[i]
                        if i in input_dict
                        else self.__equal_constraints[i](input_dict)
                    )
                    for i in self._parameter_names
                ]

                result = function(new_params, *args, **kwargs)
                return result

            return wrapper
        else:
            return function
