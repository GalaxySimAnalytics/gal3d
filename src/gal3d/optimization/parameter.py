"""
Parameter classes for optimization algorithms.

"""
import logging
import math
import re
from collections.abc import Callable, Iterable, KeysView, Mapping, Sequence
from functools import wraps
from typing import Any, TypeVar, Union

import numpy as np
from scipy.optimize import Bounds

from .util import truncate

# from optimagic import Bounds

_ParamDict = TypeVar("_ParamDict", bound="ParameterDict")
_RichParamDict = TypeVar("_RichParamDict", bound="RichParameterDict")

logger = logging.getLogger("gal3d.optimization.parameter")


__all__ = ["Parameters","Parameter", "ParameterDict", "ConstrainedParameterDict", "RichParameterDict"]


def _fmt_num(v: float, nd: int = 3, latex: bool = False) -> str:
    """
    Format number for display.

    Parameters
    ----------
    v : float
    nd : int
        Decimal digits (for fixed / scientific formats).
    latex : bool
        If True, return LaTeX-suitable tokens (e.g. \\mathrm{NaN}, \\infty).

    Returns
    -------
    str
    """
    if math.isnan(v):
        return r"\mathrm{NaN}" if latex else "NaN"
    if math.isinf(v):
        if latex:
            return r"\infty" if v > 0 else r"-\infty"
        # plain text: use 'inf' / '-inf'
        return "inf" if v > 0 else "-inf"
    # use scientific notation for very large or very small nonzero values
    if v != 0 and (abs(v) >= 10 ** (nd + 1) or abs(v) < 10 ** (-nd)):
        return f"{v:.{nd}e}"
    return f"{v:.{nd}f}"

def _escape_name(name: str) -> str:
    return re.sub(r"_", r"\_", name)

def _fmt_bounds(lb: float, ub: float, nd: int = 3, latex: bool = False) -> str:
    lbs = _fmt_num(lb, nd, latex=latex)
    ubs = _fmt_num(ub, nd, latex=latex)
    return rf"[{lbs}, {ubs}]"

def _latex_param_line(param: "Parameter", nd: int = 3, hide_zero_err: bool = True) -> str:
    """
    Build a single parameter latex fragment:
    \\lgroup value±err, [lb, ub] \rgroup
    """
    val = _fmt_num(float(param), nd, latex=True)
    bounds = _fmt_bounds(param.lb, param.ub, nd, latex=True)
    if hide_zero_err and (math.isnan(param.err) or param.err == 0.0):
        return rf"{val} \in {bounds}"
    else:
        err = _fmt_num(param.err, nd, latex=True)
        return rf"{val} \pm {err} \in {bounds}"

class Parameter(float):
    """
    A class representing a parameter with lower and upper bounds.

    This class extends the built-in `float` type to include lower (`lb`) and upper (`ub`) bounds.
    It is used to define parameters that can be fitted within a specified range.

    Attributes
    ----------
    lb : float
        The lower bound of the parameter.
    ub : float
        The upper bound of the parameter.
    err : float
        The uncertainty of the parameter.

    Methods
    -------
    __new__(cls, value, **kwargs)
        Creates a new instance of the Parameter class.
    assign_value(value)
        Assigns a new value to the parameter while preserving the bounds.
    assign_bounds(lb, ub)
        Assigns new bounds to the parameter.
    """
    lb: float
    ub: float
    err: float

    __slots__ = ["lb", "ub", "err"]

    def __new__(cls, value: Union[float, "Parameter"], lb: float | None = None, ub: float | None = None, err: float | None = None) -> "Parameter":
        """
        Creates a new instance of the Parameter class.

        Parameters
        ----------
        value : float or Parameter
            The initial value of the parameter. If `value` is an instance of `Parameter`,
            the new instance will inherit the bounds from `value` unless overridden by `lb` or `ub`.
        lb : float, optional
            The lower bound of the parameter. Default is -inf.
        ub : float, optional
            The upper bound of the parameter. Default is inf.
        err : float, optional
            The uncertainty of the parameter. Default is 0.

        Returns
        -------
        Parameter
            A new instance of the Parameter class.
        """
        instance = float.__new__(cls, value)

        if isinstance(value, Parameter):
            object.__setattr__(instance, "lb", float(lb) if lb is not None else value.lb)
            object.__setattr__(instance, "ub", float(ub) if ub is not None else value.ub)
            object.__setattr__(instance, "err", float(err) if err is not None else value.err)
            return instance
        object.__setattr__(instance, "lb", float(lb) if lb is not None else -float("inf"))
        object.__setattr__(instance, "ub", float(ub) if ub is not None else float("inf"))
        object.__setattr__(instance, "err", float(err) if err is not None else 0.0)
        return instance

    def assign_value(self, value: float) -> "Parameter":
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
        inst = self.__class__.__new__(self.__class__, value, lb=self.lb, ub=self.ub, err=self.err)
        return inst

    def assign_bounds(self, lb: float, ub: float) -> "Parameter":
        """
        Assign new bounds to the parameter.

        Parameters
        ----------
        lb : float
            The new lower bound.
        ub : float
            The new upper bound.

        Returns
        -------
        Parameter
            The instance with updated bounds.

        Examples
        --------
        >>> param = Parameter(1.0)
        >>> param = param.assign_bounds(0.0, 2.0)
        >>> param
        1.0
        """
        self.lb = float(lb)
        self.ub = float(ub)
        return self

    def __setattr__(self, name: str, value: float) -> None:
        super().__setattr__(name, float(value))

    def __hash__(self):
        return hash((float(self), self.lb, self.ub, self.err))

    def __repr__(self) -> str:
        """Return the verbose diagnostic form (old repr style)."""
        val = _fmt_num(float(self))
        lb  = _fmt_num(self.lb)
        ub  = _fmt_num(self.ub)
        if math.isnan(self.err) or self.err == 0.0:
            return f"Parameter({val} ∈ [{lb}, {ub}])"
        return f"Parameter({val} ± {_fmt_num(self.err)} ∈ [{lb}, {ub}])"

    def to_latex(self, nd: int = 3, hide_zero_err: bool = True) -> str:
        """
        Return a single LaTeX fragment (no surrounding $) for this parameter.
        """
        return _latex_param_line(self, nd=nd, hide_zero_err=hide_zero_err)

    def _repr_latex_(self):
        """Return a KaTeX-safe single-line math representation."""
        return r"$" + self.to_latex() + r"$"

class ParameterDict(dict):
    """
    A dictionary-like class that enforces all values to be of Parameter type.

    This class extends the built-in `dict` type to ensure all values are instances
    of the `Parameter` class. When assigning a value that is not a `Parameter`,
    it automatically converts it to a `Parameter` instance.

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

    def __init__(self, *args: Mapping[str,float] | Sequence[tuple[str, float]] | _ParamDict, **kwargs: float):
        super().__init__()
        self.update(*args, **kwargs)


    def __setitem__(self, key: str, value: float | Parameter) -> None:
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
        elif key in self:
            param = super().__getitem__(key)
            value = Parameter(value, lb=param.lb, ub=param.ub, err=param.err)
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, Parameter(value))


    def update(self: _ParamDict, *args: Mapping[str,float] | Sequence[tuple[str, float]] | _ParamDict, **kwargs: float) -> None: # type: ignore[override]
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
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def get_parameter(self, name: str) -> Parameter:
        if name in self:
            return self[name]
        raise KeyError(f"Parameter '{name}' not found.")

    def parameter_keys(self):
        return self.keys()

    def clip_to_bounds(self: _ParamDict) -> _ParamDict:
        """
        Check each parameter value and clip it to its bounds if out of range.

        Returns
        -------
        ParameterDict
            The updated ParameterDict instance with values clipped to [lb, ub].
        """
        for key, param in self.items():
            lb = param.lb
            ub = param.ub
            if param < lb:
                self[key] = param.assign_value(lb)
            elif param > ub:
                self[key] = param.assign_value(ub)
        return self

    def get_rounded(self, n: int = 3, round_value: bool = True, round_bound: bool = False, only_value: bool = False) -> Union[dict[str, float], "ParameterDict"]:
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
        only_value : bool, optional
            If True, return a plain dict of rounded values; otherwise return ParameterDict.

        Returns
        -------
        dict or ParameterDict
            A new ParameterDict with rounded values and/or bounds.
        """
        if n < 0:
            raise ValueError("Decimal places must be non-negative.")
        if only_value:
            return {
                k: truncate(v, n) if round_value else v
                for k, v in self.items()
            }
        else:
            return ParameterDict({
                k: Parameter(
                    truncate(v, n) if round_value else v,
                    lb=truncate(v.lb, n) if round_bound else v.lb,
                    ub=truncate(v.ub, n) if round_bound else v.ub,
                    err = v.err
                )
                for k, v in self.items()
            })
    @property
    def value(self) -> dict[str, float]:
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
        return {key: float(value) for key, value in self.items()}

    @property
    def lb(self) -> dict[str, float]:
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
    def ub(self) -> dict[str, float]:
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

    @property
    def err(self) -> dict[str, float]:
        """
        Get the uncertainties of all parameters.

        Returns
        -------
        dict
            A dictionary of parameter names and their uncertainties.

        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.err
        {'a': 0.0, 'b': 0.0}
        """
        return {key: value.err for key, value in self.items()}

    def set_value(self: _ParamDict, *args: Iterable[float] | float, **kwargs: float) -> _ParamDict:
        """
        Sets the values of multiple parameters.

        Parameters
        ----------
        *args : float
            The new values for the parameters, in the same order as they were defined.
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
        flat_args = []
        for arg in args:
            if isinstance(arg, (list, tuple, np.ndarray)):
                flat_args.extend(np.array(arg).flatten().tolist())
            else:
                flat_args.append(arg)
        if flat_args:
            params = dict(zip(self.keys(), flat_args, strict=False))
            for key, value in params.items():
                if key not in self:
                    raise KeyError(f"Parameter '{key}' does not exist")
                self[key] = self[key].assign_value(value)
        for key, value in kwargs.items():
            if key not in self:
                raise KeyError(f"Parameter '{key}' does not exist")
            self[key] = self[key].assign_value(value)
        return self

    def set_lb(self, *, only_infs: bool = False, **kwargs: float) -> "ParameterDict":
        """
        Sets the lower bounds of multiple parameters.

        Parameters
        ----------
        only_infs : bool, optional
            If True, only set bounds for parameters with infinite lower bounds.
        **kwargs : dict
            A dictionary of parameter names and their new lower bounds.

        Returns
        -------
        ParameterDict
            The updated ParameterDict instance.

        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.set_lb(a=0.0, b=1.0)
        """
        keys = set(kwargs) & set(self.keys())
        for key in keys:
            if only_infs and math.isfinite(self[key].lb):
                continue
            value = kwargs[key]
            self[key].lb = value if math.isfinite(value) else -math.inf
        return self

    def set_ub(self, *, only_infs: bool = False, **kwargs: float) -> "ParameterDict":
        """
        Sets the upper bounds of multiple parameters.

        Parameters
        ----------
        only_infs : bool, optional
            If True, only set bounds for parameters with infinite upper bounds.
        **kwargs : dict
            A dictionary of parameter names and their new upper bounds.

        Returns
        -------
        ParameterDict
            The updated ParameterDict instance.

        Examples
        --------
        >>> params = ParameterDict(a=1.0, b=2.0)
        >>> params.set_ub(a=2.0, b=3.0)
        """
        keys = set(kwargs) & set(self.keys())
        for key in keys:
            if only_infs and math.isfinite(self[key].ub):
                continue
            value = kwargs[key]
            self[key].ub = value if math.isfinite(value) else math.inf
        return self

    @property
    def scipy_bounds(self) -> Bounds:
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
        >>> bounds = params.scipy_bounds
        """
        param_keys = list(self.keys())
        return Bounds(
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

    def copy(self: _ParamDict) -> _ParamDict:
        """Create a shallow copy of the parameter dictionary."""
        from copy import copy
        return copy(self)

    def deepcopy(self) -> "ParameterDict":
        """Create a deep copy of the parameter dictionary."""
        from copy import deepcopy
        return deepcopy(self)

    def available_keys(self) -> KeysView:
        """
        Get all available parameter keys, including derived and info keys.
        """
        return self.keys()

    def __repr__(self):
        if not self:
            return f"{self.__class__.__name__}()"
        name_w = max(len(k) for k in self) + 1
        lines = []
        for k, v in self.items():
            val  = _fmt_num(float(v))
            lb   = _fmt_num(v.lb)
            ub   = _fmt_num(v.ub)
            err_s = f" ± {_fmt_num(v.err)}" if not (math.isnan(v.err) or v.err == 0.0) else ""
            lines.append(f"  {k:<{name_w}} = {val}{err_s}  [{lb}, {ub}]")
        return f"{self.__class__.__name__}(\n" + "\n".join(lines) + "\n)"

    def to_latex_lines(self, nd: int = 3, hide_zero_err: bool = True) -> dict[str,str]:
        """
        Produce LaTeX lines (no $) for each direct parameter.
        """
        lines: dict[str,str] = {}
        for k, v in self.items():
            lines[k] = _latex_param_line(v, nd=nd, hide_zero_err=hide_zero_err)
        return lines

    def _repr_latex_(self):
        """
        Basic LaTeX for direct parameters only.
        """
        lines = self.to_latex_lines()
        if not lines:
            return r"$\text{" + self.__class__.__name__ + r"()}$"
        items = list(lines.items())
        rows = [rf"\textbf{{\text{{{self.__class__.__name__}}}}} & & \\"]
        for i, (k, v) in enumerate(items):
            row = rf"\text{{{_escape_name(k)}}} & {{:}} & {v}"
            if i < len(items) - 1:
                row += r" \\"
            rows.append(row)
        body = "\n".join(rows)
        return (
            "$\n"
            "\\begin{array}{r c l}\n"
            + body +
            "\n\\end{array}\n"
            "$"
        )

    def _ipython_key_completions_(self) -> list[str]:
        """Return a list of parameter names for IPython tab completion."""
        return list(self)

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
        self._equal_constraints: dict[str, Callable[[dict[str, float]], float]] = {}
        self._constraints_parameters: dict[str, Parameter] = {}
        self._parameter_names: list[str] = []
        super().__init__(*args, **kwargs)
        self._parameter_names = list(self.keys())

    def add_equal_constraints(self, **kwargs: Callable[[dict[str, float]], float]) -> None:
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
            If the parameter name is not in self.all_parameter_names (i.e., not an original parameter).

        Notes
        -----
        - When a constraint is added, the parameter is removed from self.keys() (the direct parameter set)
          and added to self.constraint_keys() (the set of constrained parameters).
        - All parameter names (including both constrained and unconstrained) remain in self.all_parameter_names.
        - You can always retrieve all parameters (including constrained ones) using self.all_parameter_dict,
          which is useful for structure initialization and other tasks.

        Examples
        --------
        >>> def constrain_c(params):
        ...     return params['a'] * params['b']  # c = a*b
        >>> params = ConstrainedParameterDict(a=2.0, b=3.0, c=6.0)
        >>> params.add_equal_constraints(c=constrain_c)
        """
        for i in kwargs:
            if callable(kwargs[i]):
                if i not in self.all_parameter_names:
                    raise ValueError(f"Parameter '{i}' is not the original parameter")
                if i not in self._equal_constraints:
                    # Store original parameter before removing it from direct parameters
                    self._constraints_parameters[i] = self.pop(i)
                    logger.debug("Added constraint on parameter '%s', removed from direct parameters", i)
                self._equal_constraints[i] = kwargs[i]
                logger.debug("Set constraint function for parameter '%s'", i)
            else:
                error_msg = f"The constraint of '{i}' must be callable, got {type(kwargs[i])}"
                logger.error(error_msg)
                raise ValueError(error_msg)

    def fix_parameters(self, **kwargs: float) -> None:
        """
        Fix one or more parameters to constant values (equality constraints).

        Parameters
        ----------
        **kwargs : dict
            Parameter names and their fixed values.

        Examples
        --------
        >>> params.fix_parameter(x=2, y=3)
        """
        for name, value in kwargs.items():
            self.add_equal_constraints(**{name: lambda params, v=value: v})

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
            If constraint or parameter is not found.

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

        # Rebuild base dict strictly following original order (_parameter_names)
        ordered_items: list[tuple[str, Parameter]] = []
        for k in self._parameter_names:
            if k == name:
                ordered_items.append((k, param))
            elif k in self:
                # keep the current (unconstrained) parameter object
                ordered_items.append((k, self.get_parameter(k)))

        self.clear()
        # reuse normal set to keep bounds/value semantics
        self.update(ordered_items)


    def __getitem__(self, key: str) -> Parameter | float:
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
            return self.get_constraint(key)
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
                input_dict = dict(zip(list(self.keys()), params, strict=False))
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

    def get_constraint(self, name: str) -> float:

        if name in self._equal_constraints:
            return float(self._equal_constraints[name](self))
        raise KeyError(f"Constraint '{name}' not found.")

    def constraint_keys(self) -> KeysView:
        """
        Returns all constraint keys.
        """
        return self._equal_constraints.keys()

    def available_keys(self) -> KeysView:
        """
        Returns all available keys, including constrained parameters.

        Returns
        -------
        set
            A set of all available parameter keys
        """
        return dict.fromkeys(
            list(super().available_keys())+list(self.constraint_keys())
            ).keys()

    @property
    def all_parameter_names(self):
        """
        Returns the original parameter names in the order they were added.
        """
        return list(self._parameter_names)

    @property
    def all_parameter_dict(self):
        """
        Get all parameters, including constrained ones.

        Returns
        -------
        ParameterDict
            All parameters.
        """
        result = ParameterDict()
        for k in self.all_parameter_names:
            if k in self._equal_constraints:
                v = self[k]
                result[k] = Parameter(v, lb=v, ub=v)
            else:
                result[k] = self[k]
        return result

    def __setitem__(self, key: str, value: float) -> None:
        if key not in self._parameter_names:
            self._parameter_names.append(key)
        return super().__setitem__(key, value)

    def update(self: "ConstrainedParameterDict", *args: Union["ConstrainedParameterDict",Mapping[str, float],Sequence[tuple[str, float]]], **kwargs: float) -> None: # type: ignore[override]
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

    def to_latex_lines(
        self,
        nd: int = 3,
        hide_zero_err: bool = True,
        include_constraints: bool = True
    ) -> dict[str,str]:
        lines: dict[str,str] = {}
        for k in self._parameter_names:
            if k in self._equal_constraints and include_constraints:
                val = self.get_constraint(k)
                lines[k] = rf"\lgroup {_fmt_num(val, nd)},\ \text{{constraint}} \rgroup"
            elif k in self:
                p: Parameter = self.get_parameter(k)
                lines[k] = _latex_param_line( p, nd=nd, hide_zero_err=hide_zero_err)
        return lines

    def _ipython_key_completions_(self) -> list[str]:
        return list(self) + list(self._equal_constraints)

class RichParameterDict(ParameterDict):
    """
    A ParameterDict with support for derived parameters and additional info.

    Attributes
    ----------
    _derived : dict
        Functions that compute derived parameters.
    _info : dict
        Additional information associated with the parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._derived = {}
        self._info = {}

    def add_derived(self, name_or_dict: str | dict[str, Callable], func: Callable | None = None) -> None:
        """
        Add a derived parameter function or a dict of derived functions.

        Parameters
        ----------
        name_or_dict : str or dict
            If str, must provide func. If dict, keys are names and values are functions.
        func : callable, optional
            The function for the derived parameter (if name_or_dict is str).

        Raises
        ------
        TypeError
            If arguments are invalid.
        """
        if isinstance(name_or_dict, str) and func is not None:
            self._derived[name_or_dict] = func
        elif isinstance(name_or_dict, dict):
            for name, funcs in name_or_dict.items():
                self._derived[name] = funcs
        else:
            raise TypeError("Invalid arguments for add_derived.")

    def derived(self, f: Callable) -> None:
        """Add a derived parameter function.

        Examples
        --------
        >>> param = RichParameterDict(a = 2, b = 1)
        >>> @param.derived
        ... def eps_ab(p: RichParameterDict):
        ...     return 1 - p["b"] / p["a"]
        >>> param['eps_ab']
        >>> 0.5
        """
        self.add_derived(f.__name__, f)

    @classmethod
    def with_derived(cls: type[_RichParamDict], derived_dict: dict[str, Callable[[_RichParamDict], float]],
                     *args: float | Mapping | tuple[str, float], **kwargs: float) -> _RichParamDict:
        """Create a new instance with derived parameters."""
        inst = cls(*args, **kwargs)
        inst._derived.update(derived_dict)
        return inst

    def add_info(self, **kwargs: Any) -> None:
        """Add additional info."""
        self._info.update(kwargs)

    def get_derived(self, name: str, **kwargs: Any) -> Any:
        """Get the value of a derived parameter."""
        if name in self._derived:
            return self._derived[name](self)
        if kwargs.get("default") is not None:
            return kwargs["default"]
        raise KeyError(f"Derived parameter '{name}' not found.")

    def get_info(self, name: str, **kwargs: Any) -> Any:
        """Get info by key."""
        if name in self._info:
            return self._info[name]
        if kwargs.get("default") is not None:
            return kwargs["default"]
        raise KeyError(f"Info '{name}' not found.")

    def derived_keys(self) -> KeysView:
        """Get all derived parameter keys."""
        return self._derived.keys()

    def info_keys(self) -> KeysView:
        """Get all info keys."""
        return self._info.keys()

    def available_keys(self) -> KeysView:
        """
        Get all available parameter keys, including derived and info keys.
        """
        return dict.fromkeys(
            list(super().available_keys()) + list(self.derived_keys()) + list(self.info_keys())
            ).keys()

    def __getitem__(self, key: str) -> Parameter | float | Any:
        if key in self:
            return super().__getitem__(key)
        if key in self._derived:
            return self.get_derived(key)
        if key in self._info:
            return self.get_info(key)
        raise KeyError(key)

    def update(self: "RichParameterDict", *args: Union["RichParameterDict", Mapping[str, float], Sequence[tuple[str, float]]], **kwargs: float) -> None: # type: ignore[override]
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

    def _ipython_key_completions_(self) -> list[str]:
        return list(self) + list(self._derived) + list(self._info)

class Parameters(RichParameterDict,ConstrainedParameterDict):
    """
    A class as a container for parameters with constraints and rich metadata.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def deepcopy(self) -> "Parameters":
        """Create a deep copy of the Parameters instance."""
        from copy import deepcopy
        return deepcopy(self)

    def new(self,*args: Iterable[float] | float,**kwargs: float) -> "Parameters":
        """
        Create a new instance with updated values.

        Parameters
        ----------
        *args, **kwargs
            Arguments for set_value.

        Returns
        -------
        Parameters
            New instance.
        """
        new = self.deepcopy()
        new.set_value(*args, **kwargs)
        return new
    @property
    def structure_parameters(self):
        """
        Get all parameters including constraints.

        Returns
        -------
        ParameterDict
            All parameters.
        """
        return self.all_parameter_dict

    def get_rounded_values_dict(self, n: int = 3) -> dict[str, float]:
        """
        Get rounded parameter values.

        Parameters
        ----------
        n : int, optional
            Number of decimal places. Default is 3.

        Returns
        -------
        dict
            Rounded parameter values.
        """
        return self.get_rounded(n, only_value=True)

    def __getitem__(self, key: str) -> Parameter | float | Any:
        try:
            return ConstrainedParameterDict.__getitem__(self,key)
        except KeyError:
            pass
        if key in self._derived:
            return self.get_derived(key)
        if key in self._info:
            return self.get_info(key)
        raise KeyError(key)

    def _ipython_key_completions_(self) -> list[str]:
        return list(self) + list(self._equal_constraints)+list(self._derived) + list(self._info)


    def __add__(self, other: Union[dict, "Parameters"]) -> "Parameters":
        h1 = self.deepcopy()
        h1.update(other)
        return h1
