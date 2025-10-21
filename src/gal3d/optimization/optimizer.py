"""
Base classes and utilities for optimization algorithms.

"""
import logging
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy._lib._util import _dict_formatter, _RichResult
from scipy.optimize import Bounds

from gal3d.optimization.parameter import Parameter, ParameterDict, Parameters
from gal3d.plugin import PluginBase, PluginManager

__all__ = ["Optimizer", "OptimizerBase", "OptimizeResult"]

logger = logging.getLogger("gal3d.optimization.optimizer")




class OptimizeResult(_RichResult):
    """
    Representation of the result of an optimization problem.


    Attributes
    ----------
    params (x): ParameterDict
        The optimal results
    fun: float | ndarray
        The value of the objective function at the optimal parameters.
    start_fun: float | ndarray
        The value of the objective function at the start parameters.
    start_params (x0): ndarray
        The starting parameters
    cost: float
        Value of the cost function at the solution.
    algorithm: str
        The algorithm used for the optimization.
    jac, hess: ndarray
        Values of objective function's Jacobian and its Hessian at `params` (if
        available). The Hessian may be an approximation, see the documentation
        of the function in question.
    hess_inv: object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    n_fun_evals (nfev): int
        Number of the objective function evaluations.
    n_jac_evals (njev): int
        Number of derivative evaluations.
    n_hess_evals (nhev): int
        Number of evaluations of the Hessian functions.
    n_iterations (nit): int
        Number of iterations until termination.
    max_constraint_violation (maxcv): float
        The maximum constraint violation.
    grad: ndarray
        Gradient of the cost function at the solution.
    optimality: float
        First-order optimality measure.
    active_mask: ndarray of int
        Each component shows whether a corresponding constraint is active (whether a variable is at the bound):
        - 0: a constraint is not active.
        - -1: a lower bound is active.
        - 1: an upper bound is active.
    algorithm_output: dict
        Additional algorithm specific information.
    """
    _order_keys: list[str] = ["message", "success", "status", "cost", "fun", "params",
                    "col_ind", "n_iterations", "lower", "upper", "eqlin", "ineqlin",
                    "converged", "flag", "function_calls", "iterations",
                    "root"]

    # omit_keys
    def __repr__(self):
        order_keys = ["message", "success", "status", "fun", "funl", "x", "xl",
                      "col_ind", "nit", "lower", "upper", "eqlin", "ineqlin",
                      "converged", "flag", "function_calls", "iterations",
                      "root"]
        order_keys = getattr(self, "_order_keys", order_keys)
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {"slack", "con", "crossover_nit", "_order_keys","call_kws"}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"
class OptimizerBase(PluginBase):
    """
    Abstract base class for implementing optimization algorithms.
    Subclasses must implement the fitting method and define available algorithms.

    Attributes
    ----------
    algo_name (str):
        Name of the optimization algorithm.
    algo_options (dict):
        Options specific to the algorithm.
    kwargs (dict):
        Additional keyword arguments.
    """

    def __init__(self, algorithm: str, algo_options: dict | None = None):
        """
        Initializes an optimizer with a specified algorithm and options.

        Parameters
        ----------
        algorithm (str):
            The name of the optimization algorithm.
        algo_options (dict, optional):
            A dictionary of options specific to the algorithm. Defaults to None.

        Raises
        ------
            ValueError: If the specified algorithm is not valid.
        """

        if not self.has_algorithm(algorithm):
            raise ValueError(f"{algorithm} is not a valid algorithm name.\n")

        self.algo_name = algorithm

        self.algo_options = algo_options or {}
        self.kwargs: dict = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register the subclass as an optimizer plugin.
        """
        super().__init_subclass__(**kwargs)
        Optimizer.register(cls)

    def fit(self,
        fun: Callable,
        params: Parameters,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        **kwargs: Any,
        ) -> OptimizeResult:
        """
        Fit the model to the data.

        Parameters
        ----------
        fun : callable
            The objective function to minimize.
        params : Parameters
            The initial parameters for the optimization.
        func_args : tuple, optional
            Additional arguments to pass to the objective function (default is None).
        func_kwargs : dict, optional
            Additional keyword arguments to pass to the objective function (default is None).
        **kwargs : additional keyword arguments
            Additional options for the fitting algorithm.

        Returns
        -------
        result : OptimizeResult
            The result of the fitting.
        """
        func = params.decorate_func_constraints(fun)
        bounds = params.scipy_bounds
        x0 = params.values_list()
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}

        return self.fitting(
            fun=func,
            x0=x0,
            bounds=bounds,
            func_args=func_args,
            func_kwargs=func_kwargs,
            param_names=list(params.parameter_keys()),
            **kwargs,
        )

    @abstractmethod
    def fitting(
        self,
        fun: Callable,
        x0: ArrayLike,
        bounds: Bounds,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        param_names: list[str] | None = None,
        **kwargs: Any,
    ) -> OptimizeResult:
        """
        Perform the fitting process for input functions (minimization).

        This method must be implemented by subclasses.

        Parameters
        ----------
        fun : callable
            The objective function to minimize.
        x0 : array-like
            Initial guess for the parameters.
        bounds : sequence
            Bounds for the parameters.
        func_args : tuple, optional
            Additional arguments to pass to the objective function (default is None).
        func_kwargs : dict, optional
            Additional keyword arguments to pass to the objective function (default is None).
        **kwargs : additional keyword arguments
            Additional options for the fitting algorithm.

        Returns
        -------
        result : OptimizeResult
            The result of the fitting.
        """

    def create_params(
        self,
        param_values: Sequence[float],
        param_names: list[str] | None = None,
        param_lbs: Sequence[float | None] | None = None,
        param_ubs: Sequence[float | None] | None = None,
        param_errors: Sequence[float | None] | None = None
        ) -> ParameterDict:
        """
        Create a ParameterDict from the given parameter information.

        Parameters
        ----------
        param_values : Sequence[float]
            List of parameter values.
        param_names : list[str], optional
            List of parameter names. If None, default names will be generated.
        param_lbs : Sequence[float | None], optional
            List of lower bounds for the parameters. If None, no bounds will be set.
        param_ubs : Sequence[float | None], optional
            List of upper bounds for the parameters. If None, no bounds will be set.
        param_errors : Sequence[float | None], optional
            List of parameter errors. If None, no errors will be set.

        Returns
        -------
        ParameterDict
            A ParameterDict containing the created parameters.
        """
        n = len(param_values)
        param_names = [f"param_{i}" for i in range(n)] if param_names is None else param_names
        param_lbs = [None] * n if param_lbs is None else param_lbs
        param_ubs = [None] * n if param_ubs is None else param_ubs
        param_errors = [None] * n if param_errors is None else param_errors

        params = ParameterDict()
        for name, value, lb, ub, err in zip(param_names, param_values, param_lbs, param_ubs, param_errors, strict=False):
            params[name] = Parameter(value, lb=lb, ub=ub, err=err)
        return params

    def set_options(self, **kwargs: Any) -> None:
        """
        Update the algorithm options.

        Parameters
        ----------
        **kwargs : keyword arguments
            Options to update in the algorithm.
        """
        self.algo_options.update(**kwargs)

    def has_algorithm(self, algorithm: str) -> bool:
        """
        Check if the given algorithm is available.
        """
        if algorithm in self.available_algorithm():
            return True
        return False

    @classmethod
    @abstractmethod
    def available_algorithm(cls) -> list[str]:
        """
        List of available algorithms.

        Returns
        -------
        List[str]
            A list of available algorithm names.
        """


class Optimizer(PluginManager[OptimizerBase]):
    """
    Factory class for accessing registered optimizer plugins.
    """
    _plugins = {}
    _plugin_module = "gal3d.optimization.optimizer_plugins"
    _base_class = OptimizerBase
