import abc
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

from _typeshed import Incomplete
from numpy.typing import ArrayLike
from scipy._lib._util import _RichResult
from scipy.optimize import Bounds

from gal3d.optimization.optimizer_plugins.optimize_lmfit import OptimizerLMFit
from gal3d.optimization.optimizer_plugins.optimize_nlopt import OptimizerNLopt
from gal3d.optimization.optimizer_plugins.optimize_optimagic import OptimizerOptimagic
from gal3d.optimization.optimizer_plugins.optimize_scipy import OptimizerScipy
from gal3d.optimization.parameter import ParameterDict, Parameters
from gal3d.plugin import PluginBase, PluginManager

__all__ = ["Optimizer", "OptimizerBase", "OptimizeResult"]

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

class OptimizerBase(PluginBase, metaclass=abc.ABCMeta):
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
    algo_name: Incomplete
    algo_options: Incomplete
    kwargs: dict
    def __init__(self, algorithm: str, algo_options: dict | None = None) -> None:
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
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register the subclass as an optimizer plugin.
        """
    def fit(self, fun: Callable, params: Parameters, func_args: tuple | None = None, func_kwargs: dict | None = None, **kwargs: Any) -> OptimizeResult:
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
    @abstractmethod
    def fitting(self, fun: Callable, x0: ArrayLike, bounds: Bounds, func_args: tuple | None = None, func_kwargs: dict | None = None, param_names: list[str] | None = None, **kwargs: Any) -> OptimizeResult:
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
    def create_params(self, param_values: Sequence[float], param_names: list[str] | None = None, param_lbs: Sequence[float | None] | None = None, param_ubs: Sequence[float | None] | None = None, param_errors: Sequence[float | None] | None = None) -> ParameterDict:
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
    def set_options(self, **kwargs: Any) -> None:
        """
        Update the algorithm options.

        Parameters
        ----------
        **kwargs : keyword arguments
            Options to update in the algorithm.
        """
    def has_algorithm(self, algorithm: str) -> bool:
        """
        Check if the given algorithm is available.
        """
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

    @overload
    @classmethod
    def get_plugin(cls, name: Literal["OptimizerLMFit"]) -> type[OptimizerLMFit]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["OptimizerNLopt"]) -> type[OptimizerNLopt]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["OptimizerOptimagic"]) -> type[OptimizerOptimagic]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: Literal["OptimizerScipy"]) -> type[OptimizerScipy]: ...
    @overload
    @classmethod
    def get_plugin(cls, name: str) -> type[OptimizerBase]: ...
