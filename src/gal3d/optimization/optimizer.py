import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any

import numpy as np
from numpy.typing import NDArray

from gal3d.plugin import PluginBase, PluginManager

__all__ = ['Optimizer', 'OptimizerBase', 'OptimizeResult']

logger = logging.getLogger("gal3d.optimization.optimizer")




# from optimagic InternalOptimizeResult and OptimizeResult
@dataclass(frozen=True)
class OptimizeResult:
    """representation of the result of an optimization problem."""

    params: NDArray[np.float64]
    fun: float
    start_params: NDArray[np.float64]
    start_fun: float
    algorithm: str | None = None
    
    
    success: bool | None = None
    message: str | None = None
    status: int | None = None
    
    n_fun_evals: int | None = None
    n_jac_evals: int | None = None
    n_hess_evals: int | None = None
    n_iterations: int | None = None
    
    jac: NDArray[np.float64] | None = None
    hess: NDArray[np.float64] | None = None
    hess_inv: NDArray[np.float64] | None = None
    max_constraint_violation: float | None = None
    
    history: Any | None = None
    algorithm_output: dict[str, Any] | None = None
    multistart_info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        report: list[str] = []

        type_checks = {
            "fun": float,
            "start_fun": float,
            "algorithm": str,
            "success": bool,
            "message": str,
            "status": int,
            "n_fun_evals": int,
            "n_jac_evals": int,
            "n_hess_evals": int,
            "n_iterations": int,
            "jac": np.ndarray,
            "hess": np.ndarray,
            "hess_inv": np.ndarray,
            "max_constraint_violation": float,
            "algorithm_output": dict,
            "multistart_info": dict,
        }
        for field, typ in type_checks.items():
            val = getattr(self, field, None)
            if val is not None and not isinstance(val, typ):
                report.append(f"{field} must be a {typ.__name__} or None")
        
        if report:
            msg = (
                "The following arguments to OptimizeResult are invalid:\n"
                + "\n".join(report)
            )
            raise TypeError(msg)
        
    @property
    def x(self):
        return self.params

    @property
    def x0(self):
        return self.start_params

    @property
    def nfev(self) -> int | None:
        return self.n_fun_evals

    @property
    def nit(self) -> int | None:
        return self.n_iterations

    @property
    def njev(self) -> int | None:
        return self.n_jac_evals

    @property
    def nhev(self) -> int | None:
        return self.n_hess_evals
    
    def __getitem__(self, key):
        return getattr(self, key)
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
    """

    def __init__(self, algorithm: str, algo_options: dict | None = None):
        """
        Initializes an optimizer with a specified algorithm and options.

        Parameters:
        ----------
        algorithm (str): 
            The name of the optimization algorithm.
        algo_options (dict, optional): 
            A dictionary of options specific to the algorithm. Defaults to None.

        Raises:
        -------
            ValueError: If the specified algorithm is not valid.
        """

        if not self.has_algorithm(algorithm):
            raise ValueError(f"{algorithm} is not a valid algorithm name.\n")

        self.algo_name = algorithm

        self.algo_options = algo_options or {}

    def __init_subclass__(cls, **kwargs):
        """
        Register the subclass as an optimizer plugin and update the plugin stub if update_stub.
        """
        super().__init_subclass__(**kwargs)
        OptimizerManager.register(cls)

    @abstractmethod
    def fitting(
        self,
        fun,
        x0,
        bounds,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        **kwargs,
    ) -> OptimizeResult:
        """
        Perform the fitting process.

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
        pass

    def set_options(self, **kwargs):
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
    def available_algorithm(cls) -> List[str]:
        """
        List of available algorithms.

        Returns
        -------
        List[str]
            A list of available algorithm names.
        """
        pass


class OptimizerManager(PluginManager[OptimizerBase]):
    """
    Factory class for accessing registered optimizer plugins.
    """
    _plugins = {}
    _plugin_module = "gal3d.optimization.optimizer_plugins"
    _base_class = OptimizerBase
    
    
    
Optimizer = OptimizerManager

