"""
Optimizer plugin for optimagic.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import optimagic as om
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

from gal3d.optimization.optimizer import OptimizerBase, OptimizeResult
from gal3d.optimization.parameter import ParameterDict

__all__ = ["OptimizerOptimagic"]

def process_om_result(algorithm: str, x0: NDArray, om_result: om.OptimizeResult, params: ParameterDict) -> OptimizeResult:
    """Convert optimagic optimization results to standard OptimizeResult format."""
    # Process multistart info if available
    if om_result.multistart_info:
        multistart={
            "start_parameters": om_result.multistart_info.start_parameters,
            "local_optima": om_result.multistart_info.local_optima,
            "exploration_sample": om_result.multistart_info.exploration_sample,
            "exploration_results": om_result.multistart_info.exploration_results,
            "n_optimizations": om_result.multistart_info.n_optimizations
        }
    else:
        multistart = None

    # Create basic result with required attributes
    result = OptimizeResult(
        params = params,
        fun=om_result.fun,
        cost=om_result.fun,
        start_params=x0,
        start_fun=om_result.start_fun,
        algorithm=algorithm,
        success=om_result.success,
        message=om_result.message,
    )

    # Add optional attributes only if they're not None
    optional_attrs = {
        "n_fun_evals": om_result.n_fun_evals,
        "n_jac_evals": om_result.n_jac_evals,
        "n_hess_evals": om_result.n_hess_evals,
        "n_iterations": om_result.n_iterations,
        "status": om_result.status,
        "jac": om_result.jac,
        "hess": om_result.hess,
        "hess_inv": om_result.hess_inv,
        "max_constraint_violation": om_result.max_constraint_violation,
        "algorithm_output": om_result.algorithm_output,
        "history": om_result.history,
    }

    for attr_name, attr_value in optional_attrs.items():
        if attr_value is not None:
            result[attr_name] = attr_value

    # Add multistart info if available
    if multistart is not None:
        result["multistart_info"] = multistart

    return result

class OptimizerOptimagic(OptimizerBase):
    """
    optimagic
    optimagic is a Python package for numerical optimization.
    """

    def __init__(self, algorithm: str, algo_options: dict | None = None):
        if algo_options is None:
            algo_options = {}
        super().__init__(algorithm, algo_options)

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
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}
        start_params = np.asarray(x0)
        def fn(x):
            return fun(x, *func_args, **func_kwargs)
        om_result = om.minimize(
            fun=fn,
            params=start_params,
            algorithm=self.algo_name,
            bounds=bounds,
            algo_options=self.algo_options,
            **kwargs,
        )
        params = self.create_params(om_result.x, param_names=param_names, param_lbs=bounds.lb, param_ubs=bounds.ub)

        return process_om_result(self.algo_name, start_params, om_result, params)

    @classmethod
    def available_algorithm(cls):
        return om.algos.AvailableNames
