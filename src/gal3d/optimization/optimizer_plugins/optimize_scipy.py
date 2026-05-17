"""
Optimizer plugin for SciPy.

"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.optimize import Bounds, OptimizeResult as ScipyOptimizeResult

from gal3d.optimization.optimizer import OptimizerBase, OptimizeResult
from gal3d.optimization.parameter import ParameterDict

__all__ = ["OptimizerScipy"]


def process_scipy_result(
    algorithm: str, x0: NDArray, start_fun: float, scipy_res: ScipyOptimizeResult, params: ParameterDict
) -> OptimizeResult:
    """Convert SciPy optimization results to standard OptimizeResult format."""
    # create basic result object
    res = OptimizeResult(
        params=params,
        fun=scipy_res.fun,
        start_params=x0,
        start_fun=start_fun,
        algorithm=algorithm,
        success=bool(scipy_res.success),
        message=str(scipy_res.message),
    )

    attribute_map = {
        "nfev": "n_fun_evals",
        "njev": "n_jac_evals",
        "nhev": "n_hess_evals",
        "nit": "n_iterations",
        "status": "status",
        "jac": "jac",
        "hess": "hess",
        "hess_inv": "hess_inv",
        "grad": "grad",
        "optimality": "optimality",
        "maxcv": "max_constraint_violation",
        "active_mask": "active_mask",
    }

    for scipy_key, res_key in attribute_map.items():
        value = scipy_res.get(scipy_key)
        if value is not None:
            if scipy_key in ["nfev", "njev", "nhev", "nit"]:
                value = int(value)
            res[res_key] = value

    res["cost"] = scipy_res.get("cost", scipy_res.fun)

    return res


class OptimizerScipy(OptimizerBase):
    """
    scipy,optimize.minimize
    The minimize function provides a common interface to unconstrained and constrained minimization algorithms for multivariate scalar functions in scipy.optimize
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

        def fn(x):
            return fun(x, *func_args, **func_kwargs)

        if self.algo_name in ["trf", "dogbox", "lm"]:
            solver = optimize.least_squares
            kwargs["tr_options"] = self.algo_options
        else:
            solver = optimize.minimize
            kwargs["options"] = self.algo_options
        res = solver(fun=fn, x0=x0, method=self.algo_name, bounds=bounds, **self.kwargs, **kwargs)
        start_fun = fn(x0)
        start_params = np.array(x0)
        params = self.create_params(res.x, param_names=param_names, param_lbs=bounds.lb, param_ubs=bounds.ub)
        return process_scipy_result(self.algo_name, start_params, start_fun, res, params)

    @classmethod
    def available_algorithm(cls):
        return [
            "Nelder-Mead",
            "Powell",
            "CG",
            "BFGS",
            "Newton-CG",
            "L-BFGS-B",
            "TNC",
            "COBYLA",
            "COBYQA",
            "SLSQP",
            "trust-constr",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
            "trf",
            "dogbox",
            "lm",
        ]
