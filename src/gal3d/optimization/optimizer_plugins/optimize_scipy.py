from collections.abc import Callable
from typing import Any, SupportsInt

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.optimize import Bounds, OptimizeResult as ScipyOptimizeResult

from gal3d.optimization.optimizer import OptimizerBase, OptimizeResult

__all__ = ["OptimizerScipy"]

# from optimagic
def _int_if_not_none(value: SupportsInt | None) -> int | None:
    if value is None:
        return None
    return int(value)


# from optimagic
def process_scipy_result(algorithm: str, x0: NDArray, start_fun: float, scipy_res: ScipyOptimizeResult) -> OptimizeResult:
    res = OptimizeResult(
        params=scipy_res.x,
        fun=scipy_res.fun,
        start_params=x0,
        start_fun=start_fun,
        algorithm = algorithm,
        success=bool(scipy_res.success),
        message=str(scipy_res.message),
        n_fun_evals=_int_if_not_none(scipy_res.get("nfev")),
        n_jac_evals=_int_if_not_none(scipy_res.get("njev")),
        n_hess_evals=_int_if_not_none(scipy_res.get("nhev")),
        n_iterations=_int_if_not_none(scipy_res.get("nit")),
        status=scipy_res.get("status"),
        jac=scipy_res.get("jac"),
        hess=scipy_res.get("hess"),
        hess_inv=None,
        max_constraint_violation=scipy_res.get("maxcv"),
        algorithm_output=None,
    )
    return res


class OptimizerScipy(OptimizerBase):
    """
    scipy,optimize.minimize
    The minimize function provides a common interface to unconstrained and constrained minimization algorithms for multivariate scalar functions in scipy.optimize
    """

    def __init__(self, algorithm: str, algo_options: dict | None =None):
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
        **kwargs: Any,
    ) -> OptimizeResult:
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}
        def fn(x):
            return fun(x, *func_args, **func_kwargs)
        res = optimize.minimize(
            fun=fn,
            x0=x0,
            method=self.algo_name,
            bounds=bounds,
            options=self.algo_options,
            **kwargs,
        )
        start_fun = fn(x0)
        start_params = np.array(x0)
        return process_scipy_result(self.algo_name,start_params,start_fun,res)

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
        ]
