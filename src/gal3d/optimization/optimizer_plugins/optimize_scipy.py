from typing import SupportsInt

from numpy.typing import NDArray

from scipy import optimize  # type: ignore
from scipy.optimize import OptimizeResult as ScipyOptimizeResult    # type: ignore

from ..optimizer import OptimizerBase, OptimizeResult

__all__ = ['OptimizerScipy']

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

    def __init__(self, algorithm, algo_options=dict()):
        super().__init__(algorithm, algo_options)

    def fitting(
        self,
        fun,
        x0,
        bounds,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        **kwargs,
    ):
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}
        fn = lambda x: fun(x, *func_args, **func_kwargs)
        res = optimize.minimize(
            fun=fn,
            x0=x0,
            method=self.algo_name,
            bounds=bounds,
            options=self.algo_options,
            **kwargs,
        )
        start_fun = fn(x0)
        return process_scipy_result(self.algo_name,x0,start_fun,res)

    @classmethod
    def available_algorithm(cls):
        return [
            'Nelder-Mead',
            'Powell',
            'CG',
            'BFGS',
            'Newton-CG',
            'L-BFGS-B',
            'TNC',
            'COBYLA',
            'COBYQA', 
            'SLSQP',
            'trust-constr',
            'dogleg',
            'trust-ncg',
            'trust-exact',
            'trust-krylov',
        ]
