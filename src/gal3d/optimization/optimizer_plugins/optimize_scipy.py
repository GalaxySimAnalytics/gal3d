from typing import SupportsInt

from scipy import optimize
from scipy.optimize import OptimizeResult as ScipyOptimizeResult

from ..optimizer import OptimizerBase, classproperty
from .util import InternalOptimizeResult

__all__ = ['OptimizerScipy']

# from optimagic
def _int_if_not_none(value: SupportsInt | None) -> int | None:
    if value is None:
        return None
    return int(value)


# from optimagic
def process_scipy_result(scipy_res: ScipyOptimizeResult) -> InternalOptimizeResult:
    res = InternalOptimizeResult(
        x=scipy_res.x,
        fun=scipy_res.fun,
        success=bool(scipy_res.success),
        message=str(scipy_res.message),
        n_fun_evals=_int_if_not_none(scipy_res.get("nfev")),
        n_jac_evals=_int_if_not_none(scipy_res.get("njev")),
        n_hess_evals=_int_if_not_none(scipy_res.get("nhev")),
        n_iterations=_int_if_not_none(scipy_res.get("nit")),
        # TODO: Pass on more things once we can convert them to external
        status=None,
        jac=None,
        hess=None,
        hess_inv=None,
        max_constraint_violation=None,
        info=None,
        #history=None,
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
        return process_scipy_result(res)

    @classproperty
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
