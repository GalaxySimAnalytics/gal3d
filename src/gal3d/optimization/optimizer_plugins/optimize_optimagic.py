
import numpy as np
import optimagic as om

from ..optimizer import OptimizerBase,OptimizeResult

__all__ = ['OptimizerOptimagic']


class OptimizerOptimagic(OptimizerBase):
    """
    optimagic
    optimagic is a Python package for numerical optimization.
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
        res = om.minimize(
            fun=fn,
            params=np.asarray(x0),
            algorithm=self.algo_name,
            bounds=bounds,
            algo_options=self.algo_options,
            **kwargs,
        )
        
        res = OptimizeResult(
        params=res.x,
        fun=res.fun,
        start_params=x0,
        start_fun=res.start_fun,
        algorithm = self.algo_name,
        success=res.success,
        message=res.message,
        n_fun_evals=res.n_fun_evals,
        n_jac_evals=res.n_jac_evals,
        n_hess_evals=res.n_hess_evals,
        n_iterations=res.n_iterations,
        status=res.status,
        jac=res.jac,
        hess=res.hess,
        hess_inv=res.hess_inv,
        max_constraint_violation=res.max_constraint_violation,
        algorithm_output=res.algorithm_output,
        history=res.history,
        multistart_info=res.multistart_info
    )
        return res

    @classmethod
    def available_algorithm(cls):
        return om.algos.AvailableNames
