
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
        om_result = om.minimize(
            fun=fn,
            params=np.asarray(x0),
            algorithm=self.algo_name,
            bounds=bounds,
            algo_options=self.algo_options,
            **kwargs,
        )
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
        
        result = OptimizeResult(
        params=om_result.x,
        fun=om_result.fun,
        start_params=x0,
        start_fun=om_result.start_fun,
        algorithm = self.algo_name,
        success=om_result.success,
        message=om_result.message,
        n_fun_evals=om_result.n_fun_evals,
        n_jac_evals=om_result.n_jac_evals,
        n_hess_evals=om_result.n_hess_evals,
        n_iterations=om_result.n_iterations,
        status=om_result.status,
        jac=om_result.jac,
        hess=om_result.hess,
        hess_inv=om_result.hess_inv,
        max_constraint_violation=om_result.max_constraint_violation,
        algorithm_output=om_result.algorithm_output,
        history=om_result.history,
        multistart_info=multistart
    )
        return result

    @classmethod
    def available_algorithm(cls):
        return om.algos.AvailableNames
