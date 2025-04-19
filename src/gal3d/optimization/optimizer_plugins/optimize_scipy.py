from scipy import optimize
from optimagic.optimizers.scipy_optimizers import process_scipy_result


from ..optimizer import OptimizerBase, classproperty

__all__ = ['OptimizerScipy']


class OptimizerScipy(OptimizerBase):
    """scipy"""

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
            'COBYQA' 'SLSQP',
            'trust-constr',
            'dogleg',
            'trust-ncg',
            'trust-exact',
            'trust-krylov',
        ]
