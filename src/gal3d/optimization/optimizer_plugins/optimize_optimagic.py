import optimagic as om
import numpy as np

from ..optimizer import OptimizerBase, classproperty

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
        return res

    @classproperty
    def available_algorithm(cls):
        return om.algos.AvailableNames
