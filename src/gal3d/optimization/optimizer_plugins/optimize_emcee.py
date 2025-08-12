from collections.abc import Callable
from typing import Any

import emcee
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import Bounds

from gal3d.optimization.optimizer import OptimizerBase, OptimizeResult


class OptimizerEmcee(OptimizerBase):
    """
    Optimization using the emcee library (MCMC sampling). Converts least squares fun to log_prob and applies bounds.
    """

    def __init__(self, algorithm:str ="emcee", algo_options: dict | None=None):
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
        nwalkers: int = 32,
        nsteps: int = 1000,
        **kwargs: Any,
    ) -> OptimizeResult:
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}

        def log_prob(x):
            # Check bounds
            if np.any(x < bounds.lb) or np.any(x > bounds.ub):
                return -np.inf
            # fun is sum of squared residuals, convert to log_prob
            chi2 = fun(x, *func_args, **func_kwargs)
            return -0.5 * chi2
        start_params = np.asarray(x0)
        ndim: int = len(start_params)
        start_fun = fun(x0, *func_args, **func_kwargs)
        # Initialize sampler
        rng = np.random.default_rng()
        pos = start_params + 1e-4 * rng.normal(size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

        sampler.run_mcmc(pos, nsteps, progress=False)
        flat_samples = sampler.get_chain(discard=int(nsteps*0.2), thin=15, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=int(nsteps*0.2), thin=15, flat=True)
        best_idx = np.argmax(flat_log_prob)
        best_x = flat_samples[best_idx]
        best_fun = fun(best_x, *func_args, **func_kwargs)

        res = OptimizeResult(
            params=best_x,
            fun=best_fun,
            start_params = start_params,
            start_fun = start_fun,
            algorithm = "emcee",
            success=True,
            message="emcee sampling finished",
            n_fun_evals=sampler.iteration * nwalkers,
            n_iterations=sampler.iteration,
        )
        return res

    @classmethod
    def available_algorithm(cls):
        return ["emcee"]
