import emcee
from matplotlib.pyplot import cla
import numpy as np

from ..optimizer import OptimizerBase
from .util import InternalOptimizeResult


class OptimizerEmcee(OptimizerBase):
    """
    Optimization using the emcee library (MCMC sampling). Converts least squares fun to log_prob and applies bounds.
    """

    def __init__(self, algorithm="emcee", algo_options=dict()):
        super().__init__(algorithm, algo_options)

    def fitting(
        self,
        fun,
        x0,
        bounds,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        nwalkers: int = 32,
        nsteps: int = 1000,
        **kwargs,
    ):
        func_args = func_args or ()
        func_kwargs = func_kwargs or {}

        def log_prob(x):
            # Check bounds
            if np.any(x < bounds.lb) or np.any(x > bounds.ub):
                return -np.inf
            # fun is sum of squared residuals, convert to log_prob
            chi2 = fun(x, *func_args, **func_kwargs)
            return -0.5 * chi2

        ndim = len(x0)
        # Initialize sampler
        pos = x0 + 1e-4 * np.random.randn(nwalkers, ndim)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

        sampler.run_mcmc(pos, nsteps, progress=False)
        flat_samples = sampler.get_chain(discard=int(nsteps*0.2), thin=15, flat=True)
        flat_log_prob = sampler.get_log_prob(discard=int(nsteps*0.2), thin=15, flat=True)
        best_idx = np.argmax(flat_log_prob)
        best_x = flat_samples[best_idx]
        best_fun = fun(best_x, *func_args, **func_kwargs)

        res = InternalOptimizeResult(
            x=best_x,
            fun=best_fun,
            success=True,
            message="emcee sampling finished",
            n_fun_evals=sampler.iteration * nwalkers,
            n_jac_evals=None,
            n_hess_evals=None,
            n_iterations=sampler.iteration,
            status=None,
            jac=None,
            hess=None,
            hess_inv=None,
            max_constraint_violation=None,
            info=None,
        )
        return res

    @classmethod
    def available_algorithm(cls):
        return ["emcee"]