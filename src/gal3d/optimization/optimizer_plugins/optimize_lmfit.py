"""
Optimizer plugin for LMFit.
"""

from collections.abc import Callable
from typing import Any

import lmfit
import numpy as np
from lmfit.minimizer import MinimizerResult
from numpy.typing import ArrayLike
from scipy.optimize import Bounds

from gal3d.optimization.optimizer import OptimizerBase, OptimizeResult
from gal3d.optimization.parameter import ParameterDict

__all__ = ["OptimizerLMFit"]


def process_lmfit_result(start_fun: float, res: MinimizerResult, params: ParameterDict) -> OptimizeResult:
    all_keys = {i for i in dir(res) if not i.startswith("_")}

    result = OptimizeResult()

    attribute_map = {"init_values": "start_params", "nfev": "n_fun_evals", "method": "algorithm"}
    # numdifftools ? ,
    # calc_covar (bool, optional) – Whether to calculate the covariance matrix (default is True) for solvers other than ‘leastsq’ and ‘least_squares’.
    # Requires the numdifftools package to be installed.
    if getattr(res, "errorbars", False):
        for i, j in res.uvars.items():
            params[i].err = j.std_dev
        all_keys.remove("uvars")

    for lm_key in all_keys:
        value = getattr(res, lm_key, None)
        if value is None or callable(value):
            continue
        if lm_key == "params":
            for i in params:
                params[i] = res.params[i].value
            result["params_obj"] = res.params
            continue
        name = attribute_map.get(lm_key, lm_key)
        result[name] = value
    result["params"] = params
    result["start_fun"] = start_fun
    result["cost"] = res.chisqr
    return result


class OptimizerLMFit(OptimizerBase):
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

        def fn(params):
            x = list(params.valuesdict().values())
            return fun(x, *func_args, **func_kwargs)

        x0_seq = np.asarray(x0, dtype=float).tolist()
        params = self.create_params(x0_seq, param_names=param_names, param_lbs=bounds.lb, param_ubs=bounds.ub)

        lm_params = lmfit.Parameters()
        for i, param in params.items():
            lm_params.add(i, value=float(param), vary=True, min=param.lb, max=param.ub)

        res = lmfit.minimize(fcn=fn, params=lm_params, method=self.algo_name, **self.kwargs, **kwargs)
        start_fun = fn(lm_params)
        return process_lmfit_result(start_fun, res, params)

    @classmethod
    def available_algorithm(cls):
        return [
            "leastsq",
            "least_squares",
            "differential_evolution",
            "brute",
            "basinhopping",
            "ampgo",
            "nelder",
            "lbfgsb",
            "powell",
            "cg",
            "newtonr",
            "cobyla",
            "bfgs",
            "tnc",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
            "trust-constr",
            "dogleg",
            "slsqp",
            "emcee",
            "shgo",
            "dual_annealing",
        ]
