from collections.abc import Callable
from typing import Any

import nlopt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

from gal3d.optimization.optimizer import OptimizerBase, OptimizeResult

__all__ = ["OptimizerNLopt"]


# from optimagic
def _process_nlopt_results(algorithm: str, x0: NDArray, start_fun: float, nlopt_obj: Any, solution_x: NDArray, is_global: bool) -> OptimizeResult:
    messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_ftol_rel or "
            "convergence_ftol_abs was reached"
        ),
        4: (
            "Optimizer stopped because convergence_xtol_rel or "
            "convergence_xtol_abs was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    success: bool | None = nlopt_obj.last_optimize_result() in [1, 2, 3, 4]
    if is_global and not success:
        success = None
    processed = OptimizeResult(
        params=solution_x,
        fun=nlopt_obj.last_optimum_value(),
        start_params=x0,
        start_fun=start_fun,
        algorithm=algorithm,
        success=success,
        message=messages[nlopt_obj.last_optimize_result()],
        n_fun_evals=nlopt_obj.get_numevals(),
    )

    return processed
class OptimizerNLopt(OptimizerBase):
    """
    nlopt
    NLopt is a free/open-source library for nonlinear optimization
    """

    def __init__(self, algorithm: str, algo_options: dict | None=None):
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

        is_global = self.algo_name[0] == "G"
        start_params = np.asarray(x0)
        opt = nlopt.opt(getattr(nlopt, self.algo_name), len(start_params))
        def fn(x, *_):
            return fun(x, *func_args, **func_kwargs)
        opt.set_min_objective(fn)
        opt.set_lower_bounds(bounds.lb)
        opt.set_upper_bounds(bounds.ub)

        opt.set_ftol_abs(self.algo_options.get("ftol_abs", 0))
        # opt.set_ftol_rel(algo_options.get('ftol_res',0))
        opt.set_xtol_rel(self.algo_options.get("xtol_rel", 1e-5))
        opt.set_xtol_abs(self.algo_options.get("xtol_abs", 0))
        opt.set_maxeval(self.algo_options.get("maxeval", len(start_params) * 1000))

        xopt = opt.optimize(start_params)
        start_fun = fun(start_params, *func_args, **func_kwargs)
        return _process_nlopt_results(self.algo_name, start_params, start_fun, opt, xopt, is_global)

    @classmethod
    def available_algorithm(cls):
        return list(filter(lambda x: x[:3] in ["GN_", "GD_", "LN_", "LD_"], dir(nlopt)))
