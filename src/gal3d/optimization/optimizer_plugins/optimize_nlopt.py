from matplotlib.pyplot import cla
import nlopt

from ..optimizer import OptimizerBase
from .util import InternalOptimizeResult

__all__ = ['OptimizerNLopt']


# from optimagic
def _process_nlopt_results(nlopt_obj, solution_x, is_global):
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
    success = nlopt_obj.last_optimize_result() in [1, 2, 3, 4]
    if is_global and not success:
        success = None
    processed = InternalOptimizeResult(
        x=solution_x,
        fun=nlopt_obj.last_optimum_value(),
        n_fun_evals=nlopt_obj.get_numevals(),
        success=success,
        message=messages[nlopt_obj.last_optimize_result()],
    )

    return processed
class OptimizerNLopt(OptimizerBase):
    """
    nlopt
    NLopt is a free/open-source library for nonlinear optimization
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

        is_global = self.algo_name[0] == 'G'
        opt = nlopt.opt(getattr(nlopt, self.algo_name), len(x0))
        fn = lambda x, *_: fun(x, *func_args, **func_kwargs)
        opt.set_min_objective(fn)
        opt.set_lower_bounds(bounds.lb)
        opt.set_upper_bounds(bounds.ub)

        opt.set_ftol_abs(self.algo_options.get('ftol_abs', 0))
        # opt.set_ftol_rel(algo_options.get('ftol_res',0))
        opt.set_xtol_rel(self.algo_options.get('xtol_rel', 1e-5))
        opt.set_xtol_abs(self.algo_options.get('xtol_abs', 0))
        opt.set_maxeval(self.algo_options.get('maxeval', len(x0) * 1000))

        xopt = opt.optimize(x0)

        return _process_nlopt_results(opt, xopt, is_global)

    @classmethod
    def available_algorithm(cls):
        return list(filter(lambda x: x[:3] in ['GN_', 'GD_', 'LN_', 'LD_'], dir(nlopt)))
