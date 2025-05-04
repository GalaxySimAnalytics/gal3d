import nlopt
from optimagic.optimizers.nlopt_optimizers import _process_nlopt_results

from ..optimizer import OptimizerBase, classproperty

__all__ = ['OptimizerNLopt']


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

    @classproperty
    def available_algorithm(cls):
        return list(filter(lambda x: x[:3] in ['GN_', 'GD_', 'LN_', 'LD_'], dir(nlopt)))
