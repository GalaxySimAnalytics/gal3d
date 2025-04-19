from typing import Iterable, Callable, Dict, Tuple
import time
import logging

import numpy as np
from tqdm import tqdm

from .point import Particles
from .field import SphField

from .shape import Structure3D
from .optimization.optimizer import Optimizer
from .optimization.result import ModelResult

logger = logging.getLogger("gal3d.analyzer")


class Gal3DAnalyzer:

    fit_workflow: Dict[str, Tuple[Callable, Callable]] = {}

    def __init__(
        self,
        particle: Particles,
        field: SphField,
        structure: Structure3D,
        optimizer: Optimizer,
    ):

        self.particle = particle
        self.field = field
        self.structure = structure
        self.optimizer = optimizer

    def fit(self, r: float | Iterable = np.geomspace(1, 10, 200), **kwargs):

        work = self.get_work()

        if not isinstance(r, Iterable):

            return work(self, r, **kwargs)

        else:
            resall = []
            for i in tqdm(r):
                try:
                    if resall:
                        res_value = {i: resall[-1][i][0] for i in resall[-1].keys()}
                        kwargs['init_parameters'] = res_value
                    res = work(self, i, **kwargs)
                    resall.append(res)
                except Exception as e:
                    logger.error(f'Skip fitting at radius {i:.2f}, for error {e}')

            if len(resall) > 0:
                return sum(resall[1:], resall[0])
            else:
                return resall

    def get_work(self):

        for i, j in self.fit_workflow.items():
            if j[0](self):
                logger.info(f"Use {i} workflow")
                return j[1]

        raise TypeError(f"No valid workflow")

    @staticmethod
    def fit_workflow_registry(fn_or_name: str | Callable) -> Callable:

        assert isinstance(fn_or_name, str) or callable(fn_or_name)

        default_condition = lambda cls: True

        if callable(fn_or_name):

            fn = fn_or_name
            name = fn.__name__
            Gal3DAnalyzer.fit_workflow[name] = (default_condition, fn)

            def set_condition(condition: Callable):

                assert callable(condition)
                Gal3DAnalyzer.fit_workflow[name] = (condition, fn)
                return fn

            fn.set_condition = set_condition
            return fn

        fn_name = fn_or_name

        def decorator(fn: Callable) -> Callable:

            assert callable(fn)

            if callable(fn):

                Gal3DAnalyzer.fit_workflow[fn_name] = (default_condition, fn)

                def set_condition(condition: Callable):

                    assert callable(condition)
                    Gal3DAnalyzer.fit_workflow[name] = (condition, fn)
                    return fn

                fn.set_condition = set_condition
                return fn

        return decorator


@Gal3DAnalyzer.fit_workflow_registry
def get_ell_structure(self: Gal3DAnalyzer, a: float, **kwargs) -> ModelResult:
    """
    Fit an ellipsoid or generalized ellipsoid (Ellipsoid_S) structure.
    """

    var_a = kwargs.get('var_a', 0)
    var_a = min(var_a, 0.99)
    var_a = max(var_a, 0)

    init_parameters = kwargs.get('init_parameters', dict())
    upper_bounds = kwargs.get('upper_bounds', dict())
    lower_bounds = kwargs.get('lower_bounds', dict())
    fitonce = kwargs.get('fitonce', False)

    data = self.field.generate(a, for_fit=True)

    if (self.structure._geometry_name == 'Ellipsoid') or (
        self.structure._geometry_name == 'Ellipsoid_S' and fitonce
    ):

        parameters_set = self.structure.parameters.new()
        fun = parameters_set.decorate_func_contraints(self.structure._error_method)

        if 'info' in data:
            parameters_set.add_info(**data['info'])
            del data['info']

        parameters_set.set_lb(a=(a * (1 - var_a)))
        parameters_set.set_ub(a=(a * (1 + var_a)))

        for i in upper_bounds.keys() & parameters_set.keys():
            parameters_set[i].ub = upper_bounds[i]

        for i in lower_bounds.keys() & parameters_set.keys():
            parameters_set[i].lb = lower_bounds[i]

        bounds = parameters_set.scipy_bounds

        if init_parameters:
            parameters_set = parameters_set.set_value(**init_parameters)
        parameters_set.set_value(a=a)

        x0_dict = parameters_set.truncate_dict(n=4)

        # we need first fitting Ellipsoid, then fit sa,sb,sc and a for Elllipsoid_S
    if self.structure._geometry_name == 'Ellipsoid_S':

        parameters_set = self.structure.parameters.new()
        parameters_set.set_lb(a=(a * (1 - var_a)))
        parameters_set.set_ub(a=(a * (1 + var_a)))

        if init_parameters:
            updatevalue = {
                i: init_parameters[i]
                for i in init_parameters.keys() & parameters_set.keys()
            }
            parameters_set = parameters_set.set_value(**updatevalue)
        parameters_set.set_value(a=a)

        ellipsoid = Structure3D(
            self.structure._coordinate,
            "Ellipsoid",
            self.structure._error_func_name,
            self.structure._error_method_name,
        )
        params_ell = ellipsoid.parameters.new()
        params_ell.set_lb(a=(a * (1 - var_a)))
        params_ell.set_ub(a=(a * (1 + var_a)))
        if init_parameters:
            updatevalue = {
                i: init_parameters[i]
                for i in init_parameters.keys() & params_ell.keys()
            }
            params_ell.set_value(**updatevalue)
        params_ell.set_value(a=a)

        for i in upper_bounds.keys() & params_ell.keys():
            params_ell[i].ub = upper_bounds[i]

        for i in lower_bounds.keys() & params_ell.keys():
            params_ell[i].lb = lower_bounds[i]

        bounds = params_ell.scipy_bounds

        fun = params_ell.decorate_func_contraints(ellipsoid._error_method)

        x0_dict = params_ell.truncate_dict(n=4)
        ell_res = self.optimizer.fitting(
            fun, list(x0_dict.values()), bounds, func_kwargs=dict(**data)
        )

        res_value = dict(zip(list(x0_dict.keys()), ell_res.x))

        parameters_set.set_value(**res_value)

        x0_dict = parameters_set.truncate_dict(n=4)
        x0_dict.update(res_value)

        del res_value['a']
        parameters_set.set_lb(**res_value)
        parameters_set.set_ub(**res_value)

        bounds = parameters_set.scipy_bounds

        parameters_set.add_info(parent_fun=ell_res.fun)

        fun = parameters_set.decorate_func_contraints(self.structure._error_method)

        if 'info' in data:
            parameters_set.add_info(**data['info'])
            del data['info']

    op_res = self.optimizer.fitting(
        fun, list(x0_dict.values()), bounds, func_kwargs={**data}
    )
    parameters_set = parameters_set.set_value(op_res.x)

    res = ModelResult(self.structure, op_res, parameters_set)

    return res


@get_ell_structure.set_condition
def ell_condition(self: Gal3DAnalyzer):

    return self.structure._geometry_name in ['Ellipsoid', 'Ellipsoid_S']
