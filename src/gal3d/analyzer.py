from typing import Iterable, Callable, Dict, Tuple
import time
import logging

import numpy as np
from tqdm import tqdm

from .point import Particles
from .field import SphField, SphVector
from .shape import Structure3D
from .optimization.optimizer import Optimizer
from .optimization.result import ModelResult

from .configuration import _set_config_parser

logger = logging.getLogger("gal3d.analyzer")


class Gal3DAnalyzer:
    """
    An analyzer for fitting galaxy 3D structures using particle, field, and shape information.

    Attributes
    ----------
    fit_workflow : dict of str to tuple of (Callable, Callable)
        A registry for fitting workflow conditions and their corresponding functions.
    particle : Particles
        The input particle data.
    field : SphField
        The spherical field generator.
    structure : Structure3D
        The 3D geometric structure model to fit.
    optimizer : Optimizer
        The optimization engine used for fitting.
    """

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
    
    @staticmethod
    def from_config(pos, mass, config_file: str):
        cfg = _set_config_parser()
        cfg.read(config_file)
        particle_cfg = cfg['Point']
        particle = Particles(pos = pos,mass=mass,rmax=particle_cfg.getfloat("r_max"),
                             estimator_kwargs={"k_nearest":particle_cfg.getint("k_nearest",fallback=32),
                                               "r_cut": particle_cfg.getint("r_cut")})
        
        field_cfg = cfg['Field']
        field = SphField(particle,num_ray=field_cfg.getint("n_ray",fallback=1024),ray_method=field_cfg.get("ray_method",fallback="fibonacci")
            ).build_field_boundary(inner=field_cfg.getfloat("inner"),inner_mode=field_cfg.get("inner_mode"),
                                   outer=field_cfg.getfloat("outer"),outer_mode=field_cfg.get("outer_mode")  # softening_length/2
                                
            ).build_profile_sample(num_p=field_cfg.getint("n_step"),step_mode=field_cfg.get("step_mode")
            ).build_profile_interpolator(interpolator_method=field_cfg.get("interpolator"),
            ).build_isodensity_profile(method=field_cfg.get("isodensity_method"),from_rays_func=field_cfg.getboolean("from_rays_func"),
                                       res_b=field_cfg.getfloat("res_b"),res_c=field_cfg.getfloat("res_c"),num_p=field_cfg.getint("iso_step"),
                                       interpolator_method=field_cfg.get("isodensity_interpolator"))
        
        shape_cfg = cfg["Shape"]
        shape = Structure3D(coordinate=shape_cfg.get("coordinate"),
                            geometry=shape_cfg.get("geometry"),
                            error_func=shape_cfg.get("error_func"),
                            error_method=shape_cfg.get("error_method"),)
        
        optimizer_cfg = cfg['Optimizer']
        optimizer = Optimizer.get_plugin(plugin =optimizer_cfg.get("optimizer"))(algorithm=optimizer_cfg.get("algorithm")) # OptimizerScipy Powell
        
        return Gal3DAnalyzer(particle=particle,field=field,structure=shape,optimizer=optimizer)
        

    def fit(self, r: float | Iterable = np.geomspace(1, 10, 200), **kwargs) -> ModelResult:
        """
        Fit the model to a single radius or a sequence of radii.

        Parameters
        ----------
        r : float or iterable of float, optional
            The radius or sequence of radii at which to perform the fit.
            Defaults to log-spaced radii from 1 to 10.
        **kwargs : dict
            Additional keyword arguments passed to the fitting workflow.

        Returns
        -------
        ModelResult 
        """
        work = self.get_workflow()

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
                    logger.error(f'Skip fitting at radius {i:.2f}, for error: {e}')

            if len(resall) > 0:
                return sum(resall[1:], resall[0])
            else:
                return resall

    def get_workflow(self):
        """
        Determine which registered workflow to use based on its condition.

        Returns
        -------
        Callable
            The fitting workflow function to be used.

        Raises
        ------
        TypeError
            If no valid workflow condition is satisfied.
        """
        for i, j in self.fit_workflow.items():
            if j[0](self):
                logger.info(f"Use {i} workflow")
                return j[1]

        raise TypeError(f"No valid workflow")

    @staticmethod
    def fit_workflow_registry(fn_or_name: str | Callable) -> Callable:
        """
        Register a new workflow function with an optional condition.

        Parameters
        ----------
        fn_or_name : str or Callable
            If a string is given, it's used as the name for a future decorator.
            If a function is given, it's registered immediately.

        Returns
        -------
        Callable
            A decorator or the function with an attached `.set_condition` method.
        """

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
    Fit an ellipsoidal or generalized ellipsoidal structure at a given semi-major axis `a`.

    Parameters
    ----------
    self : Gal3DAnalyzer
        The analyzer instance.
    a : float
        The semi-major axis at which the structure is fitted.
    **kwargs : dict
        Optional arguments:
        - var_a : float
            Variance allowed for 'a' when setting parameter bounds.
        - init_parameters : dict
            Initial guess for parameters.
        - upper_bounds : dict
            Upper bounds for parameters.
        - lower_bounds : dict
            Lower bounds for parameters.
        - fitonce : bool
            If True and the geometry is `Ellipsoid_S`, fit only once.

    Returns
    -------
    ModelResult
        The result of the fitting process, including parameter values and optimizer output.
    """

    var_a = kwargs.get('var_a', 0.1)
    var_a = min(var_a, 0.99)
    var_a = max(var_a, 0)
    uniformity_cut =  kwargs.get('uniformity_cut', 0.75)

    init_parameters = kwargs.get('init_parameters', dict())
    upper_bounds = kwargs.get('upper_bounds', dict())
    lower_bounds = kwargs.get('lower_bounds', dict())
    fitonce = kwargs.get('fitonce', False)

    data = self.field.generate(a, for_fit=True)
    N_p = len(data['pos'])
    if N_p < 2:
        raise ValueError(f"Generate {N_p} points < 2.")
    if N_p < self.field.rays.num:
        uni = SphVector.cal_uniformity(data['pos'])
        if uni < uniformity_cut:
            raise ValueError(f"Generate {N_p} points, with uniformity: {uni:.3f} < {uniformity_cut}")

    if (self.structure._geometry_name == 'Ellipsoid') or (
        self.structure._geometry_name == 'Ellipsoid_S' and fitonce
    ):

        parameters_set = self.structure.parameters.new()
        fun = parameters_set.decorate_func_constraints(self.structure._error_method)

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

        fun = params_ell.decorate_func_constraints(ellipsoid._error_method)

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

        fun = parameters_set.decorate_func_constraints(self.structure._error_method)

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
def ell_condition(self: Gal3DAnalyzer) -> bool:
    """
    Condition for selecting the ellipsoidal fitting workflow.
    
    This function checks if the structure's geometry is one of the supported
    ellipsoidal types (Ellipsoid or Ellipsoid_S).
    
    Parameters
    ----------
    self : Gal3DAnalyzer
        The analyzer instance to check.
        
    Returns
    -------
    bool
        True if the structure's geometry is either 'Ellipsoid' or 'Ellipsoid_S',
        False otherwise.
    """
    geometry_name = self.structure._geometry_name
    is_supported = geometry_name in ['Ellipsoid', 'Ellipsoid_S']
    
    if is_supported:
        logger.debug(f"Selected ellipsoidal fitting workflow for geometry: {geometry_name}")
    else:
        logger.debug(f"Geometry {geometry_name} is not supported by ellipsoidal fitting workflow")
        
    return is_supported

# Add a new workflow for debugging/validation purposes
@Gal3DAnalyzer.fit_workflow_registry
def validate_fitting_data(self: Gal3DAnalyzer, a: float, **kwargs) -> ModelResult:
    """
    A workflow that validates fitting data and parameters before performing the fit.
    
    This method performs extensive validation of input data and parameters before
    proceeding with the actual fitting process. It helps catch issues early and
    provides detailed diagnostic information.
    
    Parameters
    ----------
    self : Gal3DAnalyzer
        The analyzer instance containing particle data, field, structure, and optimizer.
    a : float
        The semi-major axis at which to validate and fit the structure data.
    **kwargs : dict
        Optional arguments for validation and fitting:
        - var_a : float
            Variance allowed for 'a' parameter (default: 0.1)
        - uniformity_cut : float
            Minimum required point distribution uniformity (default: 0.75)
        - init_parameters : dict
            Initial parameter values for the fit
        - max_points : int
            Maximum number of points to use for fitting (default: 10000)
        - min_points : int
            Minimum required points for fitting (default: 10)
        
    Returns
    -------
    ModelResult
        The validated and fitted model result.
        
    Raises
    ------
    ValueError
        If validation fails with detailed information about the failure.
    RuntimeError
        If the fitting process fails after validation.
    """
    logger.info(f"Starting validation for fitting at a={a}")
    validation_start_time = time.time()
    
    # 1. Validate basic input parameters
    if a <= 0:
        error_msg = f"Semi-major axis must be positive, got a={a}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    var_a = kwargs.get('var_a', 0.1)
    if not (0 <= var_a < 1):
        logger.warning(f"var_a={var_a} outside recommended range [0,1), clamping to range")
        var_a = min(max(var_a, 0), 0.99)
    
    uniformity_cut = kwargs.get('uniformity_cut', 0.75)
    min_points = kwargs.get('min_points', 10)
    max_points = kwargs.get('max_points', 10000)
    
    # 2. Generate and validate field data
    try:
        logger.info(f"Generating field data at a={a}")
        field_data = self.field.generate(a, for_fit=True)
        
        # 3. Validate point count
        point_count = len(field_data.get('pos', []))
        logger.debug(f"Generated {point_count} points for fitting")
        
        if point_count < min_points:
            error_msg = f"Generated only {point_count} points, need at least {min_points} for reliable fitting"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if point_count > max_points:
            logger.warning(f"Large number of points ({point_count} > {max_points}), fitting may be slow")
            
        # 4. Validate point distribution uniformity
        if point_count < self.field.rays.num:
            uniformity = SphVector.cal_uniformity(field_data['pos'])
            logger.debug(f"Point distribution uniformity: {uniformity:.4f} (threshold: {uniformity_cut})")
            if uniformity < uniformity_cut:
                error_msg = f"Point distribution uniformity {uniformity:.4f} below threshold {uniformity_cut}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # 5. Check for NaN values in position data
        if np.isnan(field_data['pos']).any():
            error_msg = "Position data contains NaN values"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 6. Validate structure is compatible with data
        if not hasattr(self.structure, '_error_method'):
            error_msg = "Structure has no error method defined"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 7. After validation, proceed with fitting using the standard workflow
        logger.info(f"Validation successful ({time.time() - validation_start_time:.2f}s), proceeding with fitting")
        
        # Use the appropriate workflow based on structure type
        if self.structure._geometry_name in ['Ellipsoid', 'Ellipsoid_S']:
            return get_ell_structure(self, a, **kwargs)
        else:
            error_msg = f"No specific validation for geometry type '{self.structure._geometry_name}'"
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
            
    except ValueError as e:
        # Re-raise validation errors with context
        logger.error(f"Validation failed: {e}")
        raise
    except Exception as e:
        # Convert unexpected errors to RuntimeError with context
        error_msg = f"Unexpected error during validation: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


@validate_fitting_data.set_condition
def validation_condition(self: Gal3DAnalyzer) -> bool:
    """
    Condition to determine when to use the validation workflow.
    
    This workflow should be explicitly requested by setting 'validate=True'
    in the analyzer configuration.
    
    Parameters
    ----------
    self : Gal3DAnalyzer
        The analyzer instance to check.
        
    Returns
    -------
    bool
        True if validation is explicitly enabled, False otherwise.
    """
    # This workflow is not used by default, must be explicitly enabled
    return getattr(self, 'validate', False)
