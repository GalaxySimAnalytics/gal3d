from logging import config
from typing import Iterable, Callable, Dict, Tuple, Union
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
    def analyze(pos, mass, **kwargs):
        """
        Analyze the given particle data.

        Parameters
        ----------
        pos : np.ndarray
            The positions of the particles.
        mass : np.ndarray
            The masses of the particles.
        **kwargs : dict
            Additional keyword arguments for analysis.

        """
        
        logger.info("Starting analysis...")


        particle = Particles(pos=pos, mass=mass)
        
        if "res_r" not in kwargs:
            hsm = particle.hsm
            d_in = np.median(hsm)-3*np.std(hsm)
            d_ou = np.median(hsm)+3*np.std(hsm)
            res_r = np.mean(hsm[(hsm>d_in) & (hsm<d_ou)])
        else:
            res_r = kwargs["res_r"]

        #res_r = ((np.sum(particle.mass)/np.sum(particle.parameter))/(4/3*np.pi))**(1/3)*20
        #particle.estimator._tree_query_options['distance_upper_bound'] = res_r*10
        
        
        res_m = np.mean(particle.mass)
        logger.info("Estimated mass resolution: %f, spatial resolution: %f", res_m, res_r)

        Num_rays = min(1024,int(len(particle.r)/100))
        Num_rays = max(Num_rays,64)
        
        inner = res_r/2
        inner_mode = 'dist'
        
        outer = res_m/(4*np.pi/3*(3*res_r)**3)
        outer_mode = 'value'

        logger.info("Set inner radius to %f", inner)
        logger.info("Set outer value to %f", outer)

        logger.info("Building spherical field...")
        field = SphField(particle, num_ray=Num_rays
                ).build_field_boundary(inner = inner, outer=outer,inner_mode=inner_mode,outer_mode=outer_mode
                ).build_profile_sample(
                ).build_profile_interpolator(
                ).build_isodensity_profile(
                )
                
        ellipsoid_s = Structure3D(coordinate='EulerShift',geometry='Ellipsoid_S',error_func='sums_dev_rscale',error_method='isodensity_dcall')
        optimizer = Optimizer.get_plugin(plugin = 'OptimizerScipy')(algorithm='Powell') # OptimizerScipy Powell
        
        ellipsoid_s.parameters.set_ub(x=inner,y=inner,z=inner)
        ellipsoid_s.parameters.set_lb(x=-inner,y=-inner,z=-inner)

        return Gal3DAnalyzer(particle=particle,field=field,structure=ellipsoid_s,optimizer=optimizer)

    @staticmethod
    def from_config(pos, mass, config_file: str):
        """
        Create a Gal3DAnalyzer instance from a configuration file.

        Parameters
        ----------
        pos : np.ndarray
            The positions of the particles.
        mass : np.ndarray
            The masses of the particles.
        config_file : str
            The path to the configuration file.

        Returns
        -------
        Gal3DAnalyzer
            An instance of the Gal3DAnalyzer class.
        """
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
        try:
            plugin_name = optimizer_cfg.get("optimizer")
            algorithm_name = optimizer_cfg.get("algorithm")
            optimizer_plugin = Optimizer.get_plugin(plugin=plugin_name)
            optimizer = optimizer_plugin(algorithm=algorithm_name)
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise ValueError(f"Configuration file is missing required key: {e}")
        except AttributeError as e:
            logger.error(f"Invalid optimizer plugin or algorithm: {e}")
            raise ValueError(f"Invalid optimizer plugin '{plugin_name}' or algorithm '{algorithm_name}' specified in configuration.")
        
        return Gal3DAnalyzer(particle=particle,field=field,structure=shape,optimizer=optimizer)
    
    
    def fit(self, num_step:int = 200, step_mode: str = 'log'):
        """ 
        Fit the model to the data.

        Parameters
        ----------
        num_step : int
            The number of steps for the fitting process.
        step_mode : str
            The mode of the steps (e.g., 'log' for logarithmic spacing).

        """
        r_min = max(np.median(self.field.inner_r)*6,self.field.iso_pro_r[0]*3)
        r_max = min(self.field.iso_pro_r[-1],np.median(self.field.outer_r))

        if step_mode == 'log':
            r = np.geomspace(r_min, r_max, num_step)
        else:
            r = np.linspace(r_min, r_max, num_step)

        return self._fit(r)
        

    def _fit(self, r: Union[float, Iterable] = np.geomspace(1, 10, 200), **kwargs) -> ModelResult:
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
            errors = []
            for i in tqdm(r, desc="Fitting radii", disable=False):
                try:
                    if resall:
                        res_value = {key: resall[-1][key][0] for key in resall[-1].keys()}
                        kwargs['init_parameters'] = res_value
                    res = work(self, i, **kwargs)
                    resall.append(res)
                except Exception as e:
                    errors.append(f"Radius {i:.2f}: {e}")
            
            if errors:
                logger.error("Skipped fitting at some radii due to errors:\n" + "\n".join(errors))
            
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
    def fit_workflow_registry(fn_or_name: Union[str, Callable]) -> Callable:
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
            fn_name = fn.__name__
            Gal3DAnalyzer.fit_workflow[fn_name] = (default_condition, fn)

            def set_condition(condition: Callable):

                assert callable(condition)
                Gal3DAnalyzer.fit_workflow[fn_name] = (condition, fn)
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
                    Gal3DAnalyzer.fit_workflow[fn_name] = (condition, fn)
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
        Optional arguments for the fitting process.

    Returns
    -------
    ModelResult
        The result of the fitting process.
    """
    # Fast path: check for supported geometry before generating data
    geometry_name = self.structure._geometry_name
    if geometry_name not in ['Ellipsoid', 'Ellipsoid_S']:
        raise ValueError(f"Unsupported geometry type: {geometry_name}")
    
    # Process input parameters efficiently
    var_a = min(max(kwargs.get('var_a', 0.1), 0), 0.99)
    fitonce = kwargs.get('fitonce', False)
    init_parameters = kwargs.get('init_parameters', {})
    upper_bounds = kwargs.get('upper_bounds', {})
    lower_bounds = kwargs.get('lower_bounds', {})
    uniformity_cut = kwargs.get('uniformity_cut', 0.75)
    
    # Generate data only once
    data = self.field.generate(a, for_fit=True)
    
    # Extract info once if present
    info = None
    if 'info' in data:
        info = data.pop('info')
    
    # Validate data points efficiently
    N_p = len(data['pos'])
    if N_p < 2:
        raise ValueError(f"Insufficient points for fitting: {N_p} < 2")
    if N_p < self.field.rays.num:
        uni = SphVector.cal_uniformity(data['pos'])
        if uni < uniformity_cut:
            raise ValueError(f"Poor point distribution uniformity: {uni:.3f} < {uniformity_cut}")
        
    is_standard_ellipsoid = (geometry_name == 'Ellipsoid') or (geometry_name == 'Ellipsoid_S' and fitonce)
    
    if is_standard_ellipsoid:
        # Simple case: standard ellipsoid fitting
        return _fit_standard_ellipsoid(
            self, data, a, var_a, init_parameters, upper_bounds, lower_bounds, info
        )
    else:
        # Complex case: two-step generalized ellipsoid fitting
        return _fit_generalized_ellipsoid(
            self, data, a, var_a, init_parameters, upper_bounds, lower_bounds, info
        )

def _fit_standard_ellipsoid(self, data, a, var_a, init_parameters, upper_bounds, lower_bounds, info):
    """
    Fit a standard ellipsoidal structure, or fit a generalized ellipsoid only once.
    """

    # Set up parameters
    parameters_set = self.structure.parameters.new()
    
    if 'info' in data:
        parameters_set.add_info(**data['info'])
        del data['info']
    
    # Set bounds and initial values
    parameters_set.set_lb(a=(a * (1 - var_a)))
    parameters_set.set_ub(a=(a * (1 + var_a)))
    
    # Apply custom bounds using set operations for efficiency
    for param_name in set(upper_bounds) & set(parameters_set.keys()):
        parameters_set[param_name].ub = upper_bounds[param_name]
    
    for param_name in set(lower_bounds) & set(parameters_set.keys()):
        parameters_set[param_name].lb = lower_bounds[param_name]
    
    if init_parameters:
        parameters_set = parameters_set.set_value(**init_parameters)
    parameters_set.set_value(a=a)
    
    # Add info if available
    if info:
        parameters_set.add_info(**info)
    
    # Prepare optimization function and bounds
    fun = parameters_set.decorate_func_constraints(self.structure._error_method)
    bounds = parameters_set.scipy_bounds
    x0_dict = parameters_set.get_rounded_values_dict(n=4)
    x0_values = list(x0_dict.values())
    
    # Run optimization
    op_res = self.optimizer.fitting(fun, x0_values, bounds, func_kwargs=data)
    
    # Set optimized values and return result
    parameters_set = parameters_set.set_value(op_res.x)
    return ModelResult(self.structure, op_res, parameters_set)


def _fit_generalized_ellipsoid(self, data, a, var_a, init_parameters, upper_bounds, lower_bounds, info):
    """
    Fit a generalized ellipsoidal structure (Ellipsoid_S) using a two-step approach.
    """
 
    # Step 1: Create standard ellipsoid for initial fitting
    ellipsoid = Structure3D(
        self.structure._coordinate,
        "Ellipsoid",
        self.structure._error_func_name,
        self.structure._error_method_name,
    )
    
    # Reuse parameter setup logic
    ell_params = ellipsoid.parameters.new()
    
    for param_name in set(self.structure.parameters.keys()) & set(ell_params.keys()):
        ell_params[param_name].ub = self.structure.parameters[param_name].ub
        ell_params[param_name].lb = self.structure.parameters[param_name].lb

    ell_params.set_lb(a=(a * (1 - var_a)))
    ell_params.set_ub(a=(a * (1 + var_a)))
    
    # Apply parameter constraints efficiently
    for param_name in set(upper_bounds) & set(ell_params.keys()):
        ell_params[param_name].ub = upper_bounds[param_name]
    
    for param_name in set(lower_bounds) & set(ell_params.keys()):
        ell_params[param_name].lb = lower_bounds[param_name]
        
    # Set initial values
    if init_parameters:
        shared_keys = set(init_parameters) & set(ell_params.keys())
        if shared_keys:
            update_values = {k: init_parameters[k] for k in shared_keys}
            ell_params = ell_params.set_value(**update_values)
    ell_params.set_value(a=a)
    
    # Prepare first optimization
    fun = ell_params.decorate_func_constraints(ellipsoid._error_method)
    bounds = ell_params.scipy_bounds
    x0_dict = ell_params.get_rounded_values_dict(n=4)
    param_keys = list(x0_dict.keys())

    # Run first optimization
    ell_res = self.optimizer.fitting(
        fun, list(x0_dict.values()), bounds, func_kwargs=data
    )

    # Step 2: Use ellipsoid results to initialize the generalized ellipsoid fit
    res_value = dict(zip(param_keys, ell_res.x))

    # Set up parameters for the generalized ellipsoid
    parameters_set = self.structure.parameters.new()
    parameters_set.set_lb(a=(a * (1 - var_a)))
    parameters_set.set_ub(a=(a * (1 + var_a)))
    
    # Apply initial values and lock shape parameters
    if init_parameters:
        updatevalue = {
            i: init_parameters[i]
            for i in init_parameters.keys() & parameters_set.keys()
        }
        parameters_set = parameters_set.set_value(**updatevalue)
    parameters_set.set_value(**res_value)

    # Lock all parameters except 'a'
    shape_params = res_value.copy()
    if 'a' in shape_params:
        del shape_params['a']  # Keep 'a' free to vary
        
    # Set up shape parameter bounds
    if shape_params:
        parameters_set.set_lb(**shape_params)
        parameters_set.set_ub(**shape_params)
        
    # Record parent function result
    parameters_set.add_info(parent_fun=ell_res.fun)
    
    # Add original info if available
    if info:
        parameters_set.add_info(**info)
        
    # Prepare final optimization
    fun = parameters_set.decorate_func_constraints(self.structure._error_method)
    bounds = parameters_set.scipy_bounds
    x0_dict = parameters_set.get_rounded_values_dict(n=4)
    x0_dict.update(res_value)

    # Run final optimization
    op_res = self.optimizer.fitting(
        fun, list(x0_dict.values()), bounds, func_kwargs=data
    )
    
    # Update parameters and return result
    parameters_set = parameters_set.set_value(op_res.x)
    return ModelResult(self.structure, op_res, parameters_set)



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
        # TODO: Consider adding more conditions in the future if new geometry types are introduced.
        logger.warning(f"Unsupported geometry type: {geometry_name}")

    return is_supported

# Add a new workflow for validating input data and parameters before fitting
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
        pos_data = np.array(field_data['pos']) if not isinstance(field_data['pos'], np.ndarray) else field_data['pos']
        if np.isnan(pos_data).any():
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
        
        # Temporarily disable validation to avoid recursion
        original_validate = getattr(self, 'validate', False)
        self.validate = False
        
        try:
            # Get a non-validation workflow
            workflow = self.get_workflow()
            return workflow(self, a, **kwargs)
        finally:
            # Restore the original validation setting
            self.validate = original_validate

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
