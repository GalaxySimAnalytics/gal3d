import logging
from typing import TYPE_CHECKING, Any

from gal3d.field import SphVector
from gal3d.fit_workflow.fit_workflow import FitWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.shape import Structure3D

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer


class InsufficientPointsError(ValueError):
    """Raised when there are not enough points for fitting."""

class PoorUniformityError(ValueError):
    """Raised when the point distribution uniformity is too low."""

logger = logging.getLogger("gal3d.fit_workflow_plugins")

class EllipsoidFitWorkflow(FitWorkflowBase):
    """
    Workflow for fitting ellipsoidal or generalized ellipsoidal structures.
    """

    @staticmethod
    def condition(analyzer: "Gal3DAnalyzer") -> bool:
        """
        Condition for selecting the ellipsoidal fitting workflow.

        This function checks if the structure's geometry is one of the supported
        ellipsoidal types (Ellipsoid or Ellipsoid_S).

        Parameters
        ----------
        analyzer : Gal3DAnalyzer
            The analyzer instance to check.

        Returns
        -------
        bool
            True if the structure's geometry is either 'Ellipsoid' or 'Ellipsoid_S',
            False otherwise.
        """
        geometry_name = analyzer.structure._geometry_name
        is_supported = geometry_name in ["Ellipsoid", "Ellipsoid_S"]
        if is_supported:
            logger.debug("Selected ellipsoidal fitting workflow for geometry: %s", geometry_name)

        return is_supported

    def __call__(self, analyzer: "Gal3DAnalyzer", a: float, **kwargs: Any) -> ModelResult:
        """
        Fit an ellipsoidal or generalized ellipsoidal structure at a given semi-major axis `a`.

        Parameters
        ----------
        analyzer : Gal3DAnalyzer
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
        geometry_name = analyzer.structure._geometry_name
        if geometry_name not in ["Ellipsoid", "Ellipsoid_S"]:
            raise ValueError(f"Unsupported geometry type: {geometry_name}")

        # Process input parameters efficiently
        var_a = min(max(kwargs.get("var_a", 0.1), 0), 0.99)
        var_cen = min(max(kwargs.get("var_cen", 0.1), 0), 0.99)
        fitonce = kwargs.get("fitonce", False)
        init_parameters = kwargs.get("init_parameters", {})
        upper_bounds = kwargs.get("upper_bounds", {})
        lower_bounds = kwargs.get("lower_bounds", {})
        uniformity_cut = kwargs.get("uniformity_cut", 0.75)

        # Generate data only once
        data = analyzer.field.generate(a, for_fit=True)

        # Extract info once if present
        info = data.pop("info", None)

        # Validate data points efficiently
        N_p = len(data["pos"])
        if N_p < 12:
            raise InsufficientPointsError("Insufficient points for fitting: < 12")
        if N_p < analyzer.field.rays.num:
            uni = SphVector.cal_uniformity(data["pos"])
            if uni < uniformity_cut:
                raise PoorUniformityError(f"Poor point distribution uniformity detected: < {uniformity_cut}")

        is_standard_ellipsoid = (geometry_name == "Ellipsoid") or (geometry_name == "Ellipsoid_S" and fitonce)

        if is_standard_ellipsoid:
            # Simple case: standard ellipsoid fitting
            return self._fit_standard_ellipsoid(analyzer, data, a, var_a, var_cen, init_parameters, upper_bounds, lower_bounds, info)
        else:
            # Complex case: two-step generalized ellipsoid fitting
            return self._fit_generalized_ellipsoid(analyzer, data, a, var_a, var_cen, init_parameters, upper_bounds, lower_bounds, info)

    def _fit_standard_ellipsoid(self, analyzer: "Gal3DAnalyzer", data: dict, a: float, var_a: float, var_cen: float, init_parameters: dict, upper_bounds: dict, lower_bounds: dict, info: dict) -> ModelResult:
        """
        Fit a standard ellipsoidal structure, or fit a generalized ellipsoid only once.
        """

        # Set up parameters
        parameters_set = analyzer.structure.parameters.new()
        if "info" in data:
            parameters_set.add_info(**data["info"])
            del data["info"]

        # Set bounds and initial values
        parameters_set.set_lb(a=(a * (1 - var_a)))
        parameters_set.set_ub(a=(a * (1 + var_a)))
        if var_cen:
            cen = var_cen * a
            parameters_set.set_lb(x=-cen, y=-cen, z=-cen)
            parameters_set.set_ub(x=cen, y=cen, z=cen)

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
        fun = parameters_set.decorate_func_constraints(analyzer.structure._error_method)
        bounds = parameters_set.scipy_bounds
        x0_dict = parameters_set.get_rounded_values_dict(n=4)
        x0_values = list(x0_dict.values())

        # Run optimization
        op_res = analyzer.optimizer.fitting(fun, x0_values, bounds, func_kwargs=data)

        # Set optimized values and return result
        parameters_set = parameters_set.set_value(op_res.x)
        return ModelResult(analyzer.structure, op_res, parameters_set)

    def _fit_generalized_ellipsoid(self, analyzer: "Gal3DAnalyzer", data: dict, a: float, var_a: float, var_cen: float, init_parameters: dict, upper_bounds: dict, lower_bounds: dict, info: dict) -> ModelResult:
        """
        Fit a generalized ellipsoidal structure (Ellipsoid_S) using a two-step approach.
        """

        # Step 1: Create standard ellipsoid for initial fitting
        ellipsoid = Structure3D(
            analyzer.structure._coordinate,
            "Ellipsoid",
            analyzer.structure._error_func_name,
            analyzer.structure._error_method_name,
        )

        # Reuse parameter setup logic
        ell_params = ellipsoid.parameters.new()
        for param_name in set(analyzer.structure.parameters.keys()) & set(ell_params.keys()):
            ell_params[param_name].assign_bounds(
                analyzer.structure.parameters[param_name].lb,
                analyzer.structure.parameters[param_name].ub
            )
        ell_params["a"].assign_bounds(a * (1 - var_a), a * (1 + var_a))
        if var_cen:
            cen = var_cen * a
            for name in ["x", "y", "z"]:
                ell_params[name].assign_bounds(-cen, cen)

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
        ell_res = analyzer.optimizer.fitting(
            fun, list(x0_dict.values()), bounds, func_kwargs=data
        )

        # Step 2: Use ellipsoid results to initialize the generalized ellipsoid fit
        res_value = dict(zip(param_keys, ell_res.x, strict=False))

        # Set up parameters for the generalized ellipsoid
        parameters_set = analyzer.structure.parameters.new()
        parameters_set["a"].assign_bounds(a * (1 - var_a), a * (1 + var_a))

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
        if "a" in shape_params:
            del shape_params["a"]   # Keep 'a' free to vary

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
        fun = parameters_set.decorate_func_constraints(analyzer.structure._error_method)
        bounds = parameters_set.scipy_bounds
        x0_dict = parameters_set.get_rounded_values_dict(n=4)
        x0_dict.update(res_value)

        # Run final optimization
        op_res = analyzer.optimizer.fitting(
            fun, list(x0_dict.values()), bounds, func_kwargs=data
        )

        # Update parameters and return result
        parameters_set = parameters_set.set_value(op_res.x)
        return ModelResult(analyzer.structure, op_res, parameters_set)
