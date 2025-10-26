"""
Ellipsoidal fitting workflow plugin for Gal3D.
This module provides a fitting workflow for ellipsoidal and generalized ellipsoidal structures.
"""

import logging
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np

from gal3d.field import SphVector
from gal3d.model_workflow.fit_workflow import FitWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.shape import Structure3D
from gal3d.util.errors import InsufficientPointsError, PoorUniformityError

if TYPE_CHECKING:
    from gal3d.analyzer import Gal3DAnalyzer
    from gal3d.optimization.parameter import Parameters
    from gal3d.point import Particles

logger = logging.getLogger("gal3d.fit_workflow_plugins")

class EllipsoidFitWorkflow(FitWorkflowBase):
    """
    Workflow for fitting ellipsoidal or generalized ellipsoidal structures.
    """

    @staticmethod
    def condition(obj: Union["Gal3DAnalyzer", "Particles"]) -> bool:
        """
        Condition for selecting the ellipsoidal fitting workflow.

        This function checks if the structure's geometry is one of the supported
        ellipsoidal types (Ellipsoid or Ellipsoid_S).

        Parameters
        ----------
        obj : Gal3DAnalyzer or Particles
            The target instance to check.

        Returns
        -------
        bool
            True if the structure's geometry is either 'Ellipsoid' or 'Ellipsoid_S',
            False otherwise.
        """
        if hasattr(obj, "structure"):
            geometry_name = obj.structure._geometry_name
            is_supported = geometry_name in ["Ellipsoid", "Ellipsoid_S"]
            if is_supported:
                logger.debug("Selected ellipsoidal fitting workflow for geometry: %s", geometry_name)
        else:
            is_supported = False

        return is_supported

    def __call__(self, obj: Union["Gal3DAnalyzer", "Particles"], a: float, **kwargs: Any) -> ModelResult:
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
        if not self.condition(obj):
            raise ValueError("Unsupported object type for ellipsoid fitting.")
        else:
            obj = cast("Gal3DAnalyzer", obj)

        geometry_name = obj.structure._geometry_name
        if geometry_name not in ["Ellipsoid", "Ellipsoid_S"]:
            raise ValueError(f"Unsupported geometry type: {geometry_name}")

        # Process input parameters efficiently
        var_a = min(max(kwargs.get("var_a", 0.3), 0), 0.99)
        var_cen = kwargs.get("var_cen",0.1)
        single_fit = kwargs.get("single_fit", False)
        fix_eps = kwargs.get("fix_eps", True)
        init_parameters = kwargs.get("init_parameters", {})
        upper_bounds = kwargs.get("upper_bounds", {})
        lower_bounds = kwargs.get("lower_bounds", {})
        min_uniform = kwargs.get("min_uniform", 0.9)

        # Generate data only once
        data = obj.field.generate(a, for_fit=True)

        # Extract info once if present
        info = data.pop("info", {})
        info["data"] = data["pos"]

        # Validate data points efficiently
        N_p = len(data["pos"])
        if N_p < 12:
            raise InsufficientPointsError("Insufficient points for fitting: < 12")
        if N_p < obj.field.rays.num:
            uni = SphVector.cal_uniformity(data["pos"])
            if uni < min_uniform:
                raise PoorUniformityError(f"Poor point distribution uniformity detected: < {min_uniform}")

        if var_cen is not None:
            if ("x" in init_parameters and "y" in init_parameters and "z" in init_parameters):
                curr_cen = np.mean(data["pos"],axis=0)
                if abs(curr_cen[0] - init_parameters["x"])>var_cen*a:
                    del init_parameters["x"]
                if abs(curr_cen[1] - init_parameters["y"])>var_cen*a:
                    del init_parameters["y"]
                if abs(curr_cen[2] - init_parameters["z"])>var_cen*a:
                    del init_parameters["z"]


        is_standard_ellipsoid = (geometry_name == "Ellipsoid") or (geometry_name == "Ellipsoid_S" and single_fit)

        if is_standard_ellipsoid:
            # Simple case: standard ellipsoid fitting
            return self._fit_standard_ellipsoid(obj, data, a, var_a, var_cen, init_parameters, upper_bounds, lower_bounds, info)
        else:
            # Complex case: two-step generalized ellipsoid fitting
            return self._fit_generalized_ellipsoid(obj, data, a, var_a, var_cen, fix_eps, init_parameters, upper_bounds, lower_bounds, info)

    def _fit_standard_ellipsoid(self, obj: "Gal3DAnalyzer", data: dict, a: float, var_a: float, var_cen: float | None, init_parameters: dict, upper_bounds: dict, lower_bounds: dict, info: dict) -> ModelResult:
        """
        Fit a standard ellipsoidal structure, or fit a generalized ellipsoid only once.
        """

        # Set up parameters
        parameters_set = obj.structure.parameters.new()
        if "info" in data:
            parameters_set.add_info(**data["info"])
            del data["info"]

        # Set bounds and initial values
        parameters_set.set_lb(a=(a * (1 - var_a)))
        parameters_set.set_ub(a=(a * (1 + var_a)))
        if var_cen is not None:
            var_cen = min(max(var_cen, 0), 0.99)
            cen = var_cen * a
            for name in ["x", "y", "z"]:
                if name in parameters_set.parameter_keys():
                    parameters_set.get_parameter(name).assign_bounds(-cen, cen)

        # Apply custom bounds using set operations for efficiency
        for param_name in set(upper_bounds) & set(parameters_set.keys()):
            parameters_set.get_parameter(param_name).ub = upper_bounds[param_name]
        for param_name in set(lower_bounds) & set(parameters_set.keys()):
            parameters_set.get_parameter(param_name).lb = lower_bounds[param_name]

        if init_parameters:
            parameters_set = parameters_set.set_value(**init_parameters)
        parameters_set.set_value(a=a)

        # Add info if available
        if info:
            parameters_set.add_info(**info)

        # Prepare optimization function and bounds
        parameters_set["eps_ab"] = max(parameters_set["eps_ab"],0.1)
        parameters_set["eps_bc"] = max(parameters_set["eps_bc"],0.1)

        # Run optimization
        op_res = obj.optimizer.fit(
            obj.structure._error_method, parameters_set, func_kwargs=data)

        # Set optimized values and return result
        for i,j in op_res.params.items():
            parameters_set[i] = j
        return ModelResult(obj.structure, op_res, parameters_set)

    def _fit_generalized_ellipsoid(self, obj: "Gal3DAnalyzer", data: dict, a: float, var_a: float, var_cen: float | None, fix_eps: bool, init_parameters: dict, upper_bounds: dict, lower_bounds: dict, info: dict) -> ModelResult:
        """
        Fit a generalized ellipsoidal structure (Ellipsoid_S) using a two-step approach.
        """

        # Step 1: Create standard ellipsoid for initial fitting
        ellipsoid = Structure3D(
            obj.structure._coordinate,
            "Ellipsoid",
            obj.structure._error_func_name,
            obj.structure._error_method_name,
        )

        # Reuse parameter setup logic
        ell_params = ellipsoid.parameters.new()
        self._setup_ellipsoid_parameters(ell_params, obj.structure.parameters, a, var_a, var_cen)

        # Apply parameter bounds
        self._apply_parameter_bounds(ell_params, upper_bounds, lower_bounds)

        # Set initial values
        if init_parameters:
            shared_keys = set(init_parameters) & set(ell_params.keys())
            if shared_keys:
                update_values = {k: init_parameters[k] for k in shared_keys}
                ell_params = ell_params.set_value(**update_values)
        ell_params.set_value(a=a)

        # Run first optimization
        ell_res = obj.optimizer.fit(
            ellipsoid._error_method, ell_params, func_kwargs=data)

        # Step 2: Use ellipsoid results to initialize the generalized ellipsoid fit
        res_value = ell_res.params

        # Set up parameters for the generalized ellipsoid
        parameters_set = obj.structure.parameters.new()
        self._setup_ellipsoid_parameters(parameters_set, obj.structure.parameters, a, var_a, var_cen)
        self._apply_parameter_bounds(parameters_set, upper_bounds, lower_bounds)
        #parameters_set.get_parameter("a").assign_bounds(a * (1 - var_a), a * (1 + var_a))

        # Apply initial values and lock shape parameters
        if init_parameters:
            updatevalue = {
                i: init_parameters[i]
                for i in init_parameters.keys() & parameters_set.keys()
            }
            parameters_set = parameters_set.set_value(**updatevalue)
            if "sa" in updatevalue:
                parameters_set["sa"] = min(max(updatevalue["sa"], 0.5), 1.9)
            if "sb" in updatevalue:
                parameters_set["sb"] = min(max(updatevalue["sb"], 0.5), 1.9)
            if "sc" in updatevalue:
                parameters_set["sc"] = min(max(updatevalue["sc"], 0.5), 1.9)

        for i,j in res_value.items():
            parameters_set.get_parameter(i).assign_value(j)

        shape_params, fixed_name = self._prepare_shape_parameters(parameters_set,res_value,fix_eps)

        # Record parent function result
        # parameters_set.add_info(parent_fun=ell_res.fun)

        # Add original info if available
        if info:
            parameters_set.add_info(**info)

        # Prepare final optimization
        if "a" not in shape_params:
            parameters_set.get_parameter("a").assign_value(res_value["a"])
        if not fix_eps:
            parameters_set.get_parameter("eps_ab").assign_value(res_value["eps_ab"])
            parameters_set.get_parameter("eps_bc").assign_value(res_value["eps_bc"])

        # Run final optimization
        op_res = obj.optimizer.fit(
            obj.structure._error_method, parameters_set, func_kwargs=data
        )
        for i in fixed_name:
            parameters_set.del_equal_constraints(i)
        for i,j in ell_res.params.items():
            parameters_set[i] = j
        # Update parameters and return result
        for i,j in op_res.params.items():
            parameters_set[i] = j

        return ModelResult(obj.structure, op_res, parameters_set)

    def _setup_ellipsoid_parameters(self, ell_params: "Parameters", source_params: "Parameters", a: float, var_a: float, var_cen: float | None) -> None:
        """Set up parameter bounds for the ellipsoid"""
        # Copy parameter bounds from source
        for param_name in set(source_params.keys()) & set(ell_params.keys()):
            ell_params.get_parameter(param_name).assign_bounds(
                source_params.get_parameter(param_name).lb,
                source_params.get_parameter(param_name).ub
            )
        for param_name in set(source_params.constraint_keys()) & set(ell_params.keys()):
            ell_params.add_equal_constraints(**{param_name: source_params._equal_constraints[param_name]})

        # Set semi-major axis bounds
        ell_params.get_parameter("a").assign_bounds(a * (1 - var_a), a * (1 + var_a))

        # Set center bounds if needed
        if var_cen is not None:
            var_cen = min(max(var_cen, 0), 0.99)
            if var_cen > 0:
                cen = var_cen * a
                for name in ["x", "y", "z"]:
                    if name in ell_params.parameter_keys():
                        ell_params.get_parameter(name).assign_bounds(-cen, cen)
            else:
                for name in ["x", "y", "z"]:
                    if name in ell_params.parameter_keys():
                        ell_params.fix_parameters(**{name: 0.})

    def _apply_parameter_bounds(self, params: "Parameters", upper_bounds: dict[str, float], lower_bounds: dict[str, float]) -> None:
        """Apply upper and lower bounds to parameters"""
        for param_name in set(upper_bounds) & set(params.keys()):
            params.get_parameter(param_name).ub = upper_bounds[param_name]
        for param_name in set(lower_bounds) & set(params.keys()):
            params.get_parameter(param_name).lb = lower_bounds[param_name]

    def _prepare_shape_parameters(self, parameters_set: "Parameters", res_value: dict[str, float], fix_eps: bool) -> tuple[dict[str, float], list[str]]:
        """Prepare shape parameters and constraints for the final fit"""
        shape_params = res_value.copy()
        fixed_params = []

        # Keep 'a' free to vary
        if "a" in shape_params:
            parameters_set["a"] = shape_params["a"]
            del shape_params["a"]

        # Keep eps parameters free if not fixing them
        if not fix_eps:
            if "eps_ab" in shape_params:
                del shape_params["eps_ab"]
            if "eps_bc" in shape_params:
                del shape_params["eps_bc"]

        # Set up shape parameter bounds and constraints
        if shape_params:
        #    parameters_set.set_lb(only_infs=False, **shape_params)
        #    parameters_set.set_ub(only_infs=False, **shape_params)
            for param_name in shape_params.keys() & parameters_set.keys():
                parameters_set.fix_parameters(**{param_name: shape_params[param_name]})
                fixed_params.append(param_name)

        return shape_params, fixed_params
