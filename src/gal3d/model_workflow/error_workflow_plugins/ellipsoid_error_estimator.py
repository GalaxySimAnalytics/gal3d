"""
Spherical harmonics-based error estimation workflow, estimator for Ellipsoidal shapes.
This module still under development.
"""
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import numpy as np
from scipy.optimize import leastsq
from tqdm import tqdm

from gal3d.field.spherical_field.spherical_vector import SphVector
from gal3d.model_workflow.error_workflow import ErrorWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.shape import StructureCore
from gal3d.util.func_decorator import development_warning

if TYPE_CHECKING:
    from gal3d.optimization.parameter import Parameters

__all__ = ["EllipsoidErrorEstimator"]


class SphericalHarmonicsFitter:
    """Handles spherical harmonics calculation and fitting."""

    @staticmethod
    def harmonic_function(pos: np.ndarray, coef: np.ndarray, fourth: bool = False) -> np.ndarray:
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        x2 = x*x
        y2 = y*y
        z2 = z*z
        r2 = x2 + y2 + z2
        r = np.sqrt(r2)
        r2_inv = 1.0/r2

        c0 = coef[0]
        c123 = (coef[1]*x+coef[2]*y+coef[3]*z)

        c4 = coef[4]*(x2-y2)
        c5 = coef[5]*(2*z2-x2-y2)
        c6 = coef[6]*(x*y)
        c7 = coef[7]*(y*z)
        c8 = coef[8]*(z*x)
        res = c0 + c123/r + (c4 + c5 + c6 + c7 + c8)*r2_inv
        if fourth:
            c9 = coef[9]*(x2*x2-6*x2*y2+y2*y2) # C44
            c10 = coef[10]*(x2-y2)*(7*z2-r2) # C42
            c11 = coef[11]*(35*z2*z2-30*z2*r2+3*r2*r2) # C40
            res = res + (c9 + c10 + c11)*r2_inv*r2_inv
        return res

    @staticmethod
    def fit_spherical_harmonics(pos: np.ndarray, weight: np.ndarray, fourth: bool = False) -> dict[str, np.ndarray]:
        def optimize_func(x):
            return SphericalHarmonicsFitter.harmonic_function(pos, x, fourth) - weight
        def res_to_coef(res):
            coef = {}
            coef["a"] = res[0]
            coef["x"], coef["y"], coef["z"] = res[1:4]
            coef["eps_ab"], coef["eps_bc"], coef["ang1"], coef["ang2"],coef["ang3"]= res[4:9]
            if fourth:
                coef["s_ab_plus"],coef["s_ab_minus"],coef["s_abc"] = res[9:]
            return coef
        x0 = [np.mean(weight), 1, 1, 1, 1, 1, 1, 1, 1]
        if fourth:
            x0.extend([1, 1, 1])
        res = leastsq(optimize_func, x0)
        return res_to_coef(res[0])

    @staticmethod
    def decompose(pos: np.ndarray, w: np.ndarray, from_fit: bool = True, fourth: bool = False) -> dict[str, np.ndarray]:
        if from_fit:
            return SphericalHarmonicsFitter.fit_spherical_harmonics(pos, w, fourth)
        #c0 = 0.5/np.sqrt(np.pi)
        coef: dict[str, np.ndarray] = {}
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        x2 = x*x
        y2 = y*y
        z2 = z*z
        r2 = x2 + y2 + z2
        r = np.sqrt(r2)
        r_inv = 1.0/r
        r2_inv = 1.0/r2
        # Use direct calculation for coefficients
        coef = {}
        coef["a"] = np.mean(w)
        coef["x"] = np.mean(w * x * r_inv)
        coef["y"] = np.mean(w * y * r_inv)
        coef["z"] = np.mean(w * z * r_inv)

        coef["eps_ab"] = np.mean(w * (x2 - y2) * r2_inv)
        coef["eps_bc"] = np.mean(w * (2*z2 - x2 - y2) * r2_inv)
        coef["ang1"] = np.mean(w * (x*y) * r2_inv)
        coef["ang2"] = np.mean(w * (y*z) * r2_inv)
        coef["ang3"] = np.mean(w * (z*x) * r2_inv)

        if fourth:
            coef["s_ab_plus"] = np.mean(w * (x2*x2 - 6*x2*y2 + y2*y2) * r2_inv * r2_inv)
            coef["s_ab_minus"] = np.mean(w * (x2 - y2) * (7*z2 - r2) * r2_inv * r2_inv)
            coef["s_abc"] = np.mean(w * ((35*z2*z2 - 30*z2*r2) * r2_inv * r2_inv + 3.))

        return coef

_modeType = Literal["fitted", "lin", "exp"]

class EllipsoidParameterUpdater:
    """Updates parameters for ellipsoidal shapes."""

    @staticmethod
    def get_update(parameter_name: str, coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float | np.ndarray:

        return getattr(EllipsoidParameterUpdater, f"update_{parameter_name}")(coef, parameters, f_d, mode)

    @staticmethod
    def update_a(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float | np.ndarray:
        if mode == "lin":
            delta = coef["a"]/f_d[0]
        elif mode == "exp":
            delta = parameters["a"]*(math.exp(coef["a"]/(parameters["a"]*f_d[0])) - 1)
        else:
            delta = coef["a"]/f_d[0]
        return delta

    @staticmethod
    def update_x(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode=="lin":
            return coef["x"]/f_d[0]
        else:
            return coef["x"]/f_d[0]/(1-parameters["eps_ab"]**2)/(1-parameters["eps_bc"]**2)

    @staticmethod
    def update_y(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode=="lin":
            return coef["y"]/f_d[1]
        else:
            return coef["y"]/f_d[1]/(1-parameters["eps_bc"]**4)

    @staticmethod
    def update_z(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode=="lin":
            return coef["z"]/f_d[2]
        else:
            return coef["z"]/f_d[2]/(1+parameters["eps_bc"]**3)

    @staticmethod
    def update_pos(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        return np.array([coef["x"], coef["y"], coef["z"]]) / f_d

    @staticmethod
    def update_eps_ab(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        co = coef["eps_ab"]
        df = f_d[1]
        eps_ab = parameters["eps_ab"]
        eps_bc = parameters["eps_bc"]
        a = parameters["a"]
        if mode == "lin":
            return co/df/a
        elif mode == "exp":
            return 2*(math.exp(co/((1-eps_ab)*a)/df)-1) * (1-eps_ab)
        else:
            delta_0 = 2*(math.exp(co/((1-eps_ab)*a)/df)-1) * (1-eps_ab)
            c1 = math.sqrt((1+eps_ab**2)/(1-eps_ab**2))
            c2 = math.sqrt((1+eps_bc**2)/(1-eps_bc**2))
            return c1*c2*delta_0

    @staticmethod
    def update_eps_bc(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        co = coef["eps_bc"]
        df = f_d[2]
        eps_ab = parameters["eps_ab"]
        eps_bc = parameters["eps_bc"]
        a = parameters["a"]
        if mode == "lin":
            return -co/df/(a*(1-eps_ab))
        elif mode == "exp":
            return 2*(math.exp(-co/((1-eps_bc)*(a*(1-eps_ab)))/df)-1) * (1-eps_bc)
        else:
            delta_0 = 2*(math.exp(-co/((1-eps_bc)*(a*(1-eps_ab)))/df)-1) * (1-eps_bc)
            c1 = math.sqrt((1+eps_ab**2)/(1-eps_ab**2))
            c2 = math.sqrt((1+eps_bc**2)/(1-eps_bc**2))
            return c1*c2*delta_0

    @staticmethod
    def update_ang1(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        co = coef["ang1"]
        df = f_d[0]
        eps_ab = parameters["eps_ab"]
        eps_bc = parameters["eps_bc"]
        a = parameters["a"]
        if eps_ab==0:
            return 0.
        if mode == "lin":
            return co/df/a/((1-eps_ab)**2-1)*(1-eps_ab)*(1+eps_bc)

        return co/df/a/((1-eps_ab)**2-1)/(1+eps_ab)*(1+eps_bc)

    @staticmethod
    def update_angle(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        eps_ab = parameters["eps_ab"]
        a = parameters["a"]
        dx = f_d[0]
        if eps_ab==0:
            return 0.
        return coef["ang1"]/dx/a/((1-eps_ab)**2-1)/(1+eps_ab)*(1+parameters["eps_bc"])

    @staticmethod
    def update_sabc(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        ite_sabc = (np.exp(-4*coef["s_abc"]/f_d[0])-1)*(parameters["sa"]+parameters["sb"]+parameters["sc"])/3
        return ite_sabc

    @staticmethod
    def update_sab_plus(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        ite_sab_plus = (np.exp(-4*coef["s_ab_plus"]/f_d[0])-1)*(parameters["sa"]+parameters["sb"])/2
        return ite_sab_plus

    @staticmethod
    def update_sab_minus(coef: dict[str, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        ite_sab_minus = -(np.exp(-4*coef["s_ab_minus"]/f_d[0])-1)*(parameters["sa"]+parameters["sb"])/2
        return ite_sab_minus


class SphericalDecIterator:
    """Spherical decomposition iterator for ellipsoidal shapes."""
    def __init__(self, pos: np.ndarray, structure: StructureCore, *, n_sample: int = 64, from_fit: bool = True, fourth: bool = False, sigma_clip: float = 3.0, **kwargs: Any):
        self.structure = structure

        self.pos = pos  # Store original position data
        self.from_fit = from_fit
        self.fourth = fourth
        self.sigma_clip = sigma_clip
        self.kwargs = kwargs

        self.sample_area = SphVector(n_sample=n_sample, verbose=False)
        self.ind = self._assign_pos(pos, self.sample_area, self.structure)
        self.sample_pos, self.sample_w, self.gradient = self._build_sample(pos, self.sample_area, self.structure, sigma_clip=sigma_clip)
        self.coef = self.compute_spherical_harmonics(self.sample_pos, self.sample_w, from_fit=from_fit, fourth=fourth)

    def _assign_pos(self, pos: np.ndarray, sample_area: SphVector, structure: StructureCore) -> list[list[np.ndarray]]:
        aligned_pos = structure.transform_pos(pos)
        index = sample_area.assign_points(aligned_pos)
        ind: list[list[np.ndarray]] = [[] for _ in range(sample_area.num)]
        for i, j in enumerate(index):
            ind[j].append(i)
        return ind

    def _build_sample(self, pos: np.ndarray, sample_area: SphVector, structure: StructureCore, sigma_clip: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


        grad_points = np.array([[2, 0., 0.],
                                 [0., 2, 0.],
                                 [0., 0., 2],
                                 [1, 0., 0.],
                                 [0., 1, 0.],
                                 [0., 0., 1]])  # Add the last gradient point
        grad_points = structure.inverse_transform(grad_points)
        batch_pos = np.concatenate([pos, grad_points], axis=0)
        batch_fray = structure.f_ray_d(batch_pos)
        error = batch_fray[:-6] - 1
        gradient = (batch_fray[-6:-3] - batch_fray[-3:])
        gradient = np.ones(3)*np.mean(gradient)

        # Apply sigma clipping to the entire error array once
        if sigma_clip > 0:
            mean_error = np.mean(error)
            std_error = np.std(error)
            mask = np.abs(error - mean_error) <= sigma_clip * std_error
            # Create a filtered error array
            filtered_error = np.where(mask, error, np.nan)
        else:
            # No clipping if sigma_clip <= 0
            filtered_error = error

        # Pre-allocate sample weights array
        sample_w = np.full(sample_area.num, np.nan)

        # Calculate mean for each sample area using the filtered errors
        for i in range(len(self.ind)):
            if self.ind[i]:
                # Get filtered errors for this sample area
                valid_errors = filtered_error[self.ind[i]]
                valid_errors = valid_errors[~np.isnan(valid_errors)]
                if np.any(valid_errors):
                    sample_w[i] = np.mean(valid_errors)
                # If no valid errors (all were clipped), keep as NaN

        # Filter out positions with NaN weights
        valid_mask = ~np.isnan(sample_w)
        sample_pos = sample_area.pos[valid_mask]
        sample_w = sample_w[valid_mask]

        return sample_pos, sample_w, gradient

    def update_parameter(self, param_name: str, new_value: float) -> None:
        """
        Update a parameter and recalculate only what's needed.

        Parameters
        ----------
        param_name : str
            Name of the parameter to update
        new_value : float
            New value for the parameter
        """

        # Update parameter
        self.structure.parameters[param_name] = new_value

        # Determine what needs to be recalculated based on parameter type
        if param_name in ["x", "y", "z", "ang1"]:
            # Position or orientation parameters require full recalculation of point assignments
            self.ind = self._assign_pos(self.pos, self.sample_area, self.structure)
            self.sample_pos, self.sample_w, self.gradient = self._build_sample(
                self.pos, self.sample_area, self.structure, sigma_clip=self.sigma_clip
            )
        else:
            # For other parameters (a, eps_ab, eps_bc), only recalculate errors and weights
            # No need to reassign positions
            self.sample_pos, self.sample_w, self.gradient = self._build_sample(
                self.pos, self.sample_area, self.structure, sigma_clip=self.sigma_clip
            )

        # Update coefficients based on new sample positions and weights
        self.coef = self.compute_spherical_harmonics(
            self.sample_pos, self.sample_w, from_fit=self.from_fit, fourth=self.fourth
        )

    def get_parameter_update(self, parameter_name: str, mode: _modeType = "fitted") -> float | np.ndarray:

        return EllipsoidParameterUpdater.get_update(parameter_name, self.coef, self.parameters, self.gradient, mode)

    def compute_spherical_harmonics(self, pos: np.ndarray, weight: np.ndarray, from_fit: bool = True, fourth: bool = False) -> dict[str, np.ndarray]:
        return SphericalHarmonicsFitter.decompose(pos, weight, from_fit=from_fit, fourth=fourth)

    @property
    def parameters(self):
        return self.structure.parameters



class EllipsoidErrorEstimator(ErrorWorkflowBase):
    """
    Spherical harmonics-based error estimation workflow for ellipsoidal shapes.
    """

    @classmethod
    def condition(cls, result: Union["StructureCore", "ModelResult"]) -> bool:
        """
        Check if this workflow can handle the given result.

        Parameters
        ----------
        result : Union[StructureCore, ModelResult]
            The result to check.

        Returns
        -------
        bool
            True if this workflow can handle the result, False otherwise.
        """
        if isinstance(result, ModelResult):
            geometry_name = result.structure.geometry_name
        elif isinstance(result, StructureCore):
            geometry_name = result.geometry_name

        if geometry_name in ["Ellipsoid", "Ellipsoid_S"]:
            return True
        else:
            return False

    @development_warning("EllipsoidErrorEstimator.estimate_error is under development and may change in future versions.")
    @classmethod
    def estimate_error(cls, result: Union[StructureCore, "ModelResult"], param_name: list[str] | None = None, *,max_iter: int = 10, **kwargs: Any) -> dict[str, Any]:

        estimator_key = ["eps_ab","eps_bc","a","x","y","z","ang1"] if param_name is None else param_name
        if not isinstance(result, ModelResult):
            # For single structure case - optimize by computing base_iterator only once
            pos = kwargs.get("pos", None)
            if pos is None:
                raise ValueError("Position data required for structure error estimation")

            # Create base_iterator once
            base_iterator = SphericalDecIterator(pos, result, **kwargs)

            # Get initial updates for all parameters at once
            ret: dict[str, Any] = {}
            for i in estimator_key:
                ret[i] = cls._estimate_structure_error_single(result, pos, i, base_iterator,
                                                            max_iter=max_iter, **kwargs)
            return ret
        else:
            # For model result case (multiple structures)
            ret_list = defaultdict(list)

            # Process each structure only once, with optimized parameter handling
            for i in tqdm(range(len(result))):
                pos = cast("np.ndarray", result.get("data", index=i))
                structure = result[i]

                # Create iterator once per structure
                base_iterator = SphericalDecIterator(pos, structure, **kwargs)

                # Process all parameters for this structure
                structure_results = {}
                for j in estimator_key:
                    structure_results[j] = cls._estimate_structure_error_single(structure, pos, j,
                                                                            base_iterator,
                                                                            max_iter=max_iter, **kwargs)

                # Collect results
                for key, value in structure_results.items():
                    ret_list[key].append(value)

            # Convert to numpy arrays
            ret = {i: np.array(ret_list[i]) for i in ret_list}
            return ret

    @development_warning("estimate_structure_update is under development and may change in future versions.")
    @classmethod
    def estimate_structure_update(cls, result: StructureCore, pos: np.ndarray,*, estimator_key: list[str] | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
        if not cls.condition(result):
            raise ValueError("Incompatible result type.")
        if estimator_key is None:
            estimator_key = cast("list[str]", ["eps_ab","eps_bc","a","x","y","z","ang1"])

        iterator = SphericalDecIterator(pos, result, **kwargs)
        res = {i: np.array(iterator.get_parameter_update(i, kwargs.get("mode","fitted"))) for i in estimator_key}
        return res

    @classmethod
    def _estimate_structure_error_single(cls, result: StructureCore, pos: np.ndarray,
                                    param_name: str, base_iterator: SphericalDecIterator,
                                    *, max_iter: int = 10, **kwargs: Any) -> float:
        """
        Optimized method to estimate error for a single parameter.

        Parameters
        ----------
        result : StructureCore
            The structure to update
        pos : np.ndarray
            Position data
        param_name : str
            Parameter name to update
        base_iterator : SphericalDecIterator
            Pre-computed base iterator
        max_iter : int
            Maximum number of iterations

        Returns
        -------
        np.ndarray
            The estimated error for the parameter
        """
        # Set mode and max update limit based on parameter type
        mode = kwargs.get("mode", "fitted")
        if param_name in ["eps_ab","eps_bc"]:
            res_max = (1.-result.parameters[param_name])*0.5
            res_min = res_max - 0.5
        elif param_name in ["x","y","z","a"]:
            res_max = 0.1*result.parameters["a"]
            res_min = -res_max
        else:
            res_max = 0.3
            res_min = -res_max

        # Create a copy only once
        res_test = result.copy()
        original_value = res_test.parameters[param_name]

        # Create an iterator for this structure copy - will be updated incrementally
        iterator = SphericalDecIterator(pos, res_test, **kwargs)

       # Save initial coefficient for comparison
        initial_coef = iterator.coef[param_name]
        prev_coef_magnitude = abs(initial_coef)

        # Keep track of best parameter value found
        best_value = original_value
        best_coef_magnitude = prev_coef_magnitude

        # Adaptive iteration control
        step_scale = 1.0
        convergence_threshold = 0.02  # 2% improvement threshold

        prev_res: float = 0.
        # Iterative refinement with early termination checks
        for j in range(max_iter):
            res = cast("float", iterator.get_parameter_update(param_name, mode))

            # Apply adaptive step scaling
            if j > 0 and prev_res * res < 0:
                # If direction reversed, take smaller steps
                step_scale *= 0.5

            res = np.clip(res * step_scale, res_min, res_max)
            prev_res = res

            # Store current parameter value before update
            current_value = res_test.parameters[param_name]

            # Apply update
            new_value = current_value + res

            # Skip update if too small
            if abs(res) < 1e-10 * abs(current_value):
                break

            # Update parameter incrementally
            iterator.update_parameter(param_name, new_value)

            # Check result
            curr_coef = iterator.coef[param_name]
            curr_coef_magnitude = abs(curr_coef)

            # Track best value
            if curr_coef_magnitude < best_coef_magnitude:
                best_value = new_value
                best_coef_magnitude = curr_coef_magnitude

                # If improvement is very good, scale up step size
                if curr_coef_magnitude < 0.5 * prev_coef_magnitude:
                    step_scale = min(step_scale * 1.2, 1.0)

            # Early termination checks
            if (curr_coef * initial_coef < 0 or  # Changed sign
                curr_coef_magnitude < 0.05 * abs(base_iterator.coef[param_name]) or  # Good enough
                curr_coef_magnitude > prev_coef_magnitude or  # Getting worse
                (j > 2 and abs(curr_coef_magnitude - prev_coef_magnitude) / prev_coef_magnitude < convergence_threshold)):  # No significant improvement

                break

            # Update previous magnitude for next iteration
            prev_coef_magnitude = curr_coef_magnitude

        # Return parameter change
        return best_value - original_value

    @development_warning("estimate_model_update is under development and may change in future versions.")
    @classmethod
    def estimate_model_update(cls, result: ModelResult,*, estimator_key: list[str] | None = None, **kwargs: Any) -> dict[str, np.ndarray]:

        res: defaultdict[str, list[float | np.ndarray]] = defaultdict(list)
        if estimator_key is None:
            estimator_key = cast("list[str]", ["eps_ab","eps_bc","a","x","y","z","ang1"])

        for i in range(len(result)):
            pos = cast("np.ndarray", result.get("data", index=i))
            structure = result[i]
            iterator = SphericalDecIterator(pos, structure,**kwargs)
            for j in estimator_key:
                res[j].append(iterator.get_parameter_update(j, kwargs.get("mode","fitted")))

        ret: dict[str, np.ndarray] = {k: np.array(v) for k, v in res.items()}
        return ret
