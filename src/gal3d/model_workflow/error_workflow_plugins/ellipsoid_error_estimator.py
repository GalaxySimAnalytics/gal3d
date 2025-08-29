"""
Spherical harmonics-based error estimation workflow, estimator for Ellipsoidal shapes.
"""
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
from scipy.optimize import leastsq

from gal3d.field.spherical_field.spherical_vector import SphVector
from gal3d.model_workflow.error_workflow import ErrorWorkflowBase
from gal3d.optimization.result import ModelResult
from gal3d.shape import StructureCore
from gal3d.shape.with_parameter import WithParameter

if TYPE_CHECKING:
    from gal3d.optimization.parameter import Parameters

__all__ = ["EllipsoidErrorEstimator"]


class SphericalHarmonicsFitter:
    """Handles spherical harmonics calculation and fitting."""

    @staticmethod
    def first_and_second_harmonic_function(pos: np.ndarray, coef: np.ndarray) -> np.ndarray:
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        r = np.sqrt(np.sum(pos*pos, axis=1))
        c0 = coef[0]
        c1 = coef[1]*x
        c2 = coef[2]*y
        c3 = coef[3]*z
        c4 = coef[4]*(x*x-y*y)/(r*r)
        c5 = coef[5]*(2*z*z-x*x-y*y)/(r*r)
        c6 = coef[6]*(x*y)/(r*r)
        c7 = coef[7]*(y*z)/(r*r)
        c8 = coef[8]*(z*x)/(r*r)
        return c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8

    @staticmethod
    def fit_spherical_harmonics(pos: np.ndarray, weight: np.ndarray)-> dict[int, np.ndarray]:
        def optimize_func(x):
            return SphericalHarmonicsFitter.first_and_second_harmonic_function(pos, x) - weight
        def res_to_coef(res):
            return {0: res[0], 1: res[1:4], 2: res[4:]}
        res = leastsq(optimize_func, [np.mean(weight), 1, 1, 1, 1, 1, 1, 1, 1])
        return res_to_coef(res[0])

    @staticmethod
    def decompose(pos: np.ndarray, w: np.ndarray, from_fit: bool = True) -> dict[int, np.ndarray]:
        if from_fit:
            return SphericalHarmonicsFitter.fit_spherical_harmonics(pos, w)
        c0 = 1.
        #c0 = 0.5/np.sqrt(np.pi)
        coef: dict[int, np.ndarray] = {}
        r = np.sqrt(np.sum(pos*pos, axis=1))
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        coef[0] = c0/len(w)*np.array(np.sum(w))

        c1 = 1.
        # c1 = np.sqrt(3)*c0
        coef[1] = c1/len(w)*np.sum(pos.T/r*w, axis=1)

        c2 = 1.
        #c2 = np.sqrt(15)*c0
        coef[2] = c2/len(w)*np.sum(w*np.array([(x*x-y*y), 2*z*z-x*x-y*y, x*y, y*z, z*x])/(r*r), axis=1)
        #coef[2][:2] = coef[2][:2]/2
        return coef

_modeType = Literal["fitted", "lin", "exp"]

class EllipsoidParameterUpdater:
    """Updates parameters for ellipsoidal shapes."""

    @staticmethod
    def get_update(parameter_name: str, coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float | np.ndarray:

        return getattr(EllipsoidParameterUpdater, f"update_{parameter_name}")(coef, parameters, f_d, mode)

    @staticmethod
    def update_a(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float | np.ndarray:
        if mode == "lin":
            delta = coef[0]/f_d[0]
        elif mode == "exp":
            delta = parameters["a"]*(np.exp(coef[0]/(parameters["a"]*f_d[0])) - 1)
        else:
            delta = coef[0]/f_d[0]
        return delta

    @staticmethod
    def update_x(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode=="lin":
            return coef[1][0]/f_d[0]
        else:
            return coef[1][0]/f_d[0]/(1-parameters["eps_ab"]**2)/(1-parameters["eps_bc"]**2)

    @staticmethod
    def update_y(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode=="lin":
            return coef[1][1]/f_d[1]
        else:
            return coef[1][1]/f_d[1]/(1-parameters["eps_bc"]**4)

    @staticmethod
    def update_z(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode=="lin":
            return coef[1][2]/f_d[2]
        else:
            return coef[1][2]/f_d[2]/(1+parameters["eps_bc"]**3)

    @staticmethod
    def update_pos(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        return coef[1]/f_d

    @staticmethod
    def update_eps_ab(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode == "lin":
            return coef[2][0]/f_d[1]/parameters["a"]
        elif mode == "exp":
            return 2*(math.exp(coef[2][0]/((1-parameters["eps_ab"])*parameters["a"])/f_d[1])-1) * (1-parameters["eps_ab"])
        else:
            delta_0 = 2*(math.exp(coef[2][0]/((1-parameters["eps_ab"])*parameters["a"])/f_d[1])-1) * (1-parameters["eps_ab"])
            c1 = ((1+parameters["eps_ab"]**2)/(1-parameters["eps_ab"]**2))**0.5
            c2 = ((1+parameters["eps_bc"]**2)/(1-parameters["eps_bc"]**2))**0.5
            return c1*c2*delta_0

    @staticmethod
    def update_eps_bc(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        if mode == "lin":
            return -coef[2][1]/f_d[2]/(parameters["a"]*(1-parameters["eps_ab"]))
        elif mode == "exp":
            return 2*(math.exp(-coef[2][1]/((1-parameters["eps_bc"])*(parameters["a"]*(1-parameters["eps_ab"])))/f_d[2])-1) * (1-parameters["eps_bc"])
        else:
            delta_0 = 2*(math.exp(-coef[2][1]/((1-parameters["eps_bc"])*(parameters["a"]*(1-parameters["eps_ab"])))/f_d[2])-1) * (1-parameters["eps_bc"])
            c1 = math.sqrt((1+parameters["eps_ab"]**2)/(1-parameters["eps_ab"]**2))
            c2 = math.sqrt((1+parameters["eps_bc"]**2)/(1-parameters["eps_bc"]**2))
            return c1*c2*delta_0

    @staticmethod
    def update_ang1(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        eps_ab = parameters["eps_ab"]
        a = parameters["a"]
        dx = f_d[0]
        if eps_ab==0:
            return 0.
        return coef[2][2]/dx/a/((1-eps_ab)**2-1)/(1+eps_ab)*(1+parameters["eps_bc"])

    @staticmethod
    def update_angle(coef: dict[int, np.ndarray], parameters: "Parameters", f_d: np.ndarray, mode: _modeType = "fitted") -> float:
        eps_ab = parameters["eps_ab"]
        a = parameters["a"]
        dx = f_d[0]
        if eps_ab==0:
            return 0.
        return coef[2][2]/dx/a/((1-eps_ab)**2-1)/(1+eps_ab)*(1+parameters["eps_bc"])

class SphericalDecIterator:
    """Spherical decomposition iterator for ellipsoidal shapes."""
    def __init__(self, pos: np.ndarray, structure: StructureCore, *, n_sample: int = 64, from_fit: bool = True, **kwargs: Any):
        self.structure = structure
        self.sample_area = SphVector(n_sample=n_sample, verbose=False)
        self.ind = self._assign_pos(pos, self.sample_area, self.structure)
        self.sample_pos, self.sample_w, self.gradient = self._build_sample(pos, self.sample_area, self.structure)
        self.coef = self.compute_spherical_harmonics(self.sample_pos, self.sample_w, from_fit=from_fit)

    def _assign_pos(self, pos: np.ndarray, sample_area: SphVector, structure: StructureCore) -> list[list[np.ndarray]]:
        aligned_pos = structure.transform_pos(pos)
        index = sample_area.assign_points(aligned_pos)
        ind: list[list[np.ndarray]] = [[] for _ in range(sample_area.num)]
        for i, j in enumerate(index):
            ind[j].append(i)
        return ind

    def _build_sample(self, pos: np.ndarray, sample_area: SphVector, structure: StructureCore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        parameters = structure.parameters
        eye = np.eye(3)
        batch_pos = np.concatenate([pos, 2*eye + parameters["pos"], eye + parameters["pos"]], axis=0)
        batch_fray = structure.f_ray_d(batch_pos)
        error = batch_fray[:-6] - 1
        gradient = (batch_fray[-6:-3] - batch_fray[-3:])

        sample_w = np.nan * np.ones(sample_area.num)
        for i in range(len(self.ind)):
            if self.ind[i]:
                sample_w[i] = np.mean(error[self.ind[i]])

        sample_pos = sample_area.pos[~np.isnan(sample_w)]
        sample_w = sample_w[~np.isnan(sample_w)]

        return sample_pos, sample_w, gradient

    def get_parameter_update(self, parameter_name: str, mode: _modeType = "fitted") -> float | np.ndarray:

        return EllipsoidParameterUpdater.get_update(parameter_name, self.coef, self.parameters, self.gradient, mode)

    def compute_spherical_harmonics(self, pos: np.ndarray, weight: np.ndarray, from_fit: bool = True) -> dict[int, np.ndarray]:
        return SphericalHarmonicsFitter.decompose(pos, weight, from_fit=from_fit)

    @property
    def parameters(self):
        return self.structure.parameters



class EllipsoidErrorEstimator(ErrorWorkflowBase):
    """
    Spherical harmonics-based error estimation workflow for ellipsoidal shapes.
    """

    @classmethod
    def condition(cls, result: Union["StructureCore", "WithParameter", "ModelResult"]) -> bool:
        """
        Check if this workflow can handle the given result.

        Parameters
        ----------
        result : Union[StructureCore, WithParameter, ModelResult]
            The result to check.

        Returns
        -------
        bool
            True if this workflow can handle the result, False otherwise.
        """
        if isinstance(result, ModelResult):
            geometry_name = result.structure.geometry_name
        elif isinstance(result, WithParameter):
            geometry_name = result.__class__.__name__
        elif isinstance(result, StructureCore):
            geometry_name = result.geometry_name

        if geometry_name in ["Ellipsoid", "Ellipsoid_S"]:
            return True
        else:
            return False

    @classmethod
    def estimate_error(cls, result: Union[StructureCore, "WithParameter", "ModelResult"], **kwargs: Any) -> dict[str, np.ndarray]:
        if not isinstance(result, ModelResult):
            res = cls.estimate_structure_update(result, **kwargs)
            ret: dict[str, np.ndarray] = {}
            for i in res.keys():
                ret[i] = np.abs(res[i])
            return ret
        else:
            res = cls.estimate_model_update(result, **kwargs)
            ret = {}
            for i in res.keys():
                ret[i] = np.abs(res[i])
            return ret

    @classmethod
    def estimate_structure_update(cls, result: Union[StructureCore, "WithParameter"], pos: np.ndarray, **kwargs: Any) -> dict[str, np.ndarray]:
        if not cls.condition(result):
            raise ValueError("Incompatible result type.")

        structure: StructureCore
        estimator_key: list[str]
        if isinstance(result, StructureCore):
            structure = result
            estimator_key = ["eps_ab","eps_bc","a","x","y","z","ang1"]
        else:
            structure = StructureCore("ShiftEuler", result.__class__.__name__)
            structure.parameters.set_value(**result.parameters)
            estimator_key = ["eps_ab","eps_bc","a"]

        iterator = SphericalDecIterator(pos, structure, **kwargs)
        res = {i: np.array(iterator.get_parameter_update(i, kwargs.get("mode","fitted"))) for i in estimator_key}
        return res

    @classmethod
    def estimate_model_update(cls, result: ModelResult, **kwargs: Any) -> dict[str, np.ndarray]:

        res: defaultdict[str, list[float | np.ndarray]] = defaultdict(list)
        estimator_key: list[str] = ["eps_ab","eps_bc","a","x","y","z","ang1"]

        for i in range(len(result)):
            pos = result._param_sets[i].get_info("data")
            structure = result[i]
            iterator = SphericalDecIterator(pos, structure,**kwargs)
            for j in estimator_key:
                res[j].append(iterator.get_parameter_update(j, kwargs.get("mode","fitted")))

        ret: dict[str, np.ndarray] = {k: np.array(v) for k, v in res.items()}
        return ret
