"""
This is a spherical harmonics decomposition iterator for fitting generalized ellipsoids to point clouds.
But it is still in experimental stage.
If it does not work well, it may be used to estimate the error of model.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import leastsq

from gal3d.shape import StructureCore

from .spherical_vector import SphVector


# ------------------ Spherical Harmonics Fitter ------------------
class SphericalHarmonicsFitter:
    """Handles spherical harmonics calculation and fitting."""

    @staticmethod
    def first_and_second_harmonic_function(pos, coef):
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
    def fit_spherical_harmonics(pos, weight):
        def optimize_func(x):
            return SphericalHarmonicsFitter.first_and_second_harmonic_function(pos, x) - weight
        def res_to_coef(res):
            return {0: res[0], 1: res[1:4], 2: res[4:]}
        res = leastsq(optimize_func, [np.mean(weight), 1, 1, 1,  1, 1, 1, 1, 1])
        return res_to_coef(res[0])

    @staticmethod
    def decompose(pos: np.ndarray, w: np.ndarray, lmax: int = 4, from_fit: bool = True) -> dict:
        if from_fit:
            return SphericalHarmonicsFitter.fit_spherical_harmonics(pos, w)
        c0 = 1.
        #c0 = 0.5/np.sqrt(np.pi)
        coef: dict[int, np.ndarray] = {}
        r = np.sqrt(np.sum(pos*pos, axis=1))
        x, y, z = pos[:,0], pos[:,1], pos[:,2]
        coef[0] = c0/len(w)*np.array(np.sum(w))
        if lmax == 0:
            return coef
        c1 = 1.
        # c1 = np.sqrt(3)*c0
        coef[1] = c1/len(w)*np.sum(pos.T/r*w, axis=1)
        if lmax == 1:
            return coef
        c2 = 1.
        #c2 = np.sqrt(15)*c0
        coef[2] = c2/len(w)*np.sum(w*np.array([(x*x-y*y), 2*z*z-x*x-y*y, x*y, y*z, z*x])/(r*r), axis=1)
        coef[2][:2] = coef[2][:2]/2
        return coef

# ------------------ Ellipsoid Parameter Updater ------------------
class EllipsoidParameterUpdater:
    """Handles parameter update logic for ellipsoid fitting."""

    @staticmethod
    def update_a(coef, parameters, f_d, fixed_f_d=False):
        if fixed_f_d:
            delta = coef[0]/f_d[0]
        else:
            delta = parameters["a"]*(np.exp(coef[0]/(parameters["a"]*f_d[0])) - 1)
        return delta

    @staticmethod
    def update_center(coef, f_d):
        return coef[1]/f_d

    @staticmethod
    def update_eps_ab(coef, parameters, f_d, mode="fitted"):
        if mode == "lin":
            return coef[2][0]/f_d[1]/parameters["a"]
        elif mode == "exp":
            return 2*(math.exp(coef[2][0]/((1-parameters["eps_ab"])*parameters["a"])/f_d[1])-1) * (1-parameters["eps_ab"])
        else:
            delta_0 = 2*(math.exp(coef[2][0]/((1-parameters["eps_ab"])*parameters["a"])/f_d[1])-1) * (1-parameters["eps_ab"])
            x = parameters["eps_ab"]+(2+parameters["eps_ab"]*2)*delta_0
            if x <= 0 or x >=1:
                x = parameters["eps_ab"]
            coef_val = math.acos(x)
            if coef_val <= 0:
                coef_val = 1
            coef_val = math.sqrt((1+parameters["eps_ab"])/(1-parameters["eps_ab"])/coef_val)
            return delta_0*coef_val*(1+3*parameters["eps_bc"]**2)

    @staticmethod
    def update_eps_bc(coef, parameters, f_d, mode="fitted"):
        if mode == "lin":
            return -coef[2][1]/f_d[2]/(parameters["a"]*(1-parameters["eps_ab"]))
        elif mode == "exp":
            return 2*(math.exp(-coef[2][1]/((1-parameters["eps_bc"])*(parameters["a"]*(1-parameters["eps_ab"])))/f_d[2])-1) * (1-parameters["eps_bc"])
        elif mode == "fitted":
            delta_0 = 2*(math.exp(-coef[2][1]/((1-parameters["eps_bc"])*(parameters["a"]*(1-parameters["eps_ab"])))/f_d[2])-1) * (1-parameters["eps_bc"])
            if delta_0 > 1 or delta_0 < -1:
                delta_0 = np.sign(delta_0)*0.2
            x = parameters["eps_bc"]+(1+parameters["eps_bc"])*delta_0
            if x <= 0 or x >=1:
                x = parameters["eps_bc"]
            coef_val = math.acos(x)
            if coef_val <= 0:
                coef_val = 1
            coef_val = math.sqrt(math.sqrt((1+parameters["eps_bc"]**2)/(1-parameters["eps_bc"]**2)/coef_val))
            return delta_0*coef_val*(1+3*parameters["eps_ab"]**2)

    @staticmethod
    def update_angle(coef, parameters, f_d):
        eps_ab = parameters["eps_ab"]
        a = parameters["a"]
        dx = f_d[0]
        delta = coef[2][2]/dx
        c0 = delta*(2-eps_ab*eps_ab)/(2+eps_ab*eps_ab-2*eps_ab)**(3/2)
        c1 = (2-eps_ab)*(2*a)*(1-eps_ab)*(np.sqrt(2-eps_ab*eps_ab))
        c2 = delta*eps_ab*(2+eps_ab*eps_ab-2*eps_ab)**(3/2)
        if eps_ab==0:
            delta_ang = 0.
        else:
            delta_ang = - c0/(c1+c2)/eps_ab
        return delta_ang

# ------------------ Main Iterator Class ------------------
class SphericalDecIterator:
    """
    Iterator for spherical harmonics decomposition and ellipsoid fitting.
    """

    def __init__(self, n_sample, coordinate="ShiftEuler"):
        self.sample_area = SphVector(n_sample=n_sample)
        self.structure = StructureCore(coordinate, "Ellipsoid_S")
        self.parameters = self.structure.parameters
        self.parameters["eps_ab"] = 0.01
        self.parameters["eps_bc"] = 0.01
        self.history = defaultdict(list)
        self.coef = None
        self.gradient = None
        self.sample_pos = None
        self.sample_w = None
        self._ind = None



    def spherical_harmonics_dec(self, pos: np.ndarray, w: np.ndarray, lmax: int = 4, from_fit: bool = True) -> dict:
        return SphericalHarmonicsFitter.decompose(pos, w, lmax, from_fit)


    def update_sample(self, pos, assign_ind=True):
        eye = np.eye(3)
        batch_pos = np.concatenate([pos, 2*eye + self.parameters["pos"], eye + self.parameters["pos"]], axis=0)
        batch_fray = self.structure.f_ray_d(batch_pos)
        error = batch_fray[:-6] - 1
        if assign_ind:
            aligned_pos = self.structure.transform_pos(pos)
            index = self.sample_area.assign_points(aligned_pos)
            ind: list[list[np.ndarray]] = [[] for _ in range(self.sample_area.num)]
            for i, j in enumerate(index):
                ind[j].append(i)
            self._ind = ind
        sample_w = np.nan * np.ones(self.sample_area.num)
        if self._ind is not None:
            for i in range(len(self._ind)):
                if self._ind[i]:
                    sample_w[i] = np.mean(error[self._ind[i]])
        else:
            raise ValueError("No valid indices found.")
        sample_pos = self.sample_area.pos[~np.isnan(sample_w)]
        sample_w = sample_w[~np.isnan(sample_w)]
        self.sample_pos = sample_pos
        self.sample_w = sample_w
        self.gradient = (batch_fray[-6:-3] - batch_fray[-3:])

    def update_spherical_harmonics(self, lmax=4):
        if self.sample_pos is not None and self.sample_w is not None:
            self.coef = self.spherical_harmonics_dec(self.sample_pos, self.sample_w, lmax=lmax)
        else:
            raise ValueError("No valid sample positions or weights found.")

    def compute_error(self):
        if self.coef is not None:
            return np.abs(self.coef[0])+np.sum(np.abs(self.coef[1]))+np.sum(np.abs(self.coef[2]))
        else:
            raise ValueError("No valid coefficients found.")

    # ------------------ Fitting Steps ------------------

    def fit_angle(self,pos: np.ndarray, num_step: int, rtol: float = 0.001) -> int:
        n = 0
        for _ in range(num_step):
            self.update_spherical_harmonics(lmax=2) # always
            delta = EllipsoidParameterUpdater.update_angle(self.coef,self.parameters,self.gradient)
            delta = np.clip(delta, -0.5, 0.5)
            self.parameters["ang1"] += delta
            self.parameters["ang1"] = (self.parameters["ang1"]%(np.pi))
            self.history["ang1"].append(self.parameters["ang1"])
            self.update_sample(pos) # if center_pos, angle changed
            n += 1
            if abs(delta) < rtol:
                break
        return n

    def fit_center(self, pos: np.ndarray, num_step: int, rtol: float = 0.001) -> int:
        n = 0
        for _ in range(num_step):
            self.update_spherical_harmonics(lmax=2) # always
            delta = EllipsoidParameterUpdater.update_center(self.parameters,self.gradient)
            delta = np.clip(delta, -0.1*self.parameters["a"], 0.1*self.parameters["a"])
            self.parameters["x"],self.parameters["y"],self.parameters["z"] = self.parameters["pos"] + delta
            self.update_sample(pos) # if center_pos, angle changed
            self.history["x"].append(self.parameters["x"])
            self.history["y"].append(self.parameters["y"])
            self.history["z"].append(self.parameters["z"])
            n +=1
            if np.linalg.norm(delta) < rtol*self.parameters["a"]:
                break
        return n

    def fit_a(self, pos: np.ndarray, num_step: int, rtol: float = 0.001) -> int:
        n = 0
        for _ in range(num_step):
            self.update_spherical_harmonics(lmax=2) # always
            delta = EllipsoidParameterUpdater.update_a(self.coef,self.parameters,self.gradient)
            delta = max(min(delta,self.parameters["a"]),-0.5*self.parameters["a"])

            self.parameters["a"] = self.parameters["a"] + delta
            self.update_sample(pos, assign_ind=False) # if center_pos, angle changed
            self.history["a"].append(self.parameters["a"])
            n +=1
            if abs(delta) < rtol*self.parameters["a"]:
                break
        return n

    def fit_eps(self, pos: np.ndarray, num_step: int, rtol: float = 0.001, eps_mode: str = "exp") -> int:
        n = 0
        for _ in range(num_step):

            break_flag = False
            self.update_spherical_harmonics(lmax=2) # always

            delta_eps_ab = EllipsoidParameterUpdater.update_eps_ab(self.coef,self.parameters,self.gradient,mode=eps_mode)
            delta_eps_bc = EllipsoidParameterUpdater.update_eps_bc(self.coef,self.parameters,self.gradient,mode=eps_mode)

            if (delta_eps_ab < rtol*(1-self.parameters["eps_ab"])) or (delta_eps_bc < rtol*(1-self.parameters["eps_bc"])):
                break_flag= True

            if np.isnan(delta_eps_ab) or (self.parameters["eps_ab"] + delta_eps_ab) > 1 or (self.parameters["eps_ab"] + delta_eps_ab) < 0:
                break_flag = True
            else:
                self.parameters["eps_ab"] = self.parameters["eps_ab"] + delta_eps_ab

            if np.isnan(delta_eps_bc) or (self.parameters["eps_bc"] + delta_eps_bc) > 1 or (self.parameters["eps_bc"] + delta_eps_bc) < 0:
                break_flag = True
            else:
                self.parameters["eps_bc"] = self.parameters["eps_bc"] + delta_eps_bc

            self.update_sample(pos, assign_ind=False) # if center_pos, angle changed, assign_ind=True
            n +=1
            self.history["eps_ab"].append(self.parameters["eps_ab"])
            self.history["eps_bc"].append(self.parameters["eps_bc"])
            if break_flag:
                break
        return n

    def fit(self, pos, max_step = 500, ftol = 0.001, eps_mode = "fitted", select_strategy = "max",**kwargs):
        rtol = kwargs.get("rtol", 0.01)


        estimate_cen = np.median(pos,axis=0)
        self.parameters["a"] = np.median(np.linalg.norm(pos - estimate_cen,axis=1))
        self.parameters["x"],self.parameters["y"],self.parameters["z"] = estimate_cen

        self.update_sample(pos) # if center_pos, angle changed
        # if eps , sa changed
        self.update_spherical_harmonics()
        if self.coef is None:
            raise ValueError("Coefficient is not initialized.")
        def term_max():
            if self.coef is None:
                raise ValueError("Coefficient is not initialized.")
            return [np.max(np.abs(self.coef[0])), np.max(np.abs(self.coef[1])), np.max(np.abs(self.coef[2]))]
        def term_mean():
            if self.coef is None:
                raise ValueError("Coefficient is not initialized.")
            return [np.mean(np.abs(self.coef[0])), np.mean(np.abs(self.coef[1])), np.mean(np.abs(self.coef[2]))]
        def term_var():
            if self.coef is None:
                raise ValueError("Coefficient is not initialized.")
            return [np.mean(self.coef[0]**2), np.mean(self.coef[1]**2), np.mean(self.coef[2]**2)]
        if select_strategy == "max":
            self.select_terms = term_max
        elif select_strategy == "mean":
            self.select_terms = term_mean
        elif select_strategy == "var":
            self.select_terms = term_var

        step = 0
        prev_st = None
        prev_error = None
        while step < max_step:
            means = self.select_terms()
            st = int(np.argmax(means))

            if prev_st is not None and st == prev_st:
                curr_error = self.compute_error()
                if prev_error is not None and curr_error >= prev_error:
                    st = int(np.argsort(means)[-2])
            prev_st = st
            prev_error = self.compute_error()

            if st == 0:
                self.fit_a(pos,1,rtol)
            elif st==1:
                self.fit_center(pos,1,rtol)
            elif self.parameters["eps_ab"]>0. and np.abs(self.coef[2][2])> np.abs(self.coef[2][0]):
                self.fit_angle(pos,1,rtol)
            else:
                self.fit_eps(pos,1,rtol,eps_mode)

            step = step+1

            if self.compute_error() < ftol:
                break

    def estimate_error(self, pos, parameters):
        self.parameters.set_value(**parameters)
        self.update_sample(pos)
        self.update_spherical_harmonics()
        error = {}
        error["eps_ab"] = EllipsoidParameterUpdater.update_eps_ab(self.coef,self.parameters,self.gradient,"fitted")
        error["eps_bc"] = EllipsoidParameterUpdater.update_eps_bc(self.coef,self.parameters,self.gradient,"fitted")
        error["a"] = EllipsoidParameterUpdater.update_a(self.coef,self.parameters,self.gradient)
        return error

    @property
    def num_steps(self):
        return np.sum([len(self.history[i]) for i in ["x","a","eps_ab"]])
