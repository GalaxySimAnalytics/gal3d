"""
This is a spherical harmonics decomposition iterator for fitting generalized ellipsoids to point clouds.
But it is still in experimental stage.
"""
from collections import defaultdict

import numpy as np

from gal3d.shape import StructureCore

from .spherical_vector import SphVector


class SphericalDecIterator:

    def __init__(self, n_sample):
        self.sample_area = SphVector(n_sample=n_sample)
        self.structure = StructureCore("ShiftEuler","Ellipsoid_S")
        self.parameters = self.structure.parameters
        self.parameters["eps_ab"] = 0.01
        self.parameters["eps_bc"] = 0.01
        self.history = defaultdict(list)



    def spherical_harmonics_dec(self, pos: np.ndarray, w: np.ndarray, lmax: int = 4) -> None:
        # Compute the spherical harmonics for the given positions and weights
        c0 = 0.5/np.sqrt(np.pi)
        coef: dict[int, np.ndarray] = {}
        self.coef = coef
        r = np.sqrt(np.sum(pos*pos,axis=1))
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]

        coef[0] = np.array(c0*np.sum(w))
        if lmax == 0:
            return

        coef[1] = np.sqrt(3)*c0*np.sum(pos.T/r*w,axis=1)   # x, y, z
        if lmax ==1:
            return

        # eps_bc: use y*y - z*z ??
        coef[2] = np.sqrt(15)*c0*np.sum(w*np.array([(x*x-y*y), 2*z*z-x*x-y*y, x*y,y*z,z*x])/(r*r),axis=1) # eps_ab, eps_bc, xy, yz, zx
        coef[2][:2] = coef[2][:2]/2

        if lmax == 2:
            return


    def update_a(self, parameters, f_d, fixed_f_d = False):

        if fixed_f_d:
            delta = self.coef[0]/f_d[0]

        else:
            delta = parameters["a"]*(np.exp(self.coef[0]/(parameters["a"]*f_d[0])) - 1)

        delta_eps_ab = self.update_eps_ab(parameters, f_d)

        if (self.parameters["eps_ab"] + delta_eps_ab)<0:
            delta = delta - parameters["a"]*delta_eps_ab/(1-self.parameters["eps_ab"]-delta_eps_ab)
        return delta

    def update_center_x(self, parameters, f_d):
        delta = self.coef[1][0]/f_d[0]
        return delta

    def update_center_y(self, parameters, f_d):
        delta = self.coef[1][1]/f_d[1]
        return delta

    def update_center_z(self, parameters, f_d):
        delta = self.coef[1][2]/f_d[2]
        return delta

    def update_center(self, parameters, f_d):
        delta = self.coef[1]/f_d
        return delta

    def update_eps_ab(self, parameters, f_d, fixed_f_d = False):
        if fixed_f_d:
            delta = self.coef[2][0]/f_d[1]/parameters["a"]
            return delta
        else:
            delta = 2*(np.exp(self.coef[2][0]/((1-parameters["eps_ab"])*parameters["a"])/f_d[1])-1) * (1-parameters["eps_ab"])
            return delta

    def update_eps_bc(self, parameters, f_d, fixed_f_d = False):
        if fixed_f_d:
            delta = -self.coef[2][1]/f_d[2]/(parameters["a"]*(1-parameters["eps_ab"]))
            return delta
        else:
            delta = 2*(np.exp(-self.coef[2][1]/((1-parameters["eps_bc"])*(parameters["a"]*(1-parameters["eps_ab"])))/f_d[2])-1) * (1-parameters["eps_bc"])
            return delta

    def update_angle(self, parameters, f_d):
        eps_ab = parameters["eps_ab"]
        a = parameters["a"]
        dx = f_d[0]
        delta = self.coef[2][2]/dx

        c0 = delta*(2-eps_ab*eps_ab)/(2+eps_ab*eps_ab-2*eps_ab)**(3/2)
        c1 = (2-eps_ab)*(2*a)*(1-eps_ab)*(np.sqrt(2-eps_ab*eps_ab))
        c2 = delta*eps_ab*(2+eps_ab*eps_ab-2*eps_ab)**(3/2)
        if eps_ab==0:
            delta_ang = 0.
        else:
            delta_ang = - c0/(c1+c2)/eps_ab
        return delta_ang

    def update_sample(self, pos, assign_ind=True):

        eye = np.eye(3)
        batch_pos = np.concatenate([pos, 2*eye + self.parameters["pos"], eye + self.parameters["pos"]], axis=0)
        batch_fray = self.structure.f_ray_d(batch_pos)

        # use -1 or use ln?
        error = batch_fray[:-6] - 1
        #error = np.log(self.structure.f_ray_d(pos))

        if assign_ind:
            aligned_pos = self.structure.transform_pos(pos)
            index = self.sample_area.assign_points(aligned_pos)

            ind: list[list[int]] = [[] for _ in range(self.sample_area.num)]
            for i, j in enumerate(index):
                ind[j].append(i)

            self._ind = ind


        sample_w = np.nan*np.ones(self.sample_area.num)
        for i in range(len(self._ind)):
            if self._ind[i]:
                sample_w[i] = np.mean(error[self._ind[i]])

        sample_pos = self.sample_area.pos[~np.isnan(sample_w)]
        sample_w = sample_w[~np.isnan(sample_w)]

        self.sample_pos = sample_pos
        self.sample_w = sample_w

        self.gradient = (batch_fray[-6:-3] - batch_fray[-3:]) * len(self.sample_w)

    def update_spherical_harmonics(self, lmax=4):
        self.spherical_harmonics_dec(self.sample_pos,self.sample_w,lmax=lmax)


    def compute_error(self):
        return self.coef[0]+np.sum(np.abs(self.coef[1]))+np.sum(np.abs(self.coef[2]))

    def fit_angle(self,pos: np.ndarray, num_step: int, rtol: float = 0.001) -> int:
        n = 0
        for _ in range(num_step):
            self.update_spherical_harmonics(lmax=2) # always
            delta = self.update_angle(self.parameters,self.gradient)
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
            delta = self.update_center(self.parameters,self.gradient)
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
            delta = self.update_a(self.parameters,self.gradient)
            if (self.parameters["a"] + delta < 0):
                break

            delta = min(delta, self.parameters["a"])
            self.parameters["a"] = self.parameters["a"] + delta
            self.update_sample(pos, assign_ind=False) # if center_pos, angle changed
            self.history["a"].append(self.parameters["a"])
            n +=1
            if abs(delta) < rtol*self.parameters["a"]:
                break
        return n

    def fit_eps(self, pos: np.ndarray, num_step: int, rtol: float = 0.001) -> int:
        n = 0
        for _ in range(num_step):

            break_flag = False
            self.update_spherical_harmonics(lmax=2) # always

            delta_eps_ab = self.update_eps_ab(self.parameters,self.gradient)
            delta_eps_bc = self.update_eps_bc(self.parameters,self.gradient)

            if (delta_eps_ab < rtol*(1-self.parameters["eps_ab"])) or (delta_eps_bc < rtol*(1-self.parameters["eps_bc"])):
                break_flag= True

            if (self.parameters["eps_ab"] + delta_eps_ab) > 1 or (self.parameters["eps_ab"] + delta_eps_ab) < 0:
                break_flag = True
            else:
                self.parameters["eps_ab"] = self.parameters["eps_ab"] + delta_eps_ab

            if (self.parameters["eps_bc"] + delta_eps_bc) > 1 or (self.parameters["eps_bc"] + delta_eps_bc) < 0:
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

    def fit(self, pos, max_step = 500, ftol = 0.001, **kwargs):
        rtol = kwargs.get("rtol", 0.01)


        estimate_cen = np.median(pos,axis=0)
        self.parameters["a"] = np.median(np.linalg.norm(pos - estimate_cen,axis=1))
        self.parameters["x"],self.parameters["y"],self.parameters["z"] = estimate_cen


        self.update_sample(pos) # if center_pos, angle changed
        # if eps , sa changed
        self.update_spherical_harmonics()
        step = 0
        n_round = 0
        while step < max_step:
            st = int(np.argmax([np.mean(self.coef[0]**2),np.mean(self.coef[1]**2),np.mean(self.coef[2]**2)]))
            if st == 0:
                self.fit_a(pos,1,rtol)
            elif st==1:
                self.fit_center(pos,1,rtol)
            elif self.parameters["eps_ab"]>0. and np.abs(self.coef[2][2])> np.abs(self.coef[2][0]):
                self.fit_angle(pos,1,rtol)
            else:
                self.fit_eps(pos,1,rtol)

            step = step+1

            if self.compute_error() < ftol:
                break
            n_round += 1

    @property
    def num_steps(self):
        return np.sum([len(self.history[i]) for i in ["x","a","eps_ab"]])
